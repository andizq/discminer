import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay


def _coordinate_points(coord):
    """Return coordinates as an ``(npoints, ndim)`` array."""
    if isinstance(coord, (tuple, list)):
        arrays = [np.asarray(item) for item in coord]
        if not arrays:
            raise ValueError("At least one coordinate array is required")
        if any(item.shape != arrays[0].shape for item in arrays[1:]):
            raise ValueError("Coordinate arrays must all have the same shape")
        return np.column_stack([item.ravel() for item in arrays])

    points = np.asarray(coord)
    if points.ndim != 2:
        raise ValueError(
            "Coordinates must be an (npoints, ndim) array or a tuple of arrays"
        )
    return points


def _target_points_and_shape(coord):
    if isinstance(coord, (tuple, list)):
        arrays = [np.asarray(item) for item in coord]
        points = _coordinate_points(arrays)
        return points, arrays[0].shape

    points = np.asarray(coord)
    if points.ndim < 2:
        raise ValueError(
            "Target coordinates must have a final coordinate dimension"
        )
    return points.reshape(-1, points.shape[-1]), points.shape[:-1]


def get_griddata_sparse(old_coord, new_coord):
    """
    Build a reusable piecewise-linear scattered-grid interpolator.

    The Delaunay triangulation and barycentric weights depend only on the
    coordinates, so they are computed once and stored in a sparse matrix.
    Calling the returned function only performs a sparse matrix-vector
    multiplication.

    Points outside the convex hull retain the old discminer behaviour
    and are returned as zero.
    """
    source_shape = (
        np.asarray(old_coord[0]).shape
        if isinstance(old_coord, (tuple, list))
        else None
    )
    source_points = _coordinate_points(old_coord)
    target_points, target_shape = _target_points_and_shape(new_coord)

    ndim = source_points.shape[1]
    if target_points.shape[1] != ndim:
        raise ValueError("Source and target coordinates have different dimensions")

    tri = Delaunay(source_points)
    simplex_ids = tri.find_simplex(target_points)

    valid = simplex_ids >= 0
    valid_simplex = simplex_ids[valid]

    transform = tri.transform[valid_simplex]
    delta = target_points[valid] - transform[:, ndim, :]
    first_weights = np.einsum(
        "nij,nj->ni",
        transform[:, :ndim, :],
        delta,
        optimize=True,
    )
    weights = np.concatenate(
        (
            first_weights,
            1.0 - first_weights.sum(axis=1, keepdims=True),
        ),
        axis=1,
    )

    rows = np.repeat(np.flatnonzero(valid), ndim + 1)
    cols = tri.simplices[valid_simplex].ravel()
    interpolation_matrix = sp.csr_matrix(
        (weights.ravel(), (rows, cols)),
        shape=(target_points.shape[0], source_points.shape[0]),
    )

    def interpolate(values):
        values = np.asarray(values)
        nsource = source_points.shape[0]

        # Preserve the historical behaviour for a single field supplied with
        # the same shape as the source coordinate grid.
        single_field = (
            values.ndim == 1
            or (source_shape is not None and values.shape == source_shape)
        )
        if single_field:
            values = values.reshape(nsource, 1)
            field_shape = ()
        else:
            if values.ndim < 2 or values.shape[0] != nsource:
                raise ValueError(
                    f"Expected {nsource} source rows, got shape {values.shape}"
                )
            field_shape = values.shape[1:]
            values = values.reshape(nsource, -1)

        projected = interpolation_matrix @ values
        projected = projected.reshape(target_shape + field_shape)
        if single_field:
            return projected.reshape(target_shape)
        return projected

    return interpolate
