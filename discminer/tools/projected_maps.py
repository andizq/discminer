"""Prepare projected model fields and velocity overlays without synthesising cubes."""

from collections.abc import Mapping

import numpy as np


PROJECTED_ATTRIBUTES = ('velocity', 'intensity', 'linewidth', 'lineslope')


def get_projected_map(model, attribute):
    """Extract a cached field; preserve the synthesis properties and NaN masks."""
    if attribute not in PROJECTED_ATTRIBUTES:
        raise ValueError('attribute must be one of %s.' % (PROJECTED_ATTRIBUTES,))
    if not hasattr(model, 'props'):
        raise ValueError('Run make_model() on a prototype model before requesting projected maps.')
    attributes = getattr(model, '_projected_attributes', PROJECTED_ATTRIBUTES)
    if attribute not in attributes:
        raise ValueError('No cached projected %s field is available.' % attribute)
    field = model.props[attributes.index(attribute)]
    maps = {}
    for side in ('upper', 'lower'):
        if getattr(model, 'subpixels', False):
            values = np.sum(np.array([part[side] for part in field]), axis=0)
            values = values * (model.sub_dA / model.pix_dA)
        else:
            values = field[side]
        values = np.asarray(values)
        if values.ndim == 0 and hasattr(model, 'datacube'):
            values = np.broadcast_to(values, model.datacube.data.shape[1:])
        if values.ndim != 2:
            raise ValueError('The projected %s field must contain 2D maps.' % attribute)
        maps[side] = np.array(values, copy=True)
    return maps


def validate_velocity_maps(velocity2d, shape):
    """Validate upper/lower velocity maps on a cube's spatial pixel grid."""
    if velocity2d is None:
        return None
    if (not isinstance(velocity2d, Mapping) or not velocity2d
            or set(velocity2d) - {'upper', 'lower'}):
        raise ValueError('velocity2d must map upper and/or lower to 2D projected velocities in km/s.')
    maps = {}
    for side, values in velocity2d.items():
        array = np.ma.asarray(values, dtype=float)
        if array.shape != shape or min(shape) < 2:
            raise ValueError('velocity2d[%r] must match the cube spatial shape %s (at least 2x2).' % (side, shape))
        maps[side] = np.ma.masked_invalid(array, copy=True)
    return maps


def prepare_velocity_contours(velocity2d, contours_from, shape):
    """Resolve explicit maps or the legacy model/array input once per figure."""
    if velocity2d is not None and contours_from is not None:
        raise ValueError('Pass either velocity2d or contours_from, not both.')
    if velocity2d is None and contours_from is None:
        return []
    if velocity2d is None:
        if hasattr(contours_from, 'get_projected_map'):
            velocity2d = contours_from.get_projected_map('velocity')
        elif hasattr(contours_from, 'props'):
            velocity2d = get_projected_map(contours_from, 'velocity')
        elif isinstance(contours_from, Mapping):
            velocity2d = contours_from
        else:
            # Preserve contours_from's 2D or (ny, nx, nfields) array convention.
            arrays = np.moveaxis(np.atleast_3d(contours_from), -1, 0)
            maps = []
            for array in arrays:
                field = validate_velocity_maps({'upper': array}, shape)['upper']
                maps.append((field, '-'))
            return maps
    maps = validate_velocity_maps(velocity2d, shape)
    return [(maps[side], style) for side, style in [('upper', '-'), ('lower', '--')]
            if side in maps]
