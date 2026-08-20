"""Morphological measurements for PCA eigenimages and their ACFs."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table
from matplotlib.lines import Line2D
from matplotlib.path import Path as MatplotlibPath
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from skimage.measure import (
    EllipseModel,
    euler_number,
    find_contours,
    perimeter_crofton,
)

from discminer.plottools import use_discminer_style


_PHASE_MINIMUM_PEAK_TOTAL = 0.10
_PHASE_MAXIMUM_MODE_ENTROPY = 0.85
_SLOPE_MINIMUM_PHASE_COHERENCE = 0.70
_SLOPE_MINIMUM_PHASE_RINGS = 10
_CONDITIONAL_MARKER_ALPHA = 0.4


def _empty_acf_ellipse_metrics():
    return {
        "acf_semimajor_pix": np.nan,
        "acf_semiminor_pix": np.nan,
        "acf_axis_ratio": np.nan,
        "acf_pa_deg": np.nan,
        "acf_center_offset_pix": np.nan,
        "acf_ellipse_residual": np.nan,
        "acf_contour_points": 0,
    }


def _lag_axis(size):
    return np.fft.fftshift(np.fft.fftfreq(size) * size)


def acf_ellipse_metrics(autocorrelation, level=np.exp(-1)):
    """Measure the central contour of a two-dimensional ACF.

    The axis ratio is the fitted semi-minor axis divided by the semi-major
    axis. The ellipse residual is the RMS orthogonal distance of the contour
    points from the fitted ellipse, normalized by the semi-major axis.

    Parameters
    ----------
    autocorrelation : array-like
        Two-dimensional spatial autocorrelation image with zero lag at the
        FFT-shifted centre.
    level : float, optional
        Fraction of the ACF peak used for the contour. The default is 1/e.

    Returns
    -------
    dict
        Ellipse geometry and goodness-of-fit measurements. Invalid fits are
        represented by NaNs and zero contour points.
    """

    values = np.asarray(autocorrelation, dtype=float)
    if values.ndim != 2:
        raise ValueError("autocorrelation must be a two-dimensional array")
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must be finite and between zero and one")

    output = _empty_acf_ellipse_metrics()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return output
    peak = np.nanmax(finite)
    if not np.isfinite(peak) or peak <= 0.0:
        return output

    normalized = values / peak
    fill_value = min(float(np.nanmin(normalized)), level - 1.0)
    normalized = np.where(np.isfinite(normalized), normalized, fill_value)

    ny, nx = normalized.shape
    ylag = _lag_axis(ny)
    xlag = _lag_axis(nx)
    contours = find_contours(normalized, level)

    candidates = []
    for contour in contours:
        xvalues = np.interp(contour[:, 1], np.arange(nx), xlag)
        yvalues = np.interp(contour[:, 0], np.arange(ny), ylag)
        points = np.column_stack((xvalues, yvalues))
        if MatplotlibPath(points).contains_point((0.0, 0.0)):
            candidates.append(points)

    if not candidates:
        return output

    # A nested contour can occur for a structured ACF. The longest contour
    # enclosing zero lag is the most stable representation of the central
    # correlation region.
    points = max(candidates, key=len)
    ellipse = EllipseModel()
    if not ellipse.estimate(points):
        return output

    xcenter, ycenter, axis0, axis1, theta = ellipse.params
    parameters = np.asarray(ellipse.params, dtype=float)
    if np.any(~np.isfinite(parameters)) or axis0 <= 0.0 or axis1 <= 0.0:
        return output

    if axis0 >= axis1:
        semimajor = axis0
        semiminor = axis1
        major_theta = theta
    else:
        semimajor = axis1
        semiminor = axis0
        major_theta = theta + 0.5 * np.pi

    residuals = np.asarray(ellipse.residuals(points), dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    ellipse_residual = (
        np.sqrt(np.mean(residuals**2)) / semimajor
        if residuals.size
        else np.nan
    )

    output.update(
        {
            "acf_semimajor_pix": float(semimajor),
            "acf_semiminor_pix": float(semiminor),
            "acf_axis_ratio": float(semiminor / semimajor),
            "acf_pa_deg": float(np.degrees(major_theta) % 180.0),
            "acf_center_offset_pix": float(
                np.hypot(xcenter, ycenter)
            ),
            "acf_ellipse_residual": float(ellipse_residual),
            "acf_contour_points": int(points.shape[0]),
        }
    )
    return output


def _polar_sampling_grid(
    shape,
    center=None,
    n_azimuth=360,
    minimum_radius=2.0,
    ring_geometry=None,
):
    if len(shape) != 2:
        raise ValueError("shape must describe a two-dimensional image")
    if n_azimuth < 8:
        raise ValueError("n_azimuth must be at least 8")

    ny, nx = shape
    if center is None:
        ycenter = 0.5 * (ny - 1)
        xcenter = 0.5 * (nx - 1)
    else:
        ycenter, xcenter = (float(value) for value in center)
    if not (0.0 <= ycenter < ny and 0.0 <= xcenter < nx):
        raise ValueError("center must lie inside the image")

    angles = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
    if ring_geometry is not None:
        return _disc_plane_sampling_grid(
            shape,
            center=(ycenter, xcenter),
            angles=angles,
            minimum_radius=minimum_radius,
            ring_geometry=ring_geometry,
        )

    maximum_radius = min(
        ycenter,
        xcenter,
        ny - 1 - ycenter,
        nx - 1 - xcenter,
    )
    radii = np.arange(float(minimum_radius), maximum_radius + 0.5, 1.0)
    coordinates = np.asarray(
        [
            ycenter + radii[:, np.newaxis] * np.sin(angles),
            xcenter + radii[:, np.newaxis] * np.cos(angles),
        ]
    )
    return radii, angles, coordinates


def _disc_plane_sampling_grid(
    shape,
    center,
    angles,
    minimum_radius,
    ring_geometry,
):
    """Project uniformly sampled disc-plane annuli onto image pixels."""

    ny, nx = shape
    ycenter, xcenter = center
    pixel_x_au = float(ring_geometry["pixel_x_au"])
    pixel_y_au = float(ring_geometry["pixel_y_au"])
    radial_step_au = float(ring_geometry["radial_step_au"])
    if min(pixel_x_au, pixel_y_au, radial_step_au) <= 0.0:
        raise ValueError("disc-plane pixel scales must be positive")

    x_extent_au = max(xcenter, nx - 1 - xcenter) * pixel_x_au
    y_extent_au = max(ycenter, ny - 1 - ycenter) * pixel_y_au
    maximum_radius = np.hypot(x_extent_au, y_extent_au) / radial_step_au
    radii = np.arange(float(minimum_radius), maximum_radius + 0.5, 1.0)
    if radii.size == 0:
        return radii, angles, np.empty((2, 0, angles.size))

    radius_au = radii[:, np.newaxis] * radial_step_au
    azimuth = angles[np.newaxis, :]
    surface_function = ring_geometry.get("surface_function")
    if surface_function is None:
        height_au = np.zeros_like(radius_au)
    else:
        height_au = surface_function(
            {"R": radius_au * u.au.to(u.m)},
            **ring_geometry["surface_parameters"],
        ) / u.au.to(u.m)

    from discminer.grid import GridTools

    x_sky_au, y_sky_au, _ = GridTools.get_sky_from_disc_coords(
        radius_au,
        azimuth,
        height_au,
        ring_geometry["inclination_rad"],
        ring_geometry["position_angle_rad"],
        ring_geometry["center_x_au"],
        ring_geometry["center_y_au"],
    )
    coordinates = np.asarray(
        (
            ycenter + y_sky_au / pixel_y_au,
            xcenter + x_sky_au / pixel_x_au,
        )
    )
    complete = np.all(
        (coordinates[0] >= 0.0)
        & (coordinates[0] <= ny - 1)
        & (coordinates[1] >= 0.0)
        & (coordinates[1] <= nx - 1),
        axis=1,
    )
    return radii[complete], angles, coordinates[:, complete]


def load_disc_ring_geometry(parfile, source_header, surface="upper"):
    """Load Rail-compatible disc geometry for projected annular sampling."""

    if surface not in {"upper", "lower", "midplane"}:
        raise ValueError("surface must be upper, lower, or midplane")
    parfile = Path(parfile)
    if not parfile.exists():
        raise FileNotFoundError(f"DiscMiner parfile does not exist: {parfile}")
    parfile = parfile.resolve()

    from discminer import cart
    from discminer.mining_utils import load_parfile

    metadata, parameters, _ = load_parfile(parfile=str(parfile))
    orientation = parameters["orientation"]
    distance_pc = float(metadata["dpc"])
    unit_x = source_header.get("CUNIT1", "deg") or "deg"
    unit_y = source_header.get("CUNIT2", "deg") or "deg"
    pixel_x = (
        abs(float(source_header["CDELT1"]))
        * u.Unit(unit_x)
    ).to_value(u.arcsec)
    pixel_y = (
        abs(float(source_header["CDELT2"]))
        * u.Unit(unit_y)
    ).to_value(u.arcsec)
    pixel_x_au = pixel_x * distance_pc
    pixel_y_au = pixel_y * distance_pc

    surface_function = None
    surface_parameters = {}
    surface_model = "midplane"
    raw_model_kinds = metadata.get("kind", [])
    if isinstance(raw_model_kinds, str):
        raw_model_kinds = [raw_model_kinds]
    model_kinds = set(raw_model_kinds)
    if surface != "midplane":
        surface_parameters = dict(parameters[f"height_{surface}"])
        if "surf2pwl" in model_kinds:
            surface_function = getattr(cart, f"z_{surface}_powerlaw")
            surface_model = "powerlaw"
        elif surface == "upper" and "surfirregular" in model_kinds:
            surface_function = cart.z_upper_irregular
            surface_model = "irregular"
            if isinstance(surface_parameters.get("z0"), str):
                surface_parameters["z0"] = str(
                    parfile.parent / surface_parameters["z0"]
                )
        else:
            surface_function = getattr(cart, f"z_{surface}_exp_tapered")
            surface_model = "exp_tapered"

    return {
        "mode": "disc_plane",
        "parfile": str(parfile),
        "surface": surface,
        "surface_model": surface_model,
        "surface_function": surface_function,
        "surface_parameters": surface_parameters,
        "inclination_rad": float(orientation["incl"]),
        "position_angle_rad": float(orientation["PA"]),
        "center_x_au": float(orientation.get("xc", 0.0)),
        "center_y_au": float(orientation.get("yc", 0.0)),
        "distance_pc": distance_pc,
        "pixel_x_au": float(pixel_x_au),
        "pixel_y_au": float(pixel_y_au),
        "radial_step_au": float(np.sqrt(pixel_x_au * pixel_y_au)),
    }


def acf_multipole_metrics(
    autocorrelation,
    level=np.exp(-1),
    n_azimuth=360,
):
    """Measure scale-dependent even multipoles of a spatial ACF.

    At each radius, ``Q_m`` is the modulus of the azimuthal Fourier
    coefficient divided by the mean absolute ACF on the ring. The reported
    maximum is restricted to the central correlation core, where the
    ring-mean absolute ACF is at least ``level`` times the ACF peak.
    """

    output = {
        "acf_max_q2": np.nan,
        "acf_q2_radius_pix": np.nan,
        "acf_max_q4": np.nan,
        "acf_q4_radius_pix": np.nan,
        "acf_multipole_radius_limit_pix": np.nan,
    }
    values = np.asarray(autocorrelation, dtype=float)
    if values.ndim != 2:
        raise ValueError("autocorrelation must be a two-dimensional array")
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must be finite and between zero and one")

    finite = np.isfinite(values)
    if not np.any(finite):
        return output
    peak = np.nanmax(values)
    if not np.isfinite(peak) or peak <= 0.0:
        return output

    radii, angles, coordinates = _polar_sampling_grid(
        values.shape,
        center=(values.shape[0] // 2, values.shape[1] // 2),
        n_azimuth=n_azimuth,
    )
    if radii.size == 0:
        return output

    samples = map_coordinates(
        np.where(finite, values, 0.0),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    valid_samples = map_coordinates(
        finite.astype(float),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    complete = np.mean(valid_samples > 0.999, axis=1) >= 0.95
    ring_amplitude = np.mean(np.abs(samples), axis=1)
    core = complete & (ring_amplitude >= level * peak)
    if not np.any(core):
        return output

    output["acf_multipole_radius_limit_pix"] = float(radii[core][-1])
    for order in (2, 4):
        phase = np.exp(-1j * order * angles)
        coefficient = np.abs(np.mean(samples * phase, axis=1))
        ratio = np.divide(
            coefficient,
            ring_amplitude,
            out=np.full(radii.shape, np.nan, dtype=float),
            where=ring_amplitude > 0.0,
        )
        core_indices = np.flatnonzero(core & np.isfinite(ratio))
        if core_indices.size:
            peak_index = core_indices[np.argmax(ratio[core_indices])]
            output[f"acf_max_q{order}"] = float(ratio[peak_index])
            output[f"acf_q{order}_radius_pix"] = float(
                radii[peak_index]
            )
    return output


def _azimuthal_mode_profile(
    image,
    mode=2,
    mask=None,
    center=None,
    n_azimuth=360,
    minimum_coverage=0.75,
    minimum_radius=2.0,
    ring_geometry=None,
):
    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    if mode < 1:
        raise ValueError("mode must be positive")
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in the interval (0, 1]")

    finite = np.isfinite(values)
    if mask is None:
        support = finite
    else:
        support = np.asarray(mask, dtype=bool).copy()
        if support.shape != values.shape:
            raise ValueError("mask must have the same shape as image")
        support &= finite

    if not np.any(support):
        return None

    radii, angles, coordinates = _polar_sampling_grid(
        values.shape,
        center=center,
        n_azimuth=n_azimuth,
        minimum_radius=minimum_radius,
        ring_geometry=ring_geometry,
    )
    if radii.size == 0:
        return None

    samples = map_coordinates(
        np.where(finite, values, 0.0),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    sampled_support = map_coordinates(
        support.astype(float),
        coordinates,
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5

    profile = {
        "radius": [],
        "coverage": [],
        "coefficient": [],
        "explained_power": [],
        "nonaxisymmetric_power": [],
        "total_power": [],
    }
    for radius, ring, valid in zip(radii, samples, sampled_support):
        coverage = np.mean(valid)
        if coverage < minimum_coverage:
            continue
        ring = ring[valid]
        ring_angles = angles[valid]
        if ring.size < max(8, 2 * mode + 3):
            continue

        design = np.column_stack(
            (
                np.ones(ring.size),
                np.cos(mode * ring_angles),
                np.sin(mode * ring_angles),
            )
        )
        coefficients = np.linalg.lstsq(design, ring, rcond=None)[0]
        residual_constant = ring - np.mean(ring)
        residual_mode = ring - design @ coefficients
        constant_sse = float(np.sum(residual_constant**2))
        mode_sse = float(np.sum(residual_mode**2))
        ring_weight = float(radius * coverage / ring.size)
        profile["radius"].append(radius)
        profile["coverage"].append(coverage)
        profile["coefficient"].append(
            coefficients[1] - 1j * coefficients[2]
        )
        profile["explained_power"].append(
            ring_weight * max(0.0, constant_sse - mode_sse)
        )
        profile["nonaxisymmetric_power"].append(
            ring_weight * constant_sse
        )
        profile["total_power"].append(
            ring_weight * float(np.sum(ring**2))
        )

    if not profile["radius"]:
        return None
    return {
        key: np.asarray(values)
        for key, values in profile.items()
    }


def azimuthal_mode_fraction(
    image,
    mode=2,
    mask=None,
    center=None,
    n_azimuth=360,
    minimum_coverage=0.75,
    minimum_radius=2.0,
    ring_geometry=None,
):
    """Return total and non-axisymmetric power fractions for one mode.

    A constant plus sine/cosine pair is fitted independently on every radial
    ring. Allowing the phase to vary with radius keeps this strength statistic
    separate from radial phase coherence.
    """

    output = {
        f"eigenimage_f_m{mode}_total": np.nan,
        f"eigenimage_f_m{mode}_nonaxisymmetric": np.nan,
        f"eigenimage_m{mode}_rings": 0,
        f"eigenimage_m{mode}_max_radius_pix": np.nan,
    }
    profile = _azimuthal_mode_profile(
        image,
        mode=mode,
        mask=mask,
        center=center,
        n_azimuth=n_azimuth,
        minimum_coverage=minimum_coverage,
        minimum_radius=minimum_radius,
        ring_geometry=ring_geometry,
    )
    if profile is None:
        return output

    explained_power = np.sum(profile["explained_power"])
    nonaxisymmetric_power = np.sum(profile["nonaxisymmetric_power"])
    total_power = np.sum(profile["total_power"])
    nonaxisymmetric_fraction = (
        explained_power / nonaxisymmetric_power
        if nonaxisymmetric_power > 0.0
        else 0.0
    )
    total_fraction = (
        explained_power / total_power if total_power > 0.0 else 0.0
    )
    output[f"eigenimage_f_m{mode}_total"] = float(
        np.clip(total_fraction, 0.0, 1.0)
    )
    output[f"eigenimage_f_m{mode}_nonaxisymmetric"] = float(
        np.clip(nonaxisymmetric_fraction, 0.0, 1.0)
    )
    output[f"eigenimage_m{mode}_rings"] = profile["radius"].size
    output[f"eigenimage_m{mode}_max_radius_pix"] = float(
        profile["radius"][-1]
    )
    return output


def _axisymmetric_ring_profile(
    image,
    mask=None,
    center=None,
    n_azimuth=360,
    minimum_coverage=0.75,
    minimum_radius=2.0,
    ring_geometry=None,
):
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in the interval (0, 1]")

    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    finite = np.isfinite(values)
    if mask is None:
        support = finite
    else:
        support = np.asarray(mask, dtype=bool).copy()
        if support.shape != values.shape:
            raise ValueError("mask must have the same shape as image")
        support &= finite

    if not np.any(support):
        return None

    radii, angles, coordinates = _polar_sampling_grid(
        values.shape,
        center=center,
        n_azimuth=n_azimuth,
        minimum_radius=minimum_radius,
        ring_geometry=ring_geometry,
    )
    if radii.size == 0:
        return None

    samples = map_coordinates(
        np.where(finite, values, 0.0),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    sampled_support = map_coordinates(
        support.astype(float),
        coordinates,
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5

    axisymmetric_power = 0.0
    total_power = 0.0
    accepted_indices = []
    accepted_radii = []
    ring_means = []
    for index, (radius, ring, valid) in enumerate(
        zip(radii, samples, sampled_support)
    ):
        coverage = np.mean(valid)
        if coverage < minimum_coverage:
            continue
        ring = ring[valid]
        if ring.size < 8:
            continue
        ring_weight = float(radius * coverage)
        ring_mean = float(np.mean(ring))
        axisymmetric_power += ring_weight * ring_mean**2
        total_power += ring_weight * float(np.mean(ring**2))
        accepted_indices.append(index)
        accepted_radii.append(radius)
        ring_means.append(ring_mean)

    if not accepted_radii:
        return None

    accepted_indices = np.asarray(accepted_indices, dtype=int)
    return {
        "values": values,
        "support": support,
        "angles": angles,
        "coordinates": coordinates[:, accepted_indices],
        "radii": np.asarray(accepted_radii, dtype=float),
        "means": np.asarray(ring_means, dtype=float),
        "axisymmetric_power": float(axisymmetric_power),
        "total_power": float(total_power),
    }


def azimuthal_axisymmetric_fraction(
    image,
    mask=None,
    center=None,
    n_azimuth=360,
    minimum_coverage=0.75,
    minimum_radius=2.0,
    ring_geometry=None,
):
    """Return the area-weighted axisymmetric eigenimage power fraction.

    The azimuthal mean on every radial ring is the ``m=0`` contribution.
    Squaring each ring mean before integrating prevents cancellation between
    positive and negative radial zones and makes the statistic invariant to
    the arbitrary sign of a PCA eigenimage.
    """

    output = {
        "eigenimage_f_m0": np.nan,
        "eigenimage_m0_rings": 0,
        "eigenimage_m0_max_radius_pix": np.nan,
    }
    profile = _axisymmetric_ring_profile(
        image,
        mask=mask,
        center=center,
        n_azimuth=n_azimuth,
        minimum_coverage=minimum_coverage,
        minimum_radius=minimum_radius,
        ring_geometry=ring_geometry,
    )
    if profile is None:
        return output
    if profile["total_power"] <= 0.0:
        return output

    output.update(
        {
            "eigenimage_f_m0": float(
                np.clip(
                    profile["axisymmetric_power"] / profile["total_power"],
                    0.0,
                    1.0,
                )
            ),
            "eigenimage_m0_rings": profile["radii"].size,
            "eigenimage_m0_max_radius_pix": float(profile["radii"][-1]),
        }
    )
    return output


def subtract_axisymmetric_mode(
    image,
    mask=None,
    center=None,
    n_azimuth=360,
    minimum_coverage=0.75,
    minimum_radius=2.0,
    ring_geometry=None,
):
    """Return the projected ``m=0`` eigenimage and its residual.

    Ring means are evaluated with the same sampling and coverage rules as
    :func:`azimuthal_axisymmetric_fraction`. The reconstructed field and
    residual are NaN outside the valid image support or the outermost accepted
    annulus.
    """

    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    axisymmetric = np.full(values.shape, np.nan, dtype=float)
    residual = np.full(values.shape, np.nan, dtype=float)
    profile = _axisymmetric_ring_profile(
        values,
        mask=mask,
        center=center,
        n_azimuth=n_azimuth,
        minimum_coverage=minimum_coverage,
        minimum_radius=minimum_radius,
        ring_geometry=ring_geometry,
    )
    if profile is None:
        return axisymmetric, residual

    ny, nx = values.shape
    ygrid, xgrid = np.indices(values.shape, dtype=float)
    if center is None:
        ycenter = 0.5 * (ny - 1)
        xcenter = 0.5 * (nx - 1)
    else:
        ycenter, xcenter = (float(value) for value in center)

    if ring_geometry is None:
        pixel_radius = np.hypot(xgrid - xcenter, ygrid - ycenter)
        axisymmetric = np.interp(
            pixel_radius,
            profile["radii"],
            profile["means"],
            left=profile["means"][0],
            right=np.nan,
        )
    else:
        coordinates = profile["coordinates"]
        points = np.column_stack(
            (coordinates[0].ravel(), coordinates[1].ravel())
        )
        ring_values = np.repeat(
            profile["means"],
            profile["angles"].size,
        )
        if profile["radii"].size == 1:
            axisymmetric.fill(profile["means"][0])
        else:
            axisymmetric = griddata(
                points,
                ring_values,
                (ygrid, xgrid),
                method="linear",
                fill_value=np.nan,
            )

        outer_ring = coordinates[:, -1]
        outer_path = MatplotlibPath(
            np.column_stack((outer_ring[1], outer_ring[0]))
        )
        pixels = np.column_stack((xgrid.ravel(), ygrid.ravel()))
        inside_outer_ring = outer_path.contains_points(
            pixels,
            radius=1e-9,
        ).reshape(values.shape)
        axisymmetric[~inside_outer_ring] = np.nan

    domain = (
        profile["support"]
        & np.isfinite(values)
        & np.isfinite(axisymmetric)
    )
    axisymmetric[~domain] = np.nan
    residual[domain] = values[domain] - axisymmetric[domain]
    return axisymmetric, residual


def subtract_dominant_angular_mode(
    image,
    maximum_mode=6,
    include_axisymmetric=True,
    mask=None,
    center=None,
    n_azimuth=360,
    minimum_coverage=0.75,
    minimum_radius=2.0,
    ring_geometry=None,
):
    """Return the dominant fitted angular mode and its residual.

    Modes from ``m=0`` through ``maximum_mode`` are fitted simultaneously on
    every accepted ring. The selected mode maximizes its area-weighted power
    integrated over radius. Set ``include_axisymmetric=False`` to restrict
    selection to ``m>=1``. The reconstructed mode and residual are NaN outside
    the valid image support or the outermost accepted annulus.
    """

    if maximum_mode < 1:
        raise ValueError("maximum_mode must be positive")
    if n_azimuth < 2 * maximum_mode + 3:
        raise ValueError(
            "n_azimuth must exceed the number of fitted coefficients"
        )
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in the interval (0, 1]")

    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    finite = np.isfinite(values)
    if mask is None:
        support = finite
    else:
        support = np.asarray(mask, dtype=bool).copy()
        if support.shape != values.shape:
            raise ValueError("mask must have the same shape as image")
        support &= finite

    dominant = np.full(values.shape, np.nan, dtype=float)
    residual = np.full(values.shape, np.nan, dtype=float)
    if not np.any(support):
        return np.nan, dominant, residual

    radii, angles, coordinates = _polar_sampling_grid(
        values.shape,
        center=center,
        n_azimuth=n_azimuth,
        minimum_radius=minimum_radius,
        ring_geometry=ring_geometry,
    )
    if radii.size == 0:
        return np.nan, dominant, residual

    samples = map_coordinates(
        np.where(finite, values, 0.0),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    sampled_support = map_coordinates(
        support.astype(float),
        coordinates,
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5

    accepted_indices = []
    accepted_radii = []
    ring_coefficients = []
    mode_power = np.zeros(maximum_mode + 1, dtype=float)
    for index, (radius, ring, valid) in enumerate(
        zip(radii, samples, sampled_support)
    ):
        coverage = np.mean(valid)
        if coverage < minimum_coverage:
            continue
        ring = ring[valid]
        ring_angles = angles[valid]
        if ring.size < 2 * maximum_mode + 3:
            continue

        columns = [np.ones(ring.size)]
        for mode in range(1, maximum_mode + 1):
            columns.extend(
                (
                    np.cos(mode * ring_angles),
                    np.sin(mode * ring_angles),
                )
            )
        coefficients = np.linalg.lstsq(
            np.column_stack(columns),
            ring,
            rcond=None,
        )[0]
        ring_weight = float(radius * coverage)
        mode_power[0] += ring_weight * coefficients[0] ** 2
        for mode in range(1, maximum_mode + 1):
            cosine = coefficients[2 * mode - 1]
            sine = coefficients[2 * mode]
            mode_power[mode] += (
                0.5 * ring_weight * (cosine**2 + sine**2)
            )
        accepted_indices.append(index)
        accepted_radii.append(radius)
        ring_coefficients.append(coefficients)

    eligible_power = (
        mode_power if include_axisymmetric else mode_power[1:]
    )
    if not accepted_radii or not np.any(eligible_power > 0.0):
        return np.nan, dominant, residual

    peak_mode = int(np.argmax(eligible_power))
    if not include_axisymmetric:
        peak_mode += 1
    accepted_indices = np.asarray(accepted_indices, dtype=int)
    accepted_radii = np.asarray(accepted_radii, dtype=float)
    ring_coefficients = np.asarray(ring_coefficients, dtype=float)
    accepted_coordinates = coordinates[:, accepted_indices]

    ny, nx = values.shape
    ygrid, xgrid = np.indices(values.shape, dtype=float)
    if center is None:
        ycenter = 0.5 * (ny - 1)
        xcenter = 0.5 * (nx - 1)
    else:
        ycenter, xcenter = (float(value) for value in center)

    if ring_geometry is None:
        pixel_radius = np.hypot(xgrid - xcenter, ygrid - ycenter)
        if peak_mode == 0:
            dominant = np.interp(
                pixel_radius,
                accepted_radii,
                ring_coefficients[:, 0],
                left=ring_coefficients[0, 0],
                right=np.nan,
            )
        else:
            cosine = np.interp(
                pixel_radius,
                accepted_radii,
                ring_coefficients[:, 2 * peak_mode - 1],
                left=ring_coefficients[0, 2 * peak_mode - 1],
                right=np.nan,
            )
            sine = np.interp(
                pixel_radius,
                accepted_radii,
                ring_coefficients[:, 2 * peak_mode],
                left=ring_coefficients[0, 2 * peak_mode],
                right=np.nan,
            )
            pixel_angle = np.arctan2(ygrid - ycenter, xgrid - xcenter)
            dominant = (
                cosine * np.cos(peak_mode * pixel_angle)
                + sine * np.sin(peak_mode * pixel_angle)
            )
    else:
        if peak_mode == 0:
            ring_model = np.repeat(
                ring_coefficients[:, :1],
                angles.size,
                axis=1,
            )
        else:
            cosine = ring_coefficients[:, 2 * peak_mode - 1]
            sine = ring_coefficients[:, 2 * peak_mode]
            ring_model = (
                cosine[:, np.newaxis]
                * np.cos(peak_mode * angles[np.newaxis, :])
                + sine[:, np.newaxis]
                * np.sin(peak_mode * angles[np.newaxis, :])
            )
        points = np.column_stack(
            (
                accepted_coordinates[0].ravel(),
                accepted_coordinates[1].ravel(),
            )
        )
        if accepted_radii.size == 1:
            dominant.fill(float(np.mean(ring_model)))
        else:
            dominant = griddata(
                points,
                ring_model.ravel(),
                (ygrid, xgrid),
                method="linear",
                fill_value=np.nan,
            )

        outer_ring = accepted_coordinates[:, -1]
        outer_path = MatplotlibPath(
            np.column_stack((outer_ring[1], outer_ring[0]))
        )
        pixels = np.column_stack((xgrid.ravel(), ygrid.ravel()))
        inside_outer_ring = outer_path.contains_points(
            pixels,
            radius=1e-9,
        ).reshape(values.shape)
        dominant[~inside_outer_ring] = np.nan

    domain = support & np.isfinite(values) & np.isfinite(dominant)
    dominant[~domain] = np.nan
    residual[domain] = values[domain] - dominant[domain]
    return peak_mode, dominant, residual


def azimuthal_phase_coherence(
    image,
    mode=2,
    mask=None,
    center=None,
    n_azimuth=360,
    minimum_coverage=0.75,
    minimum_relative_power=0.01,
    minimum_radius=2.0,
    ring_geometry=None,
):
    """Measure how closely modal phases follow one logarithmic winding.

    The complex mode phase is fitted as a linear function of log radius.
    Coherence is the power-weighted circular concentration of the residual
    phases and ranges from zero to one. The fitted phase slope is also
    returned so fixed and winding patterns can be distinguished.
    """

    output = {
        f"eigenimage_m{mode}_phase_coherence": np.nan,
        f"eigenimage_m{mode}_phase_slope_logr": np.nan,
        f"eigenimage_m{mode}_phase_rings": 0,
    }
    if not 0.0 <= minimum_relative_power < 1.0:
        raise ValueError("minimum_relative_power must be in [0, 1)")
    profile = _azimuthal_mode_profile(
        image,
        mode=mode,
        mask=mask,
        center=center,
        n_azimuth=n_azimuth,
        minimum_coverage=minimum_coverage,
        minimum_radius=minimum_radius,
        ring_geometry=ring_geometry,
    )
    if profile is None:
        return output

    power = profile["explained_power"]
    maximum_power = np.max(power)
    useful = (
        np.isfinite(power)
        & (power > 0.0)
        & (power >= minimum_relative_power * maximum_power)
    )
    if np.count_nonzero(useful) < 3:
        return output

    radii = profile["radius"][useful]
    weights = power[useful]
    phases = np.unwrap(np.angle(profile["coefficient"][useful]))
    design = np.column_stack((np.ones(radii.size), np.log(radii)))
    weighted_design = design * np.sqrt(weights)[:, np.newaxis]
    weighted_phase = phases * np.sqrt(weights)
    intercept, slope = np.linalg.lstsq(
        weighted_design,
        weighted_phase,
        rcond=None,
    )[0]
    residual_phase = phases - (intercept + slope * np.log(radii))
    coherence = np.abs(
        np.sum(weights * np.exp(1j * residual_phase)) / np.sum(weights)
    )

    output[f"eigenimage_m{mode}_phase_coherence"] = float(coherence)
    output[f"eigenimage_m{mode}_phase_slope_logr"] = float(slope)
    output[f"eigenimage_m{mode}_phase_rings"] = int(radii.size)
    return output


def angular_mode_metrics(
    image,
    maximum_mode=6,
    mask=None,
    center=None,
    n_azimuth=360,
    minimum_coverage=0.75,
    minimum_radius=2.0,
    minimum_relative_power=0.01,
    ring_geometry=None,
):
    """Measure a simultaneous low-order angular-mode spectrum."""

    if maximum_mode < 1:
        raise ValueError("maximum_mode must be positive")
    if n_azimuth < 2 * maximum_mode + 3:
        raise ValueError(
            "n_azimuth must exceed the number of fitted coefficients"
        )
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in the interval (0, 1]")
    if not 0.0 <= minimum_relative_power < 1.0:
        raise ValueError("minimum_relative_power must be in [0, 1)")

    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    finite = np.isfinite(values)
    if mask is None:
        support = finite
    else:
        support = np.asarray(mask, dtype=bool).copy()
        if support.shape != values.shape:
            raise ValueError("mask must have the same shape as image")
        support &= finite

    output = {
        "eigenimage_m_peak": np.nan,
        "eigenimage_f_peak_total": np.nan,
        "eigenimage_f_peak_nonaxisymmetric": np.nan,
        "eigenimage_f_peak_fitted": np.nan,
        "eigenimage_m_dominant_all": np.nan,
        "eigenimage_f_dominant_all_total": np.nan,
        "eigenimage_mode_entropy": np.nan,
        "eigenimage_angular_model_fraction_total": np.nan,
        "eigenimage_angular_model_fraction_nonaxisymmetric": np.nan,
        "eigenimage_f_nonaxisymmetric": np.nan,
        "eigenimage_angular_unresolved_fraction_total": np.nan,
        "eigenimage_mpeak_phase_coherence": np.nan,
        "eigenimage_mpeak_phase_slope_logr": np.nan,
        "eigenimage_mpeak_orientation_slope_logr": np.nan,
        "eigenimage_mpeak_phase_rings": 0,
        "eigenimage_angular_mode_max": int(maximum_mode),
        "eigenimage_angular_min_radius_pix": float(minimum_radius),
    }
    if not np.any(support):
        return output

    radii, angles, coordinates = _polar_sampling_grid(
        values.shape,
        center=center,
        n_azimuth=n_azimuth,
        minimum_radius=minimum_radius,
        ring_geometry=ring_geometry,
    )
    if radii.size == 0:
        return output
    samples = map_coordinates(
        np.where(finite, values, 0.0),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    sampled_support = map_coordinates(
        support.astype(float),
        coordinates,
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5

    ring_radii = []
    ring_coefficients = []
    ring_mode_power = []
    ring_all_mode_power = []
    total_image_power = 0.0
    total_nonaxisymmetric_power = 0.0
    modeled_power = 0.0
    for radius, ring, valid in zip(radii, samples, sampled_support):
        coverage = np.mean(valid)
        if coverage < minimum_coverage:
            continue
        ring = ring[valid]
        ring_angles = angles[valid]
        if ring.size < 2 * maximum_mode + 3:
            continue

        columns = [np.ones(ring.size)]
        for mode in range(1, maximum_mode + 1):
            columns.extend(
                (
                    np.cos(mode * ring_angles),
                    np.sin(mode * ring_angles),
                )
            )
        design = np.column_stack(columns)
        coefficients = np.linalg.lstsq(design, ring, rcond=None)[0]
        residual_constant = ring - np.mean(ring)
        residual_model = ring - design @ coefficients
        constant_sse = float(np.sum(residual_constant**2))
        model_sse = float(np.sum(residual_model**2))
        ring_weight = float(radius * coverage / ring.size)
        total_image_power += ring_weight * float(np.sum(ring**2))
        total_nonaxisymmetric_power += ring_weight * constant_sse
        modeled_power += ring_weight * max(0.0, constant_sse - model_sse)

        complex_coefficients = (
            coefficients[1::2] - 1j * coefficients[2::2]
        )
        ring_radii.append(radius)
        ring_coefficients.append(complex_coefficients)
        ring_mode_power.append(
            radius * coverage * np.abs(complex_coefficients) ** 2
        )
        ring_all_mode_power.append(
            np.concatenate(
                (
                    [radius * coverage * coefficients[0] ** 2],
                    0.5
                    * radius
                    * coverage
                    * np.abs(complex_coefficients) ** 2,
                )
            )
        )

    if not ring_radii:
        return output

    ring_radii = np.asarray(ring_radii, dtype=float)
    ring_coefficients = np.asarray(ring_coefficients, dtype=complex)
    ring_mode_power = np.asarray(ring_mode_power, dtype=float)
    all_mode_power = np.sum(
        np.asarray(ring_all_mode_power, dtype=float),
        axis=0,
    )
    if np.any(all_mode_power > 0.0):
        dominant_all_mode = int(np.argmax(all_mode_power))
        output["eigenimage_m_dominant_all"] = dominant_all_mode
        if total_image_power > 0.0:
            output["eigenimage_f_dominant_all_total"] = float(
                np.clip(
                    all_mode_power[dominant_all_mode] / total_image_power,
                    0.0,
                    1.0,
                )
            )
    if total_image_power > 0.0:
        output.update(
            {
                "eigenimage_angular_model_fraction_total": float(
                    np.clip(modeled_power / total_image_power, 0.0, 1.0)
                ),
                "eigenimage_f_nonaxisymmetric": float(
                    np.clip(
                        total_nonaxisymmetric_power / total_image_power,
                        0.0,
                        1.0,
                    )
                ),
                "eigenimage_angular_unresolved_fraction_total": float(
                    np.clip(
                        (
                            total_nonaxisymmetric_power - modeled_power
                        ) / total_image_power,
                        0.0,
                        1.0,
                    )
                ),
            }
        )
    if total_nonaxisymmetric_power > 0.0:
        output[
            "eigenimage_angular_model_fraction_nonaxisymmetric"
        ] = float(
            np.clip(
                modeled_power / total_nonaxisymmetric_power,
                0.0,
                1.0,
            )
        )
    mode_power = np.sum(ring_mode_power, axis=0)
    total_mode_power = np.sum(mode_power)
    if not np.isfinite(total_mode_power) or total_mode_power <= 0.0:
        return output

    mode_fractions = mode_power / total_mode_power
    peak_index = int(np.argmax(mode_fractions))
    peak_mode = peak_index + 1
    positive = mode_fractions > 0.0
    entropy = -np.sum(
        mode_fractions[positive] * np.log(mode_fractions[positive])
    )
    if maximum_mode > 1:
        entropy /= np.log(maximum_mode)
    else:
        entropy = 0.0

    output.update(
        {
            "eigenimage_m_peak": peak_mode,
            "eigenimage_f_peak_fitted": float(
                mode_fractions[peak_index]
            ),
            "eigenimage_mode_entropy": float(entropy),
        }
    )
    peak_power = mode_fractions[peak_index] * modeled_power
    if total_image_power > 0.0:
        output["eigenimage_f_peak_total"] = float(
            np.clip(peak_power / total_image_power, 0.0, 1.0)
        )
    if total_nonaxisymmetric_power > 0.0:
        output["eigenimage_f_peak_nonaxisymmetric"] = float(
            np.clip(
                peak_power / total_nonaxisymmetric_power,
                0.0,
                1.0,
            )
        )

    phase_power = ring_mode_power[:, peak_index]
    maximum_phase_power = np.max(phase_power)
    useful = (
        np.isfinite(phase_power)
        & (phase_power > 0.0)
        & (phase_power >= minimum_relative_power * maximum_phase_power)
    )
    if np.count_nonzero(useful) < 3:
        return output

    phase_radii = ring_radii[useful]
    phase_weights = phase_power[useful]
    phases = np.unwrap(
        np.angle(ring_coefficients[useful, peak_index])
    )
    design = np.column_stack(
        (np.ones(phase_radii.size), np.log(phase_radii))
    )
    weighted_design = design * np.sqrt(phase_weights)[:, np.newaxis]
    weighted_phase = phases * np.sqrt(phase_weights)
    intercept, phase_slope = np.linalg.lstsq(
        weighted_design,
        weighted_phase,
        rcond=None,
    )[0]
    residual_phase = phases - (
        intercept + phase_slope * np.log(phase_radii)
    )
    coherence = np.abs(
        np.sum(phase_weights * np.exp(1j * residual_phase))
        / np.sum(phase_weights)
    )
    output.update(
        {
            "eigenimage_mpeak_phase_coherence": float(coherence),
            "eigenimage_mpeak_phase_slope_logr": float(phase_slope),
            "eigenimage_mpeak_orientation_slope_logr": float(
                -phase_slope / peak_mode
            ),
            "eigenimage_mpeak_phase_rings": int(phase_radii.size),
        }
    )
    return output


def excursion_set_metrics(image, mask=None, percentile=90.0):
    """Measure compactness and topology of a sign-invariant excursion set."""

    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between zero and 100")

    finite = np.isfinite(values)
    if mask is None:
        support = finite
    else:
        support = np.asarray(mask, dtype=bool).copy()
        if support.shape != values.shape:
            raise ValueError("mask must have the same shape as image")
        support &= finite

    output = {
        "eigenimage_excursion_threshold": np.nan,
        "eigenimage_excursion_area_pix": 0,
        "eigenimage_excursion_area_fraction": np.nan,
        "eigenimage_excursion_perimeter_pix": np.nan,
        "eigenimage_compactness": np.nan,
        "eigenimage_euler_characteristic": np.nan,
    }
    if not np.any(support):
        return output

    amplitude = np.abs(values)
    threshold = np.percentile(amplitude[support], percentile)
    excursion = support & (amplitude >= threshold)
    area = int(np.count_nonzero(excursion))
    perimeter = float(perimeter_crofton(excursion, directions=4))
    if area == 0 or perimeter <= 0.0:
        return output

    compactness = 4.0 * np.pi * area / perimeter**2
    output.update(
        {
            "eigenimage_excursion_threshold": float(threshold),
            "eigenimage_excursion_area_pix": area,
            "eigenimage_excursion_area_fraction": float(
                area / np.count_nonzero(support)
            ),
            "eigenimage_excursion_perimeter_pix": perimeter,
            "eigenimage_compactness": float(
                np.clip(compactness, 0.0, 1.0)
            ),
            "eigenimage_euler_characteristic": int(
                euler_number(excursion, connectivity=2)
            ),
        }
    )
    return output


def _eigenimage_center(result):
    ny, nx = result.eigenimages.shape[1:]

    # Match Cube.get_image_center() and Model._make_grid(): DiscMiner's sky
    # coordinates are based on the geometric image centre, while the fitted
    # xc and yc offsets are applied separately by the disc-to-sky projection.
    # CRPIX is only a WCS reference pixel and need not identify the source or
    # the centre of a cropped/downsampled image.
    return 0.5 * (ny - 1), 0.5 * (nx - 1)


def _beam_fwhm_pixels(result):
    header = result.source_header
    values = np.asarray(
        [
            header.get("BMAJ"),
            header.get("BMIN"),
            header.get("CDELT1"),
            header.get("CDELT2"),
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(values)) or np.any(values[:2] <= 0.0):
        return 1.0
    pixel_x, pixel_y = np.abs(values[2:])
    if pixel_x <= 0.0 or pixel_y <= 0.0:
        return 1.0
    return float(np.sqrt(values[0] * values[1] / (pixel_x * pixel_y)))


def _angular_minimum_radius(result, maximum_angular_mode):
    return max(
        2.0,
        maximum_angular_mode * _beam_fwhm_pixels(result) / np.pi,
    )


def _validated_components(result, components):
    if components is None:
        return np.arange(result.n_components, dtype=int)
    components = np.asarray(list(components), dtype=int)
    if components.ndim != 1:
        raise ValueError("components must be a one-dimensional sequence")
    if np.any(components < 0) or np.any(components >= result.n_components):
        raise IndexError(
            "components contains an index outside the available range "
            f"0 to {result.n_components - 1}"
        )
    return components


def characterize_result(
    result,
    label="data",
    artifact=None,
    components=None,
    acf_level=np.exp(-1),
    n_azimuth=360,
    minimum_azimuthal_coverage=0.75,
    phase_minimum_relative_power=0.01,
    excursion_percentile=90.0,
    maximum_angular_mode=6,
    ring_geometry=None,
):
    """Return intrinsic component and ACF-shape measurements as a table."""

    components = _validated_components(result, components)
    variance = np.asarray(result.variance_fraction, dtype=float)
    cumulative = np.asarray(result.cumulative_variance, dtype=float)
    cumulative_no_pc0 = np.full(result.n_components, np.nan, dtype=float)
    variance_no_pc0_renormalized = np.full(
        result.n_components,
        np.nan,
        dtype=float,
    )
    cumulative_no_pc0_renormalized = np.full(
        result.n_components,
        np.nan,
        dtype=float,
    )
    if result.n_components > 1:
        cumulative_no_pc0[1:] = np.cumsum(variance[1:])
        remaining_variance = np.sum(variance[1:])
        if remaining_variance > 0.0:
            variance_no_pc0_renormalized[1:] = (
                variance[1:] / remaining_variance
            )
            cumulative_no_pc0_renormalized[1:] = np.cumsum(
                variance_no_pc0_renormalized[1:]
            )
    eigenimage_center = _eigenimage_center(result)
    eigenimage_support = np.any(result.valid_mask, axis=0)
    angular_resolution_pix = _beam_fwhm_pixels(result)
    angular_minimum_radius = _angular_minimum_radius(
        result,
        maximum_angular_mode,
    )
    ring_radii, _, _ = _polar_sampling_grid(
        result.eigenimages.shape[1:],
        center=eigenimage_center,
        n_azimuth=n_azimuth,
        minimum_radius=angular_minimum_radius,
        ring_geometry=ring_geometry,
    )
    image_shape = result.eigenimages.shape[1:]
    large_enough_for_rings = min(image_shape) > 2.0 * angular_minimum_radius
    if ring_radii.size == 0 and large_enough_for_rings:
        warnings.warn(
            "No complete radial rings are available for PCA eigenimage "
            "characterization; ring-based angular diagnostics will be NaN. "
            f"Inferred center (y, x) is {eigenimage_center} for image shape "
            f"{image_shape}",
            RuntimeWarning,
            stacklevel=2,
        )

    rows = []
    for component in components:
        metrics = _empty_acf_ellipse_metrics()
        if component < result.spatial_autocorrelation.shape[0]:
            autocorrelation = result.spatial_autocorrelation[component]
            metrics = acf_ellipse_metrics(
                autocorrelation,
                level=acf_level,
            )
            metrics.update(
                acf_multipole_metrics(
                    autocorrelation,
                    level=acf_level,
                    n_azimuth=n_azimuth,
                )
            )
        else:
            metrics.update(acf_multipole_metrics(np.full((2, 2), np.nan)))

        metrics.update(
            azimuthal_axisymmetric_fraction(
                result.eigenimages[component],
                mask=eigenimage_support,
                center=eigenimage_center,
                n_azimuth=n_azimuth,
                minimum_coverage=minimum_azimuthal_coverage,
                minimum_radius=angular_minimum_radius,
                ring_geometry=ring_geometry,
            )
        )
        metrics.update(
            azimuthal_mode_fraction(
                result.eigenimages[component],
                mode=2,
                mask=eigenimage_support,
                center=eigenimage_center,
                n_azimuth=n_azimuth,
                minimum_coverage=minimum_azimuthal_coverage,
                minimum_radius=angular_minimum_radius,
                ring_geometry=ring_geometry,
            )
        )
        metrics.update(
            azimuthal_phase_coherence(
                result.eigenimages[component],
                mode=2,
                mask=eigenimage_support,
                center=eigenimage_center,
                n_azimuth=n_azimuth,
                minimum_coverage=minimum_azimuthal_coverage,
                minimum_relative_power=phase_minimum_relative_power,
                minimum_radius=angular_minimum_radius,
                ring_geometry=ring_geometry,
            )
        )
        metrics.update(
            excursion_set_metrics(
                result.eigenimages[component],
                mask=eigenimage_support,
                percentile=excursion_percentile,
            )
        )
        metrics.update(
            angular_mode_metrics(
                result.eigenimages[component],
                maximum_mode=maximum_angular_mode,
                mask=eigenimage_support,
                center=eigenimage_center,
                n_azimuth=n_azimuth,
                minimum_coverage=minimum_azimuthal_coverage,
                minimum_radius=angular_minimum_radius,
                minimum_relative_power=phase_minimum_relative_power,
                ring_geometry=ring_geometry,
            )
        )

        row = {
            "label": str(label),
            "artifact": "" if artifact is None else str(artifact),
            "component": int(component),
            "eigenvalue": float(result.eigenvalues[component]),
            "variance_fraction": float(variance[component]),
            "variance_percent": float(100.0 * variance[component]),
            "variance_fraction_no_pc0_renormalized": float(
                variance_no_pc0_renormalized[component]
            ),
            "variance_percent_no_pc0_renormalized": float(
                100.0 * variance_no_pc0_renormalized[component]
            ),
            "cumulative_variance": float(cumulative[component]),
            "cumulative_variance_no_pc0": float(
                cumulative_no_pc0[component]
            ),
            "cumulative_variance_no_pc0_renormalized": float(
                cumulative_no_pc0_renormalized[component]
            ),
            "cumulative_variance_percent_no_pc0_renormalized": float(
                100.0 * cumulative_no_pc0_renormalized[component]
            ),
            "acf_level": float(acf_level),
            "azimuth_samples": int(n_azimuth),
            "minimum_azimuthal_coverage": float(
                minimum_azimuthal_coverage
            ),
            "phase_minimum_relative_power": float(
                phase_minimum_relative_power
            ),
            "excursion_percentile": float(excursion_percentile),
            "eigenimage_angular_resolution_pix": float(
                angular_resolution_pix
            ),
            "eigenimage_ring_geometry": (
                "circular_sky"
                if ring_geometry is None
                else ring_geometry["mode"]
            ),
            "eigenimage_ring_parfile": (
                "" if ring_geometry is None else ring_geometry["parfile"]
            ),
            "eigenimage_ring_surface": (
                "" if ring_geometry is None else ring_geometry["surface"]
            ),
            "eigenimage_ring_surface_model": (
                ""
                if ring_geometry is None
                else ring_geometry["surface_model"]
            ),
            "eigenimage_ring_inclination_rad": (
                np.nan
                if ring_geometry is None
                else ring_geometry["inclination_rad"]
            ),
            "eigenimage_ring_position_angle_rad": (
                np.nan
                if ring_geometry is None
                else ring_geometry["position_angle_rad"]
            ),
            "eigenimage_ring_center_x_au": (
                np.nan
                if ring_geometry is None
                else ring_geometry["center_x_au"]
            ),
            "eigenimage_ring_center_y_au": (
                np.nan
                if ring_geometry is None
                else ring_geometry["center_y_au"]
            ),
            "eigenimage_ring_radial_step_au": (
                np.nan
                if ring_geometry is None
                else ring_geometry["radial_step_au"]
            ),
            "eigenimage_ring_radial_bins": int(ring_radii.size),
            "eigenimage_ring_min_radius_pix": (
                float(ring_radii[0]) if ring_radii.size else np.nan
            ),
            "eigenimage_ring_max_radius_pix": (
                float(ring_radii[-1]) if ring_radii.size else np.nan
            ),
            "eigenimage_center_x_pix": float(eigenimage_center[1]),
            "eigenimage_center_y_pix": float(eigenimage_center[0]),
        }
        row.update(metrics)
        rows.append(row)

    return Table(rows=rows)


def write_characterization_table(table, output, overwrite=True):
    """Write a PCA characterization table in Astropy ECSV format."""

    output = Path(output)
    table.write(output, format="ascii.ecsv", overwrite=overwrite)
    return output


def plot_characterization(
    table,
    output,
    include_pc0_cumulative=False,
    batch="first",
    dpi=200,
    show=False,
):
    """Plot one batch of PCA morphology measurements."""

    use_discminer_style()
    if batch == "first":
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
        (
            variance_axis,
            cumulative_axis,
            ratio_axis,
            residual_axis,
        ) = axes.ravel()
    elif batch == "second":
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), squeeze=False)
        q2_axis, q4_axis, fraction_axis = axes.ravel()
    elif batch == "third":
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), squeeze=False)
        phase_axis, compactness_axis, euler_axis = axes.ravel()
    else:
        raise ValueError("batch must be 'first', 'second', or 'third'")

    labels = list(dict.fromkeys(str(value) for value in table["label"]))
    for label in labels:
        subset = table[np.asarray(table["label"] == label)]
        order = np.argsort(np.asarray(subset["component"], dtype=int))
        subset = subset[order]
        components = np.asarray(subset["component"], dtype=int)

        if batch == "first":
            variance_axis.plot(
                components,
                np.asarray(subset["variance_percent"], dtype=float),
                marker="o",
                label=label,
            )

            cumulative_column = (
                "cumulative_variance"
                if include_pc0_cumulative
                else "cumulative_variance_no_pc0"
            )
            cumulative_values = 100.0 * np.asarray(
                subset[cumulative_column], dtype=float
            )
            cumulative_valid = np.isfinite(cumulative_values)
            cumulative_axis.plot(
                components[cumulative_valid],
                cumulative_values[cumulative_valid],
                marker="o",
                label=label,
            )
            ratio_axis.plot(
                components,
                np.asarray(subset["acf_axis_ratio"], dtype=float),
                marker="o",
                label=label,
            )
            residual_axis.plot(
                components,
                np.asarray(subset["acf_ellipse_residual"], dtype=float),
                marker="o",
                label=label,
            )
        elif batch == "second":
            q2_axis.plot(
                components,
                np.asarray(subset["acf_max_q2"], dtype=float),
                marker="o",
                label=label,
            )
            q4_axis.plot(
                components,
                np.asarray(subset["acf_max_q4"], dtype=float),
                marker="o",
                label=label,
            )
            fraction_axis.plot(
                components,
                np.asarray(
                    subset["eigenimage_f_m2_total"],
                    dtype=float,
                ),
                marker="o",
                label=label,
            )
        else:
            phase_axis.plot(
                components,
                np.asarray(
                    subset["eigenimage_m2_phase_coherence"],
                    dtype=float,
                ),
                marker="o",
                label=label,
            )
            compactness_axis.plot(
                components,
                np.asarray(subset["eigenimage_compactness"], dtype=float),
                marker="o",
                label=label,
            )
            euler_axis.plot(
                components,
                np.asarray(
                    subset["eigenimage_euler_characteristic"],
                    dtype=float,
                ),
                marker="o",
                label=label,
            )

    if batch == "first":
        variance_axis.set_yscale("log")
        variance_axis.set_ylabel("Variance [%]")
        variance_axis.set_title("PCA variance spectrum")
        cumulative_axis.set_ylabel("Cumulative variance [%]")
        cumulative_axis.set_title(
            "Cumulative variance"
            if include_pc0_cumulative
            else "Cumulative variance excluding PC 0 (not renormalized)"
        )
        ratio_axis.set_ylim(0.0, 1.05)
        ratio_axis.set_ylabel(r"ACF axis ratio $q=b/a$")
        ratio_axis.set_title(r"Central $1/e$ ACF elongation")
        residual_axis.set_ylabel(r"RMS ellipse residual / $a$")
        residual_axis.set_title("ACF ellipse-fit residual")
        variance_axis.legend(frameon=False)
    elif batch == "second":
        for axis, column in (
            (q2_axis, "acf_max_q2"),
            (q4_axis, "acf_max_q4"),
        ):
            values = np.asarray(table[column], dtype=float)
            finite_values = values[np.isfinite(values)]
            peak = np.max(finite_values) if finite_values.size else 0.0
            axis.set_ylim(0.0, min(1.05, max(0.05, 1.15 * peak)))
        fraction_axis.set_ylim(0.0, 1.05)
        q2_axis.set_ylabel(r"$\max Q_2(r)$")
        q2_axis.set_title("ACF quadrupole strength")
        q4_axis.set_ylabel(r"$\max Q_4(r)$")
        q4_axis.set_title("ACF fourth-order strength")
        fraction_axis.set_ylabel(r"$f_{m=2,\,\mathrm{total}}$")
        fraction_axis.set_title(r"Total eigenimage $m=2$ power")
        q2_axis.legend(frameon=False)
    else:
        for axis, column in (
            (phase_axis, "eigenimage_m2_phase_coherence"),
            (compactness_axis, "eigenimage_compactness"),
        ):
            values = np.asarray(table[column], dtype=float)
            finite_values = values[np.isfinite(values)]
            peak = np.max(finite_values) if finite_values.size else 0.0
            axis.set_ylim(0.0, min(1.05, max(0.05, 1.15 * peak)))

        euler_values = np.asarray(
            table["eigenimage_euler_characteristic"],
            dtype=float,
        )
        euler_values = euler_values[np.isfinite(euler_values)]
        if euler_values.size:
            lower = np.min(euler_values)
            upper = np.max(euler_values)
            padding = max(1.0, 0.1 * (upper - lower))
            euler_axis.set_ylim(
                min(0.0, lower - padding),
                max(0.0, upper + padding),
            )
        euler_axis.axhline(0.0, color="0.5", linewidth=1.0)
        phase_axis.set_ylabel(r"$m=2$ phase coherence")
        phase_axis.set_title("Ordered two-fold phase")
        compactness_axis.set_ylabel(r"Compactness $4\pi A/P^2$")
        compactness_axis.set_title("Eigenimage excursion compactness")
        euler_axis.set_ylabel(r"Euler characteristic $\chi$")
        euler_axis.set_title(r"Regions minus holes")
        phase_axis.legend(frameon=False)

    for axis in axes.ravel():
        axis.set_xlabel("PCA component")
        axis.grid(alpha=0.3)

    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output


def _plot_component_series(axis, table, column):
    labels = list(dict.fromkeys(str(value) for value in table["label"]))
    for label in labels:
        subset = table[np.asarray(table["label"] == label)]
        order = np.argsort(np.asarray(subset["component"], dtype=int))
        subset = subset[order]
        axis.plot(
            np.asarray(subset["component"], dtype=int),
            np.asarray(subset[column], dtype=float),
            marker="o",
            label=label,
        )


def _phase_diagnostic_support(table, require_slope_quality=False):
    """Return rows whose dominant mode supports phase interpretation."""

    peak_fraction = np.asarray(
        table["eigenimage_f_peak_total"],
        dtype=float,
    )
    entropy = np.asarray(table["eigenimage_mode_entropy"], dtype=float)
    supported = (
        np.isfinite(peak_fraction)
        & np.isfinite(entropy)
        & (peak_fraction >= _PHASE_MINIMUM_PEAK_TOTAL)
        & (entropy <= _PHASE_MAXIMUM_MODE_ENTROPY)
    )
    if require_slope_quality:
        coherence = np.asarray(
            table["eigenimage_mpeak_phase_coherence"],
            dtype=float,
        )
        rings = np.asarray(
            table["eigenimage_mpeak_phase_rings"],
            dtype=float,
        )
        supported &= (
            np.isfinite(coherence)
            & np.isfinite(rings)
            & (coherence >= _SLOPE_MINIMUM_PHASE_COHERENCE)
            & (rings >= _SLOPE_MINIMUM_PHASE_RINGS)
        )
    return supported


def _plot_conditional_phase_series(
    axis,
    table,
    column,
    require_slope_quality=False,
):
    """Plot phase values while fading rows with weak angular support."""

    labels = list(dict.fromkeys(str(value) for value in table["label"]))
    for label in labels:
        subset = table[np.asarray(table["label"] == label)]
        order = np.argsort(np.asarray(subset["component"], dtype=int))
        subset = subset[order]
        components = np.asarray(subset["component"], dtype=int)
        values = np.asarray(subset[column], dtype=float)
        line = axis.plot(components, values, label=label)[0]
        color = line.get_color()
        finite = np.isfinite(values)
        supported = _phase_diagnostic_support(
            subset,
            require_slope_quality=require_slope_quality,
        )
        axis.scatter(
            components[finite & supported],
            values[finite & supported],
            s=36,
            facecolors=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        axis.scatter(
            components[finite & ~supported],
            values[finite & ~supported],
            s=36,
            facecolors="0.75",
            edgecolors=color,
            alpha=_CONDITIONAL_MARKER_ALPHA,
            zorder=3,
        )


def _finite_column(table, column):
    values = np.asarray(table[column], dtype=float)
    return values[np.isfinite(values)]


def _set_data_ylim(axis, values, lower_bound=None, upper_bound=None):
    if values.size == 0:
        return
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = max(upper - lower, 0.05 * max(abs(lower), abs(upper), 1.0))
    padding = 0.1 * span
    lower -= padding
    upper += padding
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    if upper_bound is not None:
        upper = min(upper_bound, upper)
    axis.set_ylim(lower, upper)


def core_characterization_paths(prefix):
    prefix = Path(prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return {
        "component_importance": prefix.with_name(
            f"{prefix.name}_component_importance.png"
        ),
        "acf_morphology": prefix.with_name(
            f"{prefix.name}_acf_morphology.png"
        ),
        "eigenimage_structure": prefix.with_name(
            f"{prefix.name}_eigenimage_structure.png"
        ),
        "angularmode": prefix.with_name(
            f"{prefix.name}_angularmode.png"
        ),
    }


def excursion_characterization_path(prefix):
    """Return the optional eigenimage-excursion figure path."""

    prefix = Path(prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return prefix.with_name(f"{prefix.name}_eigenimage_excursions.png")


def m0_residual_characterization_path(prefix):
    """Return the optional axisymmetric-subtraction figure path."""

    prefix = Path(prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return prefix.with_name(f"{prefix.name}_eigenimage_m0_residuals.png")


def mpeak_residual_characterization_path(prefix):
    """Return the optional dominant-mode-subtraction figure path."""

    prefix = Path(prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return prefix.with_name(f"{prefix.name}_eigenimage_mpeak_residuals.png")


def _symmetric_color_limit(values, robust=True, percentile=99.5):
    if robust and not 0.0 < percentile <= 100.0:
        raise ValueError(
            "percentile must be greater than zero and at most 100"
        )
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    if robust:
        limit = float(np.percentile(finite, percentile))
    else:
        limit = float(np.max(finite))
    if not np.isfinite(limit) or limit <= 0.0:
        return 1.0
    return limit


def _short_plot_label(label):
    label = str(label)
    path = Path(label)
    if path.suffix and len(label) > 40:
        parent_parts = path.parent.parts[-2:]
        if parent_parts:
            return "/".join(parent_parts)
    return label


def _plot_angular_mode_residuals(
    results,
    labels,
    component_lists,
    ring_geometries,
    output,
    mode_selection,
    n_azimuth=360,
    minimum_azimuthal_coverage=0.75,
    maximum_angular_mode=6,
    cmap="RdBu_r",
    robust=True,
    percentile=99.5,
    dpi=200,
    show=False,
):
    """Plot eigenimages, one selected angular mode, and their residuals."""

    results = list(results)
    labels = [str(label) for label in labels]
    component_lists = [list(values) for values in component_lists]
    ring_geometries = list(ring_geometries)
    if mode_selection not in {"m0", "mpeak"}:
        raise ValueError("mode_selection must be m0 or mpeak")
    if not results:
        raise ValueError("results must contain at least one PCA result")
    if len(labels) != len(results):
        raise ValueError("labels must contain one value per PCA result")
    if len(component_lists) != len(results):
        raise ValueError(
            "component_lists must contain one sequence per PCA result"
        )
    if len(ring_geometries) != len(results):
        raise ValueError(
            "ring_geometries must contain one value per PCA result"
        )

    validated_components = [
        result._validate_component_indices(components, "components")
        for result, components in zip(results, component_lists)
    ]
    if any(len(components) == 0 for components in validated_components):
        raise ValueError(
            "each component list must select at least one component"
        )
    displayed_components = sorted(
        {
            int(component)
            for components in validated_components
            for component in components
        }
    )

    use_discminer_style()
    ncols = len(displayed_components)
    nrows = 3 * len(results)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.8 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    if mode_selection == "m0":
        row_kinds = ("Original", r"$m=0$", r"Residual ($m\geq1$)")
    else:
        row_kinds = (
            r"Non-axisymmetric ($m\geq1$)",
            r"Dominant $m_\mathrm{peak}\geq1$",
            "Residual",
        )
    for result_index, (
        result,
        label,
        components,
        ring_geometry,
    ) in enumerate(
        zip(results, labels, validated_components, ring_geometries)
    ):
        component_set = set(int(component) for component in components)
        first_component = next(
            component
            for component in displayed_components
            if component in component_set
        )
        support = np.any(result.valid_mask, axis=0)
        center = _eigenimage_center(result)
        minimum_radius = _angular_minimum_radius(
            result,
            maximum_angular_mode,
        )
        row_offset = 3 * result_index
        display_label = _short_plot_label(label)

        for column, component in enumerate(displayed_components):
            component_axes = axes[row_offset:row_offset + 3, column]
            if component not in component_set:
                for axis in component_axes:
                    axis.axis("off")
                continue

            original = np.asarray(
                result.eigenimages[component],
                dtype=float,
            )
            original_masked = np.where(support, original, np.nan)
            if mode_selection == "m0":
                selected_mode = 0
                plot_input = original_masked
                removed_mode, residual = subtract_axisymmetric_mode(
                    original,
                    mask=support,
                    center=center,
                    n_azimuth=n_azimuth,
                    minimum_coverage=minimum_azimuthal_coverage,
                    minimum_radius=minimum_radius,
                    ring_geometry=ring_geometry,
                )
            else:
                _, nonaxisymmetric = subtract_axisymmetric_mode(
                    original,
                    mask=support,
                    center=center,
                    n_azimuth=n_azimuth,
                    minimum_coverage=minimum_azimuthal_coverage,
                    minimum_radius=minimum_radius,
                    ring_geometry=ring_geometry,
                )
                (
                    selected_mode,
                    removed_mode,
                    _,
                ) = subtract_dominant_angular_mode(
                    original,
                    maximum_mode=maximum_angular_mode,
                    include_axisymmetric=False,
                    mask=support,
                    center=center,
                    n_azimuth=n_azimuth,
                    minimum_coverage=minimum_azimuthal_coverage,
                    minimum_radius=minimum_radius,
                    ring_geometry=ring_geometry,
                )
                plot_input = nonaxisymmetric
                residual = np.full(original.shape, np.nan, dtype=float)
                residual_domain = (
                    np.isfinite(nonaxisymmetric)
                    & np.isfinite(removed_mode)
                )
                residual[residual_domain] = (
                    nonaxisymmetric[residual_domain]
                    - removed_mode[residual_domain]
                )
            shared_values = np.concatenate(
                (plot_input.ravel(), removed_mode.ravel())
            )
            shared_limit = _symmetric_color_limit(
                shared_values,
                robust=robust,
                percentile=percentile,
            )
            residual_limit = _symmetric_color_limit(
                residual,
                robust=robust,
                percentile=percentile,
            )

            for local_row, (axis, image) in enumerate(
                zip(
                    component_axes,
                    (plot_input, removed_mode, residual),
                )
            ):
                color_limit = (
                    shared_limit if local_row < 2 else residual_limit
                )
                plotted = axis.imshow(
                    np.ma.masked_invalid(image),
                    origin="lower",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=-color_limit,
                    vmax=color_limit,
                )
                fig.colorbar(
                    plotted,
                    ax=axis,
                    fraction=0.046,
                    pad=0.04,
                )
                axis.set_xticks([])
                axis.set_yticks([])
                if not np.any(np.isfinite(image)):
                    axis.text(
                        0.5,
                        0.5,
                        "no accepted rings",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                if mode_selection == "mpeak" and local_row == 1:
                    mode_text = (
                        f"m={int(selected_mode)}"
                        if np.isfinite(selected_mode)
                        else "m undefined"
                    )
                    axis.text(
                        0.04,
                        0.05,
                        mode_text,
                        transform=axis.transAxes,
                        fontsize=9,
                        color="black",
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "white",
                            "edgecolor": "0.5",
                            "alpha": 0.85,
                        },
                    )
                if component == first_component:
                    axis.set_ylabel(row_kinds[local_row])
            title = f"PC {component}"
            if component == first_component:
                title = f"{display_label}\n{title}"
            component_axes[0].set_title(title)

    scaling = (
        f"robust P{percentile:g}"
        if robust
        else "full-range"
    )
    decomposition = (
        "axisymmetric"
        if mode_selection == "m0"
        else "non-axisymmetric dominant-mode"
    )
    fig.suptitle(
        f"PCA eigenimage {decomposition} decomposition "
        f"({scaling} symmetric scaling)"
    )
    output = Path(output)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output


def plot_m0_residuals(
    results,
    labels,
    component_lists,
    ring_geometries,
    output,
    n_azimuth=360,
    minimum_azimuthal_coverage=0.75,
    maximum_angular_mode=6,
    cmap="RdBu_r",
    robust=True,
    percentile=99.5,
    dpi=200,
    show=False,
):
    """Plot each eigenimage, its projected ``m=0`` field, and residual."""

    return _plot_angular_mode_residuals(
        results,
        labels,
        component_lists,
        ring_geometries,
        output,
        mode_selection="m0",
        n_azimuth=n_azimuth,
        minimum_azimuthal_coverage=minimum_azimuthal_coverage,
        maximum_angular_mode=maximum_angular_mode,
        cmap=cmap,
        robust=robust,
        percentile=percentile,
        dpi=dpi,
        show=show,
    )


def plot_mpeak_residuals(
    results,
    labels,
    component_lists,
    ring_geometries,
    output,
    n_azimuth=360,
    minimum_azimuthal_coverage=0.75,
    maximum_angular_mode=6,
    cmap="RdBu_r",
    robust=True,
    percentile=99.5,
    dpi=200,
    show=False,
):
    """Plot each eigenimage, its dominant angular mode, and residual."""

    return _plot_angular_mode_residuals(
        results,
        labels,
        component_lists,
        ring_geometries,
        output,
        mode_selection="mpeak",
        n_azimuth=n_azimuth,
        minimum_azimuthal_coverage=minimum_azimuthal_coverage,
        maximum_angular_mode=maximum_angular_mode,
        cmap=cmap,
        robust=robust,
        percentile=percentile,
        dpi=dpi,
        show=show,
    )


def plot_excursion_sets(
    results,
    labels,
    component_lists,
    output,
    percentile=90.0,
    dpi=200,
    show=False,
):
    """Plot eigenimages with the excursion sets used for compactness."""

    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be between zero and 100")
    results = list(results)
    labels = [str(label) for label in labels]
    component_lists = [list(values) for values in component_lists]
    if not results:
        raise ValueError("results must contain at least one PCA result")
    if len(labels) != len(results):
        raise ValueError("labels must contain one value per PCA result")
    if len(component_lists) != len(results):
        raise ValueError(
            "component_lists must contain one sequence per PCA result"
        )

    validated_components = [
        result._validate_component_indices(components, "components")
        for result, components in zip(results, component_lists)
    ]
    if any(len(components) == 0 for components in validated_components):
        raise ValueError(
            "each component list must select at least one component"
        )
    displayed_components = sorted(
        {
            int(component)
            for components in validated_components
            for component in components
        }
    )
    if not displayed_components:
        raise ValueError("component_lists must select at least one component")

    use_discminer_style()
    nrows = len(results)
    ncols = len(displayed_components)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols, 3.0 * nrows),
        squeeze=False,
    )

    for row, (result, label, components) in enumerate(
        zip(results, labels, validated_components)
    ):
        component_set = set(int(component) for component in components)
        first_component = next(
            component
            for component in displayed_components
            if component in component_set
        )
        support = np.any(result.valid_mask, axis=0)
        for column, component in enumerate(displayed_components):
            axis = axes[row, column]
            if component not in component_set:
                axis.axis("off")
                continue

            values = np.asarray(result.eigenimages[component], dtype=float)
            metrics = excursion_set_metrics(
                values,
                mask=support,
                percentile=percentile,
            )
            threshold = metrics["eigenimage_excursion_threshold"]
            finite_values = np.abs(values[support & np.isfinite(values)])
            color_limit = (
                float(np.percentile(finite_values, 99.5))
                if finite_values.size
                else 1.0
            )
            if not np.isfinite(color_limit) or color_limit <= 0.0:
                color_limit = 1.0
            if np.isfinite(threshold):
                color_limit = max(color_limit, float(threshold))

            axis.imshow(
                np.ma.masked_where(~support, values),
                origin="lower",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-color_limit,
                vmax=color_limit,
            )
            if np.isfinite(threshold):
                excursion = support & (np.abs(values) >= threshold)
                axis.contour(
                    excursion.astype(float),
                    levels=[0.5],
                    colors=["black"],
                    linewidths=0.8,
                    origin="lower",
                )
                if np.any(support & (values >= threshold)):
                    axis.contour(
                        np.where(support, values, np.nan),
                        levels=[threshold],
                        colors=["#D62728"],
                        linewidths=1.5,
                        origin="lower",
                    )
                if np.any(support & (values <= -threshold)):
                    axis.contour(
                        np.where(support, values, np.nan),
                        levels=[-threshold],
                        colors=["#1F77B4"],
                        linewidths=1.5,
                        linestyles="dashed",
                        origin="lower",
                    )

            compactness = metrics["eigenimage_compactness"]
            compactness_text = (
                f"{compactness:.3f}"
                if np.isfinite(compactness)
                else "undefined"
            )
            axis.text(
                0.03,
                0.04,
                f"P{percentile:g}  |  C={compactness_text}",
                transform=axis.transAxes,
                fontsize=9,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "0.5",
                    "alpha": 0.85,
                },
            )
            if row == 0:
                axis.set_title(f"PC {component}")
            if component == first_component:
                axis.set_ylabel(label)
            axis.set_xticks([])
            axis.set_yticks([])

    legend = [
        Line2D(
            [0],
            [0],
            color="#D62728",
            linewidth=1.5,
            label="positive excursion",
        ),
        Line2D(
            [0],
            [0],
            color="#1F77B4",
            linewidth=1.5,
            linestyle="--",
            label="negative excursion",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=0.8,
            label="combined excursion boundary",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        rf"Eigenimage excursion sets: $|\mathrm{{PC}}| \geq "
        rf"P_{{{percentile:g}}}(|\mathrm{{PC}}|)$"
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94), pad=0.6)
    output = Path(output)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output


def _plot_core_variance(
    table,
    output,
    include_pc0_cumulative=False,
    dpi=200,
    show=False,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    (
        variance_axis,
        normalized_variance_axis,
        cumulative_axis,
        normalized_cumulative_axis,
    ) = axes.ravel()
    _plot_component_series(variance_axis, table, "variance_percent")
    _plot_component_series(
        normalized_variance_axis,
        table,
        "variance_percent_no_pc0_renormalized",
    )
    _plot_component_series(
        normalized_cumulative_axis,
        table,
        "cumulative_variance_percent_no_pc0_renormalized",
    )
    cumulative_column = (
        "cumulative_variance"
        if include_pc0_cumulative
        else "cumulative_variance_no_pc0"
    )
    labels = list(dict.fromkeys(str(value) for value in table["label"]))
    for label in labels:
        subset = table[np.asarray(table["label"] == label)]
        order = np.argsort(np.asarray(subset["component"], dtype=int))
        subset = subset[order]
        components = np.asarray(subset["component"], dtype=int)
        cumulative = 100.0 * np.asarray(subset[cumulative_column], dtype=float)
        valid = np.isfinite(cumulative)
        cumulative_axis.plot(
            components[valid],
            cumulative[valid],
            marker="o",
            label=label,
        )
    variance_axis.set_yscale("log")
    normalized_variance_axis.set_yscale("log")
    variance_axis.set_ylabel("Variance [% of total]")
    variance_axis.set_title("Absolute variance spectrum")
    normalized_variance_axis.set_ylabel("Variance [% beyond PC 0]")
    normalized_variance_axis.set_title(
        "Renormalized variance spectrum beyond PC 0"
    )
    cumulative_axis.set_ylabel("Cumulative variance [% of total]")
    cumulative_axis.set_title(
        "Cumulative variance"
        if include_pc0_cumulative
        else "Absolute cumulative variance beyond PC 0"
    )
    normalized_cumulative_axis.set_ylabel(
        "Cumulative variance [% beyond PC 0]"
    )
    normalized_cumulative_axis.set_title(
        "Renormalized cumulative variance beyond PC 0"
    )
    normalized_values = _finite_column(
        table,
        "cumulative_variance_percent_no_pc0_renormalized",
    )
    if normalized_values.size:
        normalized_minimum = float(np.min(normalized_values))
        normalized_maximum = float(np.max(normalized_values))
        normalized_span = normalized_maximum - normalized_minimum
        if normalized_span <= 0.0:
            normalized_span = max(abs(normalized_maximum), 1.0)
        normalized_padding = 0.15 * normalized_span
        normalized_cumulative_axis.set_ylim(
            normalized_minimum - normalized_padding,
            normalized_maximum + normalized_padding,
        )
    label_fontsize = 14
    title_fontsize = 16
    tick_fontsize = 12
    variance_axis.legend(frameon=False, fontsize=tick_fontsize)
    for axis in axes.ravel():
        axis.set_xlabel("PCA component", fontsize=label_fontsize)
        axis.xaxis.label.set_size(label_fontsize)
        axis.yaxis.label.set_size(label_fontsize)
        axis.title.set_size(title_fontsize)
        axis.tick_params(axis="both", labelsize=tick_fontsize)
        axis.grid(alpha=0.3)
    fig.tight_layout(pad=1.2)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_core_acf(table, output, dpi=200, show=False):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    ratio_axis, residual_axis, q2_axis, q2_radius_axis = axes.ravel()
    for axis, column in (
        (ratio_axis, "acf_axis_ratio"),
        (residual_axis, "acf_ellipse_residual"),
        (q2_axis, "acf_max_q2"),
        (q2_radius_axis, "acf_q2_radius_pix"),
    ):
        _plot_component_series(axis, table, column)
    _set_data_ylim(
        ratio_axis,
        _finite_column(table, "acf_axis_ratio"),
        lower_bound=0.0,
        upper_bound=1.05,
    )
    for axis, column in (
        (residual_axis, "acf_ellipse_residual"),
        (q2_axis, "acf_max_q2"),
    ):
        _set_data_ylim(
            axis,
            _finite_column(table, column),
            lower_bound=0.0,
        )
    q2_radius_values = _finite_column(table, "acf_q2_radius_pix")
    if q2_radius_values.size and np.all(q2_radius_values > 0.0):
        q2_radius_axis.set_yscale("log")
        q2_radius_axis.set_ylim(
            0.8 * np.min(q2_radius_values),
            1.2 * np.max(q2_radius_values),
        )
    ratio_axis.set_ylabel(r"ACF axis ratio $q=b/a$")
    ratio_axis.set_title(r"Central $1/e$ ACF elongation")
    residual_axis.set_ylabel(r"RMS ellipse residual / $a$")
    residual_axis.set_title("ACF ellipse-fit residual")
    q2_axis.set_ylabel(r"$\max Q_2(r)$")
    q2_axis.set_title("ACF quadrupole strength")
    q2_radius_axis.set_ylabel(r"Radius of $\max Q_2$ [pixel]")
    q2_radius_axis.set_title("Scale of strongest ACF anisotropy")
    ratio_axis.legend(frameon=False)
    for axis in axes.ravel():
        axis.set_xlabel("PCA component")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_core_eigenimage(
    table,
    output,
    angular_normalization="total",
    dpi=200,
    show=False,
):
    mode_fraction_column = (
        "eigenimage_f_m2_total"
        if angular_normalization == "total"
        else "eigenimage_f_m2_nonaxisymmetric"
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    (
        axisymmetric_axis,
        fraction_axis,
        phase_axis,
        slope_axis,
        compactness_axis,
        unused_axis,
    ) = axes.ravel()
    for axis, column in (
        (axisymmetric_axis, "eigenimage_f_m0"),
        (fraction_axis, mode_fraction_column),
        (phase_axis, "eigenimage_m2_phase_coherence"),
        (slope_axis, "eigenimage_m2_phase_slope_logr"),
        (compactness_axis, "eigenimage_compactness"),
    ):
        _plot_component_series(axis, table, column)
    for axis, column in (
        (axisymmetric_axis, "eigenimage_f_m0"),
        (fraction_axis, mode_fraction_column),
        (phase_axis, "eigenimage_m2_phase_coherence"),
    ):
        _set_data_ylim(
            axis,
            _finite_column(table, column),
            lower_bound=0.0,
            upper_bound=1.05,
        )
    slope_values = _finite_column(table, "eigenimage_m2_phase_slope_logr")
    if slope_values.size:
        maximum = max(np.max(np.abs(slope_values)), 0.1)
        slope_axis.set_ylim(-1.1 * maximum, 1.1 * maximum)
    compactness_values = _finite_column(table, "eigenimage_compactness")
    if compactness_values.size and np.all(compactness_values > 0.0):
        compactness_axis.set_yscale("log")
        compactness_axis.set_ylim(
            0.8 * np.min(compactness_values),
            min(1.0, 1.2 * np.max(compactness_values)),
        )
    axisymmetric_axis.set_ylabel(r"$f_{m=0}$")
    axisymmetric_axis.set_title("Axisymmetric eigenimage power")
    if angular_normalization == "total":
        fraction_axis.set_ylabel(r"$f_{m=2,\,\mathrm{total}}$")
        fraction_axis.set_title(r"Total eigenimage $m=2$ power")
    else:
        fraction_axis.set_ylabel(r"$f_{m=2,\,\mathrm{nonaxi}}$")
        fraction_axis.set_title(r"$m=2$ share of non-axisymmetric power")
    phase_axis.set_ylabel(r"$m=2$ phase coherence")
    phase_axis.set_title("Ordered two-fold phase")
    slope_axis.axhline(0.0, color="0.5", linewidth=1.0)
    slope_axis.set_ylabel(r"Phase slope [rad per $\ln r$]")
    slope_axis.set_title(r"$m=2$ winding slope")
    compactness_axis.set_ylabel(r"Compactness $4\pi A/P^2$")
    compactness_axis.set_title("Eigenimage excursion compactness")
    axisymmetric_axis.legend(frameon=False)
    unused_axis.axis("off")
    for axis in axes.ravel()[:-1]:
        axis.set_xlabel("PCA component")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_core_angularmode(
    table,
    output,
    angular_normalization="total",
    dpi=200,
    show=False,
):
    peak_fraction_column = (
        "eigenimage_f_peak_total"
        if angular_normalization == "total"
        else "eigenimage_f_peak_nonaxisymmetric"
    )
    model_fraction_column = (
        "eigenimage_angular_model_fraction_total"
        if angular_normalization == "total"
        else "eigenimage_angular_model_fraction_nonaxisymmetric"
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    (
        peak_axis,
        peak_fraction_axis,
        entropy_axis,
        model_fraction_axis,
        coherence_axis,
        slope_axis,
    ) = axes.ravel()
    for axis, column in (
        (peak_axis, "eigenimage_m_peak"),
        (peak_fraction_axis, peak_fraction_column),
        (entropy_axis, "eigenimage_mode_entropy"),
        (model_fraction_axis, model_fraction_column),
    ):
        _plot_component_series(axis, table, column)
    _plot_conditional_phase_series(
        coherence_axis,
        table,
        "eigenimage_mpeak_phase_coherence",
    )
    _plot_conditional_phase_series(
        slope_axis,
        table,
        "eigenimage_mpeak_orientation_slope_logr",
        require_slope_quality=True,
    )

    maximum_mode_values = _finite_column(
        table,
        "eigenimage_angular_mode_max",
    )
    maximum_mode = (
        int(np.max(maximum_mode_values))
        if maximum_mode_values.size
        else 1
    )
    peak_axis.set_ylim(0.5, maximum_mode + 0.5)
    peak_axis.set_yticks(np.arange(1, maximum_mode + 1))
    for axis, column in (
        (peak_fraction_axis, peak_fraction_column),
        (entropy_axis, "eigenimage_mode_entropy"),
        (model_fraction_axis, model_fraction_column),
        (coherence_axis, "eigenimage_mpeak_phase_coherence"),
    ):
        _set_data_ylim(
            axis,
            _finite_column(table, column),
            lower_bound=0.0,
            upper_bound=1.05,
        )
    slope_values = _finite_column(
        table,
        "eigenimage_mpeak_orientation_slope_logr",
    )
    if slope_values.size:
        maximum = max(np.max(np.abs(slope_values)), 0.1)
        slope_axis.set_ylim(-1.1 * maximum, 1.1 * maximum)
    slope_axis.axhline(0.0, color="0.5", linewidth=1.0)

    peak_axis.set_ylabel(r"$m_{\rm peak}\;(m\geq1)$")
    peak_axis.set_title("Characteristic non-axisymmetric mode")
    if angular_normalization == "total":
        peak_fraction_axis.set_ylabel(r"$f_{\rm peak,total}$")
        peak_fraction_axis.set_title(
            "Dominant non-axisymmetric mode / total power"
        )
        model_fraction_axis.set_ylabel(r"$f_{1:m_{\max},\,\rm total}$")
        model_fraction_axis.set_title("Low-order total power")
    else:
        peak_fraction_axis.set_ylabel(r"$f_{\rm peak,nonaxi}$")
        peak_fraction_axis.set_title("Dominant non-axisymmetric power")
        model_fraction_axis.set_ylabel(r"$f_{1:m_{\max},\,\rm nonaxi}$")
        model_fraction_axis.set_title("Modeled non-axisymmetric power")
    entropy_axis.set_ylabel("Mode entropy")
    entropy_axis.set_title("Non-axisymmetric mode complexity")
    coherence_axis.set_ylabel("Phase coherence")
    coherence_axis.set_title("Dominant-mode phase order")
    slope_axis.set_ylabel(r"$d\phi_{\rm peak}/d\ln r$ [rad]")
    slope_axis.set_title("Physical orientation slope")
    peak_axis.legend(frameon=False)
    slope_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markerfacecolor="0.75",
                markeredgecolor="0.35",
                alpha=_CONDITIONAL_MARKER_ALPHA,
                label="conditional: weak or complex mode",
            )
        ],
        frameon=False,
        fontsize="small",
    )
    for axis in axes.ravel():
        axis.set_xlabel("PCA component")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_core_characterization(
    table,
    prefix,
    groups=("variance", "acf", "eigenimage", "angularmode"),
    include_pc0_cumulative=False,
    angular_normalization="total",
    dpi=200,
    show=False,
):
    """Write selected retained groups of core PCA diagnostics."""

    use_discminer_style()
    if isinstance(groups, str):
        groups = (groups,)
    groups = tuple(groups)
    valid_groups = {"variance", "acf", "eigenimage", "angularmode"}
    invalid = set(groups) - valid_groups
    if invalid:
        raise ValueError(
            "groups contains unknown values: " + ", ".join(sorted(invalid))
        )
    if angular_normalization not in {"total", "nonaxisymmetric"}:
        raise ValueError(
            "angular_normalization must be total or nonaxisymmetric"
        )

    all_outputs = core_characterization_paths(prefix)
    output_keys = {
        "variance": "component_importance",
        "acf": "acf_morphology",
        "eigenimage": "eigenimage_structure",
        "angularmode": "angularmode",
    }
    outputs = {group: all_outputs[output_keys[group]] for group in groups}

    for group, output in outputs.items():
        if group == "variance":
            _plot_core_variance(
                table,
                output,
                include_pc0_cumulative=include_pc0_cumulative,
                dpi=dpi,
                show=show,
            )
        elif group == "acf":
            _plot_core_acf(table, output, dpi=dpi, show=show)
        elif group == "eigenimage":
            _plot_core_eigenimage(
                table,
                output,
                angular_normalization=angular_normalization,
                dpi=dpi,
                show=show,
            )
        else:
            _plot_core_angularmode(
                table,
                output,
                angular_normalization=angular_normalization,
                dpi=dpi,
                show=show,
            )
    return outputs
