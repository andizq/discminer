"""PCA decomposition of discminer data cubes."""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits

from .artifact import PCAResult


def _import_turbustat_pca():
    try:
        from turbustat.statistics import PCA
    except ImportError as exc:
        if getattr(exc, "name", None) == "pkg_resources":
            raise ImportError(
                "TurbuStat 1.3 requires the legacy pkg_resources module. "
                "Install the compatible PCA dependencies with: "
                "pip install 'discminer[pca]'"
            ) from exc
        raise ImportError(
            "The PCA workflow requires TurbuStat. Install the optional "
            "dependencies with: pip install 'discminer[pca]'"
        ) from exc
    return PCA


def _padded_measurements(values, n_components):
    output = np.full(n_components, np.nan, dtype=float)
    values = np.asarray(values, dtype=float)
    output[: values.size] = values
    return output


def run_pca(
    input_cube,
    distance,
    n_components=-1,
    mean_sub=False,
    spatial_method="contour",
    spectral_method="walk-down",
    beam_correct=True,
    show_progress=True,
    outer_radius_au=None,
):
    """Run PCA and return a self-contained :class:`PCAResult`.

    Parameters
    ----------
    input_cube : path-like
        FITS cube to decompose.
    distance : astropy.units.Quantity
        Distance to the source.
    n_components : int, optional
        Number of components used for width measurements. ``-1`` uses every
        spectral component. All components are always stored so that the
        original cube can be reconstructed.
    mean_sub : bool, optional
        Subtract each channel's spatial mean during decomposition.
    spatial_method, spectral_method : str, optional
        Width-estimation methods passed to TurbuStat.
    beam_correct : bool, optional
        Apply TurbuStat's spatial beam correction.
    show_progress : bool, optional
        Show TurbuStat's covariance progress indicator.
    outer_radius_au : float, optional
        Disc outer radius saved for normalized width plots.
    """

    from discminer.core import Data

    PCA = _import_turbustat_pca()

    input_cube = Path(input_cube)
    if not isinstance(distance, u.Quantity):
        raise TypeError("distance must be an astropy Quantity")
    distance = distance.to(u.pc)

    datacube = Data(str(input_cube), distance)
    data = np.asarray(datacube.data, dtype=float)
    if data.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional cube, got shape {data.shape}"
        )
    valid_mask = np.isfinite(data)
    pca_data = np.nan_to_num(data, nan=0.0)

    n_channels = data.shape[0]
    if n_components == -1:
        width_components = n_channels
    elif not 1 <= n_components <= n_channels:
        raise ValueError(
            "n_components must be -1 or between 1 and the number of channels"
        )
    else:
        width_components = int(n_components)

    hdu = fits.PrimaryHDU(data=pca_data, header=datacube.header)
    pca = PCA(hdu, distance=distance)
    pca.compute_pca(
        mean_sub=mean_sub,
        n_eigs=width_components,
        show_progress=show_progress,
    )
    try:
        pca.find_spatial_widths(
            method=spatial_method,
            beam_fwhm=None,
            brunt_beamcorrect=beam_correct,
            diagnosticplots=False,
        )
    except TypeError as exc:
        if "0-dimensional arrays" not in str(exc):
            raise
        raise RuntimeError(
            "TurbuStat 1.3's contour spatial-width fit is incompatible with "
            f"NumPy {np.__version__}. Install the compatible PCA dependencies "
            "with: python -m pip install 'numpy<2' 'astropy<8'"
        ) from exc
    pca.find_spectral_widths(method=spectral_method)

    spatial_autocorrelation = np.asarray(
        pca.autocorr_images(n_eigs=width_components),
        dtype=float,
    )
    if spatial_autocorrelation.ndim == 2:
        spatial_autocorrelation = spatial_autocorrelation[np.newaxis, ...]
    spatial_autocorrelation -= np.asarray(pca.noise_ACF(), dtype=float)

    spectral_autocorrelation = np.asarray(
        pca.autocorr_spec(n_eigs=width_components),
        dtype=float,
    )
    if spectral_autocorrelation.ndim == 1:
        spectral_autocorrelation = spectral_autocorrelation[:, np.newaxis]

    eigenimages = np.asarray(pca.eigimages(n_channels), dtype=float)
    if eigenimages.ndim == 2:
        eigenimages = eigenimages[np.newaxis, ...]
    # TurbuStat returns each eigenimage with its two spatial axes transposed
    # relative to the input cube. Store standard FITS/NumPy (component, y, x)
    # ordering so artifact reconstruction does not need hidden transposes.
    eigenimages = eigenimages.transpose(0, 2, 1)

    if mean_sub:
        channel_mean = np.mean(pca_data, axis=(1, 2))
    else:
        channel_mean = np.zeros(n_channels, dtype=float)

    spatial_width = _padded_measurements(
        pca.spatial_width(unit=u.au).value, n_channels
    )
    spatial_width_error = _padded_measurements(
        pca.spatial_width_error(unit=u.au).value, n_channels
    )
    spectral_width = _padded_measurements(
        pca.spectral_width(unit=u.km / u.s).value, n_channels
    )
    spectral_width_error = _padded_measurements(
        pca.spectral_width_error(unit=u.km / u.s).value, n_channels
    )

    return PCAResult(
        source_header=datacube.header,
        source_path=str(input_cube),
        distance_pc=distance.value,
        mean_sub=mean_sub,
        selected_components=width_components,
        eigenimages=eigenimages,
        eigenvectors=np.asarray(pca.eigvecs[:, :n_channels], dtype=float),
        eigenvalues=np.asarray(pca.eigvals[:n_channels], dtype=float),
        covariance=np.asarray(pca.cov_matrix, dtype=float),
        velocity=np.asarray(datacube.vchannels, dtype=float),
        channel_mean=channel_mean,
        spatial_width=spatial_width,
        spatial_width_error=spatial_width_error,
        spectral_width=spectral_width,
        spectral_width_error=spectral_width_error,
        valid_mask=valid_mask,
        spatial_autocorrelation=spatial_autocorrelation,
        spectral_autocorrelation=spectral_autocorrelation,
        outer_radius_au=outer_radius_au,
        eigen_cut_method="components",
        min_eigenvalue=None,
        spatial_method=spatial_method,
        spectral_method=spectral_method,
        beam_correct=beam_correct,
    )
