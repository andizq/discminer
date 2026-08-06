import numpy as np
from astropy.io import fits

from discminer.pca.artifact import (
    PCAResult,
    read_pca_artifact,
    write_pca_artifact,
)


def make_source_header(n_channels=3, ny=2, nx=2):
    return fits.Header(
        {
            "NAXIS": 3,
            "NAXIS1": nx,
            "NAXIS2": ny,
            "NAXIS3": n_channels,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "VRAD",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CUNIT3": "km/s",
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRPIX3": 1.0,
            "CRVAL1": 10.0,
            "CRVAL2": -30.0,
            "CRVAL3": -1.0,
            "CDELT1": -1.0 / 3600.0,
            "CDELT2": 1.0 / 3600.0,
            "CDELT3": 1.0,
            "BUNIT": "K",
        }
    )


def make_result(mean_sub=False):
    cube = np.arange(12, dtype=float).reshape(3, 2, 2)
    channel_mean = (
        np.mean(cube, axis=(1, 2)) if mean_sub else np.zeros(3)
    )
    eigenimages = cube - channel_mean[:, np.newaxis, np.newaxis]
    widths = np.array([10.0, 5.0, np.nan])
    return PCAResult(
        source_header=make_source_header(),
        source_path="stacked_cube.fits",
        distance_pc=100.0,
        mean_sub=mean_sub,
        selected_components=2,
        eigenimages=eigenimages,
        eigenvectors=np.eye(3),
        eigenvalues=np.array([6.0, 3.0, 1.0]),
        covariance=np.diag([6.0, 3.0, 1.0]),
        velocity=np.array([-1.0, 0.0, 1.0]),
        channel_mean=channel_mean,
        spatial_width=widths,
        spatial_width_error=np.array([1.0, 0.5, np.nan]),
        spectral_width=np.array([0.4, 0.2, np.nan]),
        spectral_width_error=np.array([0.04, 0.02, np.nan]),
        valid_mask=np.ones_like(cube, dtype=bool),
        outer_radius_au=50.0,
        beam_correct=True,
    )


def test_artifact_round_trip(tmp_path):
    expected = make_result()
    filename = tmp_path / "pca.fits"

    write_pca_artifact(expected, filename)
    actual = read_pca_artifact(filename)

    assert actual.source_path == expected.source_path
    assert actual.distance_pc == expected.distance_pc
    assert actual.selected_components == expected.selected_components
    assert actual.outer_radius_au == expected.outer_radius_au
    assert actual.source_header["CTYPE3"] == "VRAD"
    np.testing.assert_allclose(actual.eigenimages, expected.eigenimages)
    np.testing.assert_allclose(actual.eigenvectors, expected.eigenvectors)
    np.testing.assert_allclose(actual.eigenvalues, expected.eigenvalues)
    np.testing.assert_allclose(actual.covariance, expected.covariance)
    np.testing.assert_allclose(
        actual.spatial_width, expected.spatial_width, equal_nan=True
    )
    np.testing.assert_array_equal(actual.valid_mask, expected.valid_mask)


def test_artifact_overwrites_by_default(tmp_path):
    filename = tmp_path / "pca.fits"
    write_pca_artifact(make_result(), filename)
    write_pca_artifact(make_result(), filename)

    assert filename.is_file()


def test_artifact_component_axis_is_not_a_velocity_axis(tmp_path):
    filename = tmp_path / "pca.fits"
    write_pca_artifact(make_result(), filename)

    with fits.open(filename) as hdul:
        header = hdul["EIGENIMAGES"].header
        assert header["CTYPE3"] == "PCA-COMP"
        assert header["CRVAL3"] == 0.0
        assert header["CDELT3"] == 1.0
        assert "CUNIT3" not in header
        assert "RESTFRQ" not in header
        assert hdul["MEAN"].data.shape == (3,)
