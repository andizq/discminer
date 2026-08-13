import numpy as np
from astropy import units as u
import pytest

from discminer.pca import analysis
from discminer.pca.artifact import read_pca_artifact, write_pca_artifact
from test_pca_artifact import make_source_header


class FakeData:
    def __init__(self, filename, distance):
        self.data = np.arange(12, dtype=float).reshape(3, 2, 2)
        self.header = make_source_header()
        self.vchannels = np.array([-1.0, 0.0, 1.0])


class FakePCA:
    def __init__(self, hdu, distance):
        self.eigvecs = np.eye(3)
        self.eigvals = np.array([6.0, 3.0, 1.0])
        self.cov_matrix = np.diag(self.eigvals)

    def compute_pca(self, mean_sub, n_eigs, show_progress):
        self.n_eigs = n_eigs

    def find_spatial_widths(self, **kwargs):
        pass

    def find_spectral_widths(self, method):
        pass

    def autocorr_images(self, n_eigs):
        return np.full((n_eigs, 2, 2), 2.0)

    def noise_ACF(self):
        return np.full((2, 2), 0.25)

    def autocorr_spec(self, n_eigs):
        return np.full((3, n_eigs), 0.5)

    def eigimages(self, n_eigs):
        return np.arange(4 * n_eigs, dtype=float).reshape(n_eigs, 2, 2)

    def spatial_width(self, unit):
        return np.array([10.0, 5.0]) * u.au

    def spatial_width_error(self, unit):
        return np.array([1.0, 0.5]) * u.au

    def spectral_width(self, unit):
        return np.array([0.4, 0.2]) * (u.km / u.s)

    def spectral_width_error(self, unit):
        return np.array([0.04, 0.02]) * (u.km / u.s)


_MASKED_CUBE_RNG = np.random.default_rng(3)
MASKED_CUBE = _MASKED_CUBE_RNG.normal(size=(6, 5, 5))
# This varying channel mask makes TurbuStat's pairwise NaN covariance
# indefinite when the NaNs are passed through unchanged.
MASKED_CUBE[_MASKED_CUBE_RNG.random(MASKED_CUBE.shape) < 0.45] = np.nan


class MaskedFakeData:
    def __init__(self, filename, distance):
        self.data = MASKED_CUBE.copy()
        self.header = make_source_header(n_channels=6, ny=5, nx=5)
        self.vchannels = np.linspace(-2.5, 2.5, 6)


class DecompositionFakePCA:
    """Small PCA double preserving TurbuStat's data conventions."""

    def __init__(self, hdu, distance):
        self.data = np.asarray(hdu.data, dtype=float)
        assert np.all(np.isfinite(self.data))

    def compute_pca(self, mean_sub, n_eigs, show_progress):
        self.n_eigs = n_eigs
        if mean_sub:
            means = np.mean(self.data, axis=(1, 2), keepdims=True)
            self.decomposition_data = self.data - means
            divisor = self.data.shape[1] * self.data.shape[2] - 1
        else:
            self.decomposition_data = self.data
            divisor = self.data.shape[1] * self.data.shape[2]

        flattened = self.decomposition_data.reshape(self.data.shape[0], -1)
        self.cov_matrix = flattened @ flattened.T / divisor
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        order = np.argsort(eigenvalues)[::-1]
        self.eigvals = eigenvalues[order]
        self.eigvecs = eigenvectors[:, order]

    def find_spatial_widths(self, **kwargs):
        pass

    def find_spectral_widths(self, method):
        pass

    def autocorr_images(self, n_eigs):
        return np.ones((n_eigs,) + self.data.shape[1:])

    def noise_ACF(self):
        return np.zeros(self.data.shape[1:])

    def autocorr_spec(self, n_eigs):
        return np.ones((self.data.shape[0], n_eigs))

    def eigimages(self, n_eigs):
        images = np.einsum(
            "ci,cyx->iyx",
            self.eigvecs[:, :n_eigs],
            self.decomposition_data,
        )
        return images.transpose(0, 2, 1)

    def spatial_width(self, unit):
        return np.ones(self.n_eigs) * u.au

    def spatial_width_error(self, unit):
        return np.full(self.n_eigs, 0.1) * u.au

    def spectral_width(self, unit):
        return np.ones(self.n_eigs) * (u.km / u.s)

    def spectral_width_error(self, unit):
        return np.full(self.n_eigs, 0.1) * (u.km / u.s)


@pytest.fixture
def masked_pca_result(monkeypatch):
    import discminer.core

    monkeypatch.setattr(discminer.core, "Data", MaskedFakeData)
    monkeypatch.setattr(
        analysis,
        "_import_turbustat_pca",
        lambda: DecompositionFakePCA,
    )
    return analysis.run_pca(
        "masked_cube.fits",
        100.0 * u.pc,
        n_components=-1,
        mean_sub=True,
        show_progress=False,
    )


def test_run_pca_retains_width_diagnostic_autocorrelations(monkeypatch):
    import discminer.core

    monkeypatch.setattr(discminer.core, "Data", FakeData)
    monkeypatch.setattr(analysis, "_import_turbustat_pca", lambda: FakePCA)

    result = analysis.run_pca(
        "cube.fits",
        100.0 * u.pc,
        n_components=2,
        show_progress=False,
    )

    np.testing.assert_allclose(result.spatial_autocorrelation, 1.75)
    np.testing.assert_allclose(result.spectral_autocorrelation, 0.5)
    assert result.spatial_autocorrelation.shape == (2, 2, 2)
    assert result.spectral_autocorrelation.shape == (3, 2)


def test_masked_cube_artifact_has_no_negative_eigenvalues(
    masked_pca_result,
    tmp_path,
):
    filename = tmp_path / "pca_masked_cube.fits"
    write_pca_artifact(masked_pca_result, filename)

    artifact = read_pca_artifact(filename)

    assert np.all(artifact.eigenvalues >= 0.0)


def test_masked_cube_reconstruction_restores_original_nans(
    masked_pca_result,
):
    reconstructed = masked_pca_result.reconstruct()

    np.testing.assert_array_equal(
        np.isnan(reconstructed),
        ~np.isfinite(MASKED_CUBE),
    )


def test_masked_cube_complete_reconstruction_matches_finite_input(
    masked_pca_result,
):
    reconstructed = masked_pca_result.reconstruct()
    valid = np.isfinite(MASKED_CUBE)
    zero_filled = np.nan_to_num(MASKED_CUBE, nan=0.0)

    np.testing.assert_allclose(reconstructed[valid], MASKED_CUBE[valid])
    np.testing.assert_allclose(
        masked_pca_result.channel_mean,
        np.mean(zero_filled, axis=(1, 2)),
    )
