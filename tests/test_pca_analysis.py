import numpy as np
from astropy import units as u

from discminer.pca import analysis
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
