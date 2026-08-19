import argparse

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402

from discminer.pca.artifact import PCAResult  # noqa: E402
from discminer.pca.cli import add_pca_parser  # noqa: E402
from discminer.pca.spectral_characterization import (  # noqa: E402
    TEMPLATE_NAMES,
    build_spectral_templates,
    characterize_eigenvectors,
    characterize_spectra,
    parity_correlation,
    plot_spectral_characterization,
    write_spectral_characterization,
)


def make_spectral_result():
    velocity = np.linspace(-5.0, 5.0, 101)
    reference = np.exp(-0.5 * (velocity / 0.7) ** 2)
    reference /= np.linalg.norm(reference)
    eigenvectors = np.zeros((velocity.size, 5))
    eigenvectors[:, 0] = reference
    eigenimages = np.arange(5 * 16, dtype=float).reshape(5, 4, 4)
    return PCAResult(
        source_header=fits.Header(
            {
                "NAXIS": 3,
                "NAXIS1": 4,
                "NAXIS2": 4,
                "NAXIS3": velocity.size,
                "CTYPE3": "VRAD",
                "CUNIT3": "km/s",
            }
        ),
        source_path="stacked_cube.fits",
        distance_pc=100.0,
        mean_sub=False,
        selected_components=5,
        eigenimages=eigenimages,
        eigenvectors=eigenvectors,
        eigenvalues=np.array([50.0, 20.0, 10.0, 5.0, 1.0]),
        covariance=np.eye(velocity.size),
        velocity=velocity,
        channel_mean=np.zeros(velocity.size),
        spatial_width=np.ones(5),
        spatial_width_error=np.full(5, 0.1),
        spectral_width=np.ones(5),
        spectral_width_error=np.full(5, 0.1),
        valid_mask=np.ones((velocity.size, 4, 4), dtype=bool),
        spatial_autocorrelation=np.ones((5, 4, 4)),
        spectral_autocorrelation=np.ones((velocity.size, 5)),
    )


def test_parity_correlation_identifies_even_and_odd_spectra():
    velocity = np.linspace(-2.0, 2.0, 41)

    assert parity_correlation(velocity, velocity**2) == pytest.approx(1.0)
    assert parity_correlation(velocity, velocity) == pytest.approx(-1.0)


def test_empirical_templates_recover_matching_eigenvectors():
    result = make_spectral_result()
    templates = build_spectral_templates(result)
    result.eigenvectors = templates.values.copy()

    table = characterize_eigenvectors(
        result,
        components=range(5),
        templates=templates,
    )

    for component, name in enumerate(TEMPLATE_NAMES):
        assert table[f"overlap_{name}"][component] == pytest.approx(1.0)
        assert table["dominant_template"][component] == name
        assert table["template_subspace_fraction"][component] == (
            pytest.approx(1.0)
        )
    assert table["parity_correlation"][0] == pytest.approx(1.0)
    assert table["parity_correlation"][1] == pytest.approx(-1.0)
    assert table["parity_correlation"][2] == pytest.approx(1.0)


def test_spectral_characterization_outputs_are_written(tmp_path):
    result = make_spectral_result()
    templates = build_spectral_templates(result)
    result.eigenvectors = templates.values.copy()
    table, templates = characterize_spectra(result, components=range(5))
    table_output = tmp_path / "spectral.ecsv"
    plot_output = tmp_path / "spectral.svg"

    write_spectral_characterization(table, table_output)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        plot_spectral_characterization(
            result,
            table,
            templates,
            plot_output,
        )

    restored = Table.read(table_output)
    assert restored.meta["reference_component"] == 0
    assert "parity_correlation" in restored.colnames
    figure_text = plot_output.read_text()
    assert "Absolute template overlap" in figure_text
    assert "Centroid" in figure_text


def test_spectral_template_settings_are_validated():
    result = make_spectral_result()

    with pytest.raises(ValueError, match="must be odd"):
        build_spectral_templates(result, smoothing_window=10)
    with pytest.raises(ValueError, match="at least 4"):
        build_spectral_templates(result, smoothing_order=3)


def test_characterize_spectra_parser_options():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_pca_parser(subparsers)

    args = parser.parse_args(
        [
            "pca",
            "characterize-spectra",
            "pca_cube.fits",
            "--components",
            "1,2,3",
            "--reference-component",
            "0",
            "--smoothing-window",
            "15",
            "--center-velocity",
            "0.1",
        ]
    )

    assert args.pca_command == "characterize-spectra"
    assert args.components == ["1,2,3"]
    assert args.reference_component == 0
    assert args.smoothing_window == 15
    assert args.center_velocity == 0.1
