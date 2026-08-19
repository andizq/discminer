import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy.table import Table  # noqa: E402

from discminer.pca.characterization import (  # noqa: E402
    _eigenimage_center,
    _phase_diagnostic_support,
    _polar_sampling_grid,
    acf_ellipse_metrics,
    acf_multipole_metrics,
    angular_mode_metrics,
    azimuthal_axisymmetric_fraction,
    azimuthal_mode_fraction,
    azimuthal_phase_coherence,
    characterize_result,
    core_characterization_paths,
    excursion_characterization_path,
    excursion_set_metrics,
    load_disc_ring_geometry,
    plot_core_characterization,
    plot_excursion_sets,
    write_characterization_table,
)
from discminer.pca.cli import (  # noqa: E402
    _resolve_characterization_parfiles,
    add_pca_parser,
)
from test_pca_artifact import make_result  # noqa: E402


def elliptical_acf(axis0=12.0, axis1=6.0, angle_deg=30.0):
    ycoord, xcoord = np.indices((101, 101), dtype=float)
    xcoord -= 50.0
    ycoord -= 50.0
    angle = np.deg2rad(angle_deg)
    xrot = xcoord * np.cos(angle) + ycoord * np.sin(angle)
    yrot = -xcoord * np.sin(angle) + ycoord * np.cos(angle)
    return np.exp(
        -0.5 * ((xrot / axis0) ** 2 + (yrot / axis1) ** 2)
    )


def angular_pattern(mode=None, amplitude=0.0, size=101, scale=15.0):
    ycoord, xcoord = np.indices((size, size), dtype=float)
    center = size // 2
    xcoord -= center
    ycoord -= center
    radius = np.hypot(xcoord, ycoord)
    angle = np.arctan2(ycoord, xcoord)
    radial = np.exp(-0.5 * (radius / scale) ** 2)
    if mode is None:
        return radial
    return radial * (1.0 + amplitude * np.cos(mode * angle))


def winding_m2_pattern(winding=0.7, size=101, scale=15.0):
    ycoord, xcoord = np.indices((size, size), dtype=float)
    center = size // 2
    xcoord -= center
    ycoord -= center
    radius = np.hypot(xcoord, ycoord)
    angle = np.arctan2(ycoord, xcoord)
    safe_radius = np.maximum(radius, 1.0)
    radial = np.exp(-0.5 * (radius / scale) ** 2)
    phase = 2.0 * (angle - winding * np.log(safe_radius))
    return radial * (1.0 + np.cos(phase))


def test_acf_ellipse_metrics_recovers_axis_ratio_and_angle():
    metrics = acf_ellipse_metrics(elliptical_acf())

    assert metrics["acf_axis_ratio"] == pytest.approx(0.5, abs=0.01)
    assert metrics["acf_pa_deg"] == pytest.approx(30.0, abs=1.0)
    assert metrics["acf_center_offset_pix"] < 0.01
    assert metrics["acf_ellipse_residual"] < 0.01
    assert metrics["acf_contour_points"] > 0


def test_acf_multipoles_recover_controlled_angular_modes():
    circular = acf_multipole_metrics(angular_pattern())
    quadrupole = acf_multipole_metrics(
        angular_pattern(mode=2, amplitude=0.6)
    )
    fourth_order = acf_multipole_metrics(
        angular_pattern(mode=4, amplitude=0.4)
    )

    assert circular["acf_max_q2"] < 0.01
    assert circular["acf_max_q4"] < 0.01
    assert quadrupole["acf_max_q2"] == pytest.approx(0.3, abs=0.02)
    assert quadrupole["acf_max_q4"] < 0.02
    assert fourth_order["acf_max_q2"] < 0.02
    assert fourth_order["acf_max_q4"] == pytest.approx(0.2, abs=0.02)


def test_azimuthal_mode_fraction_separates_strength_from_phase():
    size = 101
    center = (size // 2, size // 2)
    pure_m2 = angular_pattern(mode=2, amplitude=1.0)
    mixed = pure_m2 + angular_pattern(mode=3, amplitude=1.0)
    pure_m3 = angular_pattern(mode=3, amplitude=1.0)

    m2_fraction = azimuthal_mode_fraction(pure_m2, center=center)
    mixed_fraction = azimuthal_mode_fraction(mixed, center=center)
    m3_fraction = azimuthal_mode_fraction(pure_m3, center=center)

    assert m2_fraction[
        "eigenimage_f_m2_nonaxisymmetric"
    ] > 0.99
    assert m2_fraction["eigenimage_f_m2_total"] == pytest.approx(
        1.0 / 3.0,
        abs=0.02,
    )
    assert mixed_fraction[
        "eigenimage_f_m2_nonaxisymmetric"
    ] == pytest.approx(0.5, abs=0.03)
    assert mixed_fraction["eigenimage_f_m2_total"] == pytest.approx(
        0.1,
        abs=0.02,
    )
    assert m3_fraction[
        "eigenimage_f_m2_nonaxisymmetric"
    ] < 0.01
    assert m3_fraction["eigenimage_f_m2_total"] < 0.01


def test_axisymmetric_fraction_measures_ring_mean_power():
    radial = angular_pattern()
    radial_plus_m2 = angular_pattern(mode=2, amplitude=1.0)
    pure_m2 = radial_plus_m2 - radial

    axisymmetric = azimuthal_axisymmetric_fraction(radial)
    mixed = azimuthal_axisymmetric_fraction(radial_plus_m2)
    nonaxisymmetric = azimuthal_axisymmetric_fraction(pure_m2)
    negative = azimuthal_axisymmetric_fraction(-radial_plus_m2)

    assert axisymmetric["eigenimage_f_m0"] > 0.999
    assert mixed["eigenimage_f_m0"] == pytest.approx(2.0 / 3.0, abs=0.02)
    assert nonaxisymmetric["eigenimage_f_m0"] < 0.001
    assert negative["eigenimage_f_m0"] == pytest.approx(
        mixed["eigenimage_f_m0"]
    )


def test_disc_plane_rings_follow_parfile_orientation_and_surface(tmp_path):
    parfile = tmp_path / "parfile.json"
    parfile.write_text(
        json.dumps(
            {
                "metadata": {"dpc": 1.0, "kind": ["allfree"]},
                "params": {
                    "orientation": {
                        "incl": np.pi / 3.0,
                        "PA": 0.0,
                        "xc": 0.0,
                        "yc": 0.0,
                    },
                    "height_upper": {
                        "z0": 10.0,
                        "p": 0.0,
                        "Rb": 1000.0,
                        "q": 2.0,
                    },
                    "height_lower": {
                        "z0": 10.0,
                        "p": 0.0,
                        "Rb": 1000.0,
                        "q": 2.0,
                    },
                },
                "custom": {},
            }
        )
    )
    header = {
        "CDELT1": -1.0 / 3600.0,
        "CDELT2": 1.0 / 3600.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }
    midplane = load_disc_ring_geometry(parfile, header, surface="midplane")
    upper = load_disc_ring_geometry(parfile, header, surface="upper")

    radii, _, midplane_coords = _polar_sampling_grid(
        (101, 101),
        center=(50, 50),
        minimum_radius=2.0,
        ring_geometry=midplane,
    )
    _, _, upper_coords = _polar_sampling_grid(
        (101, 101),
        center=(50, 50),
        minimum_radius=2.0,
        ring_geometry=upper,
    )
    ring_index = np.flatnonzero(radii == 10.0)[0]
    assert np.ptp(midplane_coords[1, ring_index]) == pytest.approx(20.0)
    assert np.ptp(midplane_coords[0, ring_index]) == pytest.approx(10.0)
    assert np.mean(midplane_coords[0, ring_index]) == pytest.approx(50.0)
    assert np.mean(upper_coords[0, ring_index]) == pytest.approx(
        50.0 - 10.0 * np.sin(np.pi / 3.0),
        abs=0.01,
    )


def test_deprojection_recovers_axisymmetric_disc_power(tmp_path):
    parfile = tmp_path / "parfile.json"
    parfile.write_text(
        json.dumps(
            {
                "metadata": {"dpc": 1.0, "kind": ["allfree"]},
                "params": {
                    "orientation": {
                        "incl": np.pi / 3.0,
                        "PA": 0.0,
                        "xc": 0.0,
                        "yc": 0.0,
                    },
                    "height_upper": {},
                    "height_lower": {},
                },
                "custom": {},
            }
        )
    )
    header = {
        "CDELT1": -1.0 / 3600.0,
        "CDELT2": 1.0 / 3600.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }
    geometry = load_disc_ring_geometry(
        parfile,
        header,
        surface="midplane",
    )
    ycoord, xcoord = np.indices((101, 101), dtype=float)
    xcoord -= 50.0
    ycoord = (ycoord - 50.0) / np.cos(np.pi / 3.0)
    image = np.exp(-0.5 * (np.hypot(xcoord, ycoord) / 15.0) ** 2)

    circular = azimuthal_axisymmetric_fraction(image, center=(50, 50))
    deprojected = azimuthal_axisymmetric_fraction(
        image,
        center=(50, 50),
        ring_geometry=geometry,
    )

    assert deprojected["eigenimage_f_m0"] > 0.999
    assert deprojected["eigenimage_f_m0"] > circular["eigenimage_f_m0"]


def test_phase_coherence_recovers_fixed_and_winding_m2_patterns():
    center = (50, 50)
    fixed = azimuthal_phase_coherence(
        angular_pattern(mode=2, amplitude=1.0),
        center=center,
    )
    winding = azimuthal_phase_coherence(
        winding_m2_pattern(winding=0.7),
        center=center,
    )

    assert fixed["eigenimage_m2_phase_coherence"] > 0.99
    assert fixed["eigenimage_m2_phase_slope_logr"] == pytest.approx(
        0.0,
        abs=0.02,
    )
    assert winding["eigenimage_m2_phase_coherence"] > 0.98
    assert winding["eigenimage_m2_phase_slope_logr"] == pytest.approx(
        -1.4,
        abs=0.08,
    )


def test_angular_mode_metrics_find_dominant_mode_and_orientation_slope():
    center = (50, 50)
    fixed_m3 = angular_mode_metrics(
        angular_pattern(mode=3, amplitude=1.0),
        maximum_mode=6,
        center=center,
    )
    winding_m2 = angular_mode_metrics(
        winding_m2_pattern(winding=0.7),
        maximum_mode=6,
        center=center,
    )

    assert fixed_m3["eigenimage_m_peak"] == 3
    assert fixed_m3["eigenimage_f_peak_fitted"] > 0.99
    assert fixed_m3[
        "eigenimage_f_peak_nonaxisymmetric"
    ] > 0.99
    assert fixed_m3["eigenimage_f_peak_total"] == pytest.approx(
        1.0 / 3.0,
        abs=0.02,
    )
    assert fixed_m3["eigenimage_mode_entropy"] < 0.02
    assert fixed_m3[
        "eigenimage_angular_model_fraction_nonaxisymmetric"
    ] > 0.99
    assert fixed_m3[
        "eigenimage_angular_model_fraction_total"
    ] == pytest.approx(1.0 / 3.0, abs=0.02)
    assert fixed_m3["eigenimage_f_nonaxisymmetric"] == pytest.approx(
        1.0 / 3.0,
        abs=0.02,
    )
    assert fixed_m3[
        "eigenimage_angular_unresolved_fraction_total"
    ] < 0.01
    assert (
        fixed_m3["eigenimage_angular_model_fraction_total"]
        + fixed_m3["eigenimage_angular_unresolved_fraction_total"]
    ) == pytest.approx(
        fixed_m3["eigenimage_f_nonaxisymmetric"],
        abs=1.0e-12,
    )
    assert fixed_m3["eigenimage_mpeak_phase_coherence"] > 0.99
    assert fixed_m3[
        "eigenimage_mpeak_orientation_slope_logr"
    ] == pytest.approx(0.0, abs=0.02)

    assert winding_m2["eigenimage_m_peak"] == 2
    assert winding_m2["eigenimage_f_peak_fitted"] > 0.99
    assert winding_m2["eigenimage_mpeak_phase_coherence"] > 0.98
    assert winding_m2[
        "eigenimage_mpeak_orientation_slope_logr"
    ] == pytest.approx(0.7, abs=0.05)


def test_excursion_metrics_measure_compactness_and_topology():
    ycoord, xcoord = np.indices((101, 101), dtype=float)
    xcoord -= 50.0
    ycoord -= 50.0
    radius = np.hypot(xcoord, ycoord)
    circle = np.exp(-0.5 * (radius / 15.0) ** 2)
    annulus = np.exp(-0.5 * ((radius - 25.0) / 3.0) ** 2)
    blob0 = np.exp(-0.5 * ((xcoord - 20.0) ** 2 + ycoord**2) / 8.0**2)
    blob1 = np.exp(-0.5 * ((xcoord + 20.0) ** 2 + ycoord**2) / 8.0**2)
    two_blobs = np.maximum(blob0, blob1)

    circle_metrics = excursion_set_metrics(circle)
    negative_metrics = excursion_set_metrics(-circle)
    annulus_metrics = excursion_set_metrics(annulus)
    blob_metrics = excursion_set_metrics(two_blobs)

    assert circle_metrics["eigenimage_euler_characteristic"] == 1
    assert annulus_metrics["eigenimage_euler_characteristic"] == 0
    assert blob_metrics["eigenimage_euler_characteristic"] == 2
    assert circle_metrics["eigenimage_compactness"] > 0.9
    assert circle_metrics["eigenimage_compactness"] > blob_metrics[
        "eigenimage_compactness"
    ]
    assert negative_metrics["eigenimage_compactness"] == pytest.approx(
        circle_metrics["eigenimage_compactness"]
    )


def test_characterization_keeps_eigenvalues_independent_of_widths():
    result = make_result()
    result.selected_components = 2
    result.spatial_autocorrelation = np.stack(
        [elliptical_acf(), elliptical_acf(axis0=10.0, axis1=8.0)]
    )

    table = characterize_result(result, components=[0, 1, 2])

    np.testing.assert_allclose(table["variance_fraction"], [0.6, 0.3, 0.1])
    assert np.isnan(table["cumulative_variance_no_pc0"][0])
    np.testing.assert_allclose(
        table["cumulative_variance_no_pc0"][1:], [0.3, 0.4]
    )
    assert np.isnan(table["variance_fraction_no_pc0_renormalized"][0])
    np.testing.assert_allclose(
        table["variance_fraction_no_pc0_renormalized"][1:],
        [0.75, 0.25],
    )
    np.testing.assert_allclose(
        table["cumulative_variance_no_pc0_renormalized"][1:],
        [0.75, 1.0],
    )
    np.testing.assert_allclose(
        table["cumulative_variance_percent_no_pc0_renormalized"][1:],
        [75.0, 100.0],
    )
    assert np.isfinite(table["acf_axis_ratio"][0])
    assert np.isnan(table["acf_axis_ratio"][2])
    assert "acf_max_q2" in table.colnames
    assert "acf_max_q4" in table.colnames
    assert "eigenimage_f_m0" in table.colnames
    assert "eigenimage_f_m2_total" in table.colnames
    assert "eigenimage_f_m2_nonaxisymmetric" in table.colnames
    assert "eigenimage_m2_phase_coherence" in table.colnames
    assert "eigenimage_compactness" in table.colnames
    assert "eigenimage_euler_characteristic" in table.colnames
    assert "eigenimage_m_peak" in table.colnames
    assert "eigenimage_f_peak_total" in table.colnames
    assert "eigenimage_f_peak_nonaxisymmetric" in table.colnames
    assert "eigenimage_f_peak_fitted" in table.colnames
    assert "eigenimage_mode_entropy" in table.colnames
    assert "eigenimage_angular_model_fraction_total" in table.colnames
    assert (
        "eigenimage_angular_model_fraction_nonaxisymmetric"
        in table.colnames
    )
    assert "eigenimage_f_nonaxisymmetric" in table.colnames
    assert (
        "eigenimage_angular_unresolved_fraction_total"
        in table.colnames
    )
    assert "eigenimage_mpeak_orientation_slope_logr" in table.colnames
    assert np.all(table["eigenimage_ring_geometry"] == "circular_sky")
    assert "eigenimage_ring_radial_bins" in table.colnames


def test_eigenimage_center_uses_in_frame_wcs_zero_offset():
    result = make_result()
    result.eigenimages = np.zeros((3, 190, 190))
    result.source_header.update(
        {
            "NAXIS1": 190,
            "NAXIS2": 190,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRVAL1": 94.0 / 3600.0,
            "CRVAL2": -94.0 / 3600.0,
            "CDELT1": -1.0 / 3600.0,
            "CDELT2": 1.0 / 3600.0,
        }
    )

    ycenter, xcenter = _eigenimage_center(result)

    assert xcenter == pytest.approx(94.0)
    assert ycenter == pytest.approx(94.0)


def test_characterization_warns_when_no_radial_rings_are_available():
    result = make_result()
    result.eigenimages = np.zeros((3, 190, 190))
    result.valid_mask = np.ones((3, 190, 190), dtype=bool)

    with pytest.warns(RuntimeWarning, match="No complete radial rings"):
        table = characterize_result(result, components=[0])

    assert table["eigenimage_ring_radial_bins"][0] == 0


def test_characterization_outputs_are_written(tmp_path):
    result = make_result()
    result.selected_components = 2
    result.spatial_autocorrelation = np.stack(
        [elliptical_acf(), elliptical_acf(axis0=10.0, axis1=8.0)]
    )
    table = characterize_result(result, components=[0, 1, 2])
    table_output = tmp_path / "characterization.ecsv"

    write_characterization_table(table, table_output)

    restored = Table.read(table_output, format="ascii.ecsv")
    assert len(restored) == 3
    assert table_output.stat().st_size > 0


def test_phase_diagnostic_support_applies_provisional_quality_rule():
    table = Table(
        {
            "eigenimage_f_peak_total": [0.10, 0.09, 0.20, 0.20, 0.20],
            "eigenimage_mode_entropy": [0.85, 0.50, 0.86, 0.50, 0.50],
            "eigenimage_mpeak_phase_coherence": [
                0.70,
                0.90,
                0.90,
                0.69,
                0.90,
            ],
            "eigenimage_mpeak_phase_rings": [10, 20, 20, 20, 9],
        }
    )

    np.testing.assert_array_equal(
        _phase_diagnostic_support(table),
        [True, False, False, True, True],
    )
    np.testing.assert_array_equal(
        _phase_diagnostic_support(table, require_slope_quality=True),
        [True, False, False, False, False],
    )


def test_core_characterization_writes_four_grouped_figures(tmp_path):
    result = make_result()
    result.selected_components = 2
    result.spatial_autocorrelation = np.stack(
        [elliptical_acf(), elliptical_acf(axis0=10.0, axis1=8.0)]
    )
    table = characterize_result(result, components=[0, 1, 2])
    prefix = tmp_path / "pca_core.png"

    outputs = plot_core_characterization(table, prefix)

    paths = core_characterization_paths(prefix)
    assert set(outputs) == {
        "variance",
        "acf",
        "eigenimage",
        "angularmode",
    }
    assert outputs["variance"] == paths["component_importance"]
    assert outputs["acf"] == paths["acf_morphology"]
    assert outputs["eigenimage"] == paths["eigenimage_structure"]
    assert outputs["angularmode"] == paths["angularmode"]
    for output in outputs.values():
        assert output.stat().st_size > 0

    selected = plot_core_characterization(
        table,
        tmp_path / "selected",
        groups="angularmode",
        angular_normalization="nonaxisymmetric",
    )
    assert set(selected) == {"angularmode"}
    assert selected["angularmode"].stat().st_size > 0

    with pytest.raises(ValueError, match="angular_normalization"):
        plot_core_characterization(
            table,
            tmp_path / "invalid",
            angular_normalization="invalid",
        )


def test_excursion_plot_highlights_selected_eigenimage_regions(tmp_path):
    result = make_result()
    output = excursion_characterization_path(tmp_path / "pca_core.png")

    restored = plot_excursion_sets(
        [result, result],
        ["smooth", "spiral"],
        [[0, 1], [1, 2]],
        output,
        percentile=85.0,
    )

    assert restored == output
    assert restored.name == "pca_core_eigenimage_excursions.png"
    assert restored.stat().st_size > 0


def test_characterize_parser_exposes_pc0_cumulative_flag():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_pca_parser(subparsers)

    args = parser.parse_args(
        [
            "pca",
            "characterize",
            "pca_cube.fits",
            "--include-pc0-cumulative",
            "--plot-group",
            "eigenimage",
            "--azimuth-samples",
            "180",
            "--excursion-percentile",
            "85",
            "--plot-excursions",
            "--maximum-angular-mode",
            "5",
            "--angular-normalization",
            "nonaxisymmetric",
            "--parfile",
            "fit/parfile.json",
            "--surface",
            "lower",
        ]
    )

    assert args.pca_command == "characterize"
    assert args.include_pc0_cumulative
    assert args.plot_group == "eigenimage"
    assert args.azimuth_samples == 180
    assert args.excursion_percentile == 85
    assert args.plot_excursions
    assert args.maximum_angular_mode == 5
    assert args.angular_normalization == "nonaxisymmetric"
    assert args.parfile == ["fit/parfile.json"]
    assert args.deprojection_surface == "lower"

    circular = parser.parse_args(
        ["pca", "characterize", "pca_cube.fits", "--circular-deproj"]
    )
    assert circular.circular_deproj
    assert circular.angular_normalization == "total"
    assert not circular.plot_excursions


def test_characterize_parfiles_are_discovered_per_artifact(tmp_path):
    artifact_directories = [tmp_path / "smooth", tmp_path / "spiral"]
    for directory in artifact_directories:
        directory.mkdir()
        (directory / "parfile.json").write_text("{}")
    artifacts = [directory / "pca.fits" for directory in artifact_directories]

    discovered = _resolve_characterization_parfiles(artifacts)
    assert discovered == [
        directory / "parfile.json" for directory in artifact_directories
    ]
    assert _resolve_characterization_parfiles(
        artifacts,
        circular_sky=True,
    ) == [None, None]
    assert _resolve_characterization_parfiles(
        artifacts,
        explicit_parfiles=["shared.json"],
    ) == [Path("shared.json"), Path("shared.json")]

    with pytest.raises(ValueError, match="one shared path or one path per"):
        _resolve_characterization_parfiles(
            artifacts,
            explicit_parfiles=["a.json", "b.json", "c.json"],
        )


def test_characterize_parser_exposes_plot_group_prefix():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_pca_parser(subparsers)

    args = parser.parse_args(
        [
            "pca",
            "characterize",
            "pca_cube.fits",
            "--plot-group",
            "angularmode",
            "--plot-prefix",
            "figures/pca_core",
        ]
    )

    assert args.plot_group == "angularmode"
    assert args.plot_prefix == "figures/pca_core"


@pytest.mark.parametrize("option", ["--plot-batch", "--legacy-plot-batch"])
def test_characterize_parser_rejects_removed_batch_options(option):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_pca_parser(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "pca",
                "characterize",
                "pca_cube.fits",
                option,
                "first",
            ]
        )
