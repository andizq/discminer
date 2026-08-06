import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from discminer.pca.cli import _default_output  # noqa: E402
from discminer.pca.plotting import (  # noqa: E402
    covariance_velocity_window,
    plot_covariance,
)
from test_pca_artifact import make_result  # noqa: E402


def test_covariance_window_uses_central_velocity_span():
    velocity = np.linspace(-5.0, 5.0, 101)

    lower, upper = covariance_velocity_window(
        velocity, central_fraction=0.4
    )

    assert lower == -2.0
    assert upper == 2.0


def test_covariance_window_accepts_physical_half_width():
    velocity = np.linspace(-5.0, 5.0, 101)

    lower, upper = covariance_velocity_window(
        velocity, velocity_limit=1.25
    )

    assert lower == -1.25
    assert upper == 1.25


def test_covariance_plot_is_written(tmp_path):
    result = make_result()
    output = tmp_path / "covariance.png"

    plot_covariance(result, output)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_default_pca_products_use_prefixes_without_duplication():
    artifact = _default_output("cube_data.fits", None, ".fits")
    covariance = _default_output(artifact, "covariance", ".png")

    assert artifact.name == "pca_cube_data.fits"
    assert covariance.name == "pca_covariance_cube_data.png"
