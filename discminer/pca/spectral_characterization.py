"""Model-independent spectral characterization of PCA eigenvectors."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.signal import savgol_filter

from discminer.plottools import use_discminer_style

from .artifact import PCAResult


TEMPLATE_NAMES = (
    "intensity",
    "centroid",
    "linewidth",
    "third_derivative",
    "fourth_derivative",
)

TEMPLATE_LABELS = {
    "intensity": "Intensity",
    "centroid": "Centroid",
    "linewidth": "Linewidth",
    "third_derivative": "Third derivative",
    "fourth_derivative": "Fourth derivative",
}


@dataclass
class SpectralTemplates:
    """Orthonormal empirical line-profile response templates."""

    velocity: np.ndarray
    reference_profile: np.ndarray
    values: np.ndarray
    center_velocity: float
    reference_component: int
    smoothing_window: int
    smoothing_order: int


def _unit_vector(values):
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Cannot normalize an empty spectral template")
    return values / norm


def _orthogonalize(values, previous):
    values = np.asarray(values, dtype=float).copy()
    for template in previous:
        values -= np.dot(values, template) * template
    return _unit_vector(values)


def _sorted_velocity(result):
    velocity = np.asarray(result.velocity, dtype=float)
    if np.any(~np.isfinite(velocity)):
        raise ValueError(
            "The PCA velocity axis must contain only finite values"
        )
    order = np.argsort(velocity)
    velocity = velocity[order]
    differences = np.diff(velocity)
    if np.any(differences <= 0.0):
        raise ValueError("The PCA velocity channels must be distinct")
    channel_width = float(np.median(differences))
    if not np.allclose(
        differences,
        channel_width,
        rtol=1.0e-5,
        atol=1.0e-10 * max(1.0, abs(channel_width)),
    ):
        raise ValueError(
            "Empirical derivative templates require a uniformly sampled "
            "velocity axis"
        )
    return velocity, order, channel_width


def build_spectral_templates(
    result: PCAResult,
    reference_component=0,
    smoothing_window=11,
    smoothing_order=4,
    center_velocity=None,
):
    """Build orthonormal empirical profile-response templates.

    The templates are constructed in the order intensity, centroid,
    linewidth, third derivative, and fourth derivative. Each new template is
    orthogonalized against the preceding templates, separating higher-order
    line-shape changes from lower-order responses.
    """

    reference_component = int(reference_component)
    result._validate_component_indices(
        [reference_component],
        "reference_component",
    )
    smoothing_window = int(smoothing_window)
    smoothing_order = int(smoothing_order)
    if smoothing_window <= smoothing_order:
        raise ValueError("smoothing_window must exceed smoothing_order")
    if smoothing_window % 2 != 1:
        raise ValueError("smoothing_window must be odd")
    if smoothing_order < 4:
        raise ValueError("smoothing_order must be at least 4")
    if smoothing_window > result.n_channels:
        raise ValueError(
            "smoothing_window cannot exceed the number of velocity channels"
        )

    velocity, order, channel_width = _sorted_velocity(result)
    reference = np.asarray(
        result.eigenvectors[order, reference_component],
        dtype=float,
    )
    if np.sum(reference) < 0.0:
        reference = -reference

    if center_velocity is None:
        if velocity[0] <= 0.0 <= velocity[-1]:
            center_velocity = 0.0
        else:
            center_velocity = velocity[np.argmax(np.abs(reference))]
    center_velocity = float(center_velocity)
    if not velocity[0] <= center_velocity <= velocity[-1]:
        raise ValueError("center_velocity must lie within the velocity axis")

    derivatives = [
        savgol_filter(
            reference,
            smoothing_window,
            smoothing_order,
            deriv=derivative,
            delta=channel_width,
            mode="interp",
        )
        for derivative in range(5)
    ]
    offset_velocity = velocity - center_velocity
    candidates = (
        derivatives[0],
        -derivatives[1],
        -offset_velocity * derivatives[1],
        -derivatives[3],
        derivatives[4],
    )

    templates = []
    for candidate in candidates:
        templates.append(_orthogonalize(candidate, templates))

    return SpectralTemplates(
        velocity=velocity,
        reference_profile=derivatives[0],
        values=np.column_stack(templates),
        center_velocity=center_velocity,
        reference_component=reference_component,
        smoothing_window=smoothing_window,
        smoothing_order=smoothing_order,
    )


def parity_correlation(velocity, values, center_velocity=0.0):
    """Return correlation with the spectrum reflected about a velocity."""

    velocity = np.asarray(velocity, dtype=float)
    values = np.asarray(values, dtype=float)
    if velocity.ndim != 1 or values.shape != velocity.shape:
        raise ValueError("velocity and values must be matching vectors")
    order = np.argsort(velocity)
    velocity = velocity[order]
    values = values[order]
    reflected_velocity = 2.0 * float(center_velocity) - velocity
    reflected = np.interp(
        reflected_velocity,
        velocity,
        values,
        left=np.nan,
        right=np.nan,
    )
    valid = np.isfinite(values) & np.isfinite(reflected)
    if np.count_nonzero(valid) < 2:
        return np.nan
    norm = np.linalg.norm(values[valid]) * np.linalg.norm(reflected[valid])
    if not np.isfinite(norm) or norm == 0.0:
        return np.nan
    return float(np.dot(values[valid], reflected[valid]) / norm)


def characterize_eigenvectors(
    result: PCAResult,
    components,
    templates: SpectralTemplates,
    artifact=None,
):
    """Measure parity and empirical-template overlaps for PCA components."""

    components = result._validate_component_indices(components, "components")
    _, velocity_order, _ = _sorted_velocity(result)
    eigenvectors = np.asarray(
        result.eigenvectors[velocity_order][:, components],
        dtype=float,
    )
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
    signed_overlaps = eigenvectors.T @ templates.values
    overlaps = np.abs(signed_overlaps)
    variance = np.asarray(result.variance_fraction, dtype=float)

    rows = []
    for row_index, component in enumerate(components):
        component_overlaps = overlaps[row_index]
        overlap_order = np.argsort(component_overlaps)[::-1]
        dominant_index = int(overlap_order[0])
        second_index = int(overlap_order[1])
        dominant_overlap = float(component_overlaps[dominant_index])
        second_overlap = float(component_overlaps[second_index])
        parity = parity_correlation(
            templates.velocity,
            eigenvectors[:, row_index],
            center_velocity=templates.center_velocity,
        )
        row = {
            "artifact": "" if artifact is None else str(artifact),
            "component": int(component),
            "eigenvalue": float(result.eigenvalues[component]),
            "variance_fraction": float(variance[component]),
            "variance_percent": float(100.0 * variance[component]),
            "parity_correlation": parity,
            "even_fraction_from_parity": float(0.5 * (1.0 + parity)),
            "odd_fraction_from_parity": float(0.5 * (1.0 - parity)),
            "template_subspace_fraction": float(
                np.sum(component_overlaps**2)
            ),
            "dominant_template": TEMPLATE_NAMES[dominant_index],
            "dominant_overlap": dominant_overlap,
            "second_template": TEMPLATE_NAMES[second_index],
            "second_overlap": second_overlap,
            "dominant_overlap_margin": (
                dominant_overlap - second_overlap
            ),
        }
        for template_index, name in enumerate(TEMPLATE_NAMES):
            row[f"overlap_{name}"] = float(
                component_overlaps[template_index]
            )
            row[f"signed_overlap_{name}"] = float(
                signed_overlaps[row_index, template_index]
            )
        rows.append(row)

    table = Table(rows=rows)
    table.meta.update(
        {
            "description": (
                "Model-independent PCA eigenspectrum characterization"
            ),
            "reference_component": templates.reference_component,
            "smoothing_window_channels": templates.smoothing_window,
            "smoothing_polynomial_order": templates.smoothing_order,
            "center_velocity_kms": templates.center_velocity,
            "channel_weighting": "uniform",
            "template_order": ",".join(TEMPLATE_NAMES),
            "template_overlap": "absolute cosine similarity",
        }
    )
    return table


def characterize_spectra(
    result: PCAResult,
    components,
    artifact=None,
    reference_component=0,
    smoothing_window=11,
    smoothing_order=4,
    center_velocity=None,
):
    """Build empirical templates and characterize selected eigenvectors."""

    templates = build_spectral_templates(
        result,
        reference_component=reference_component,
        smoothing_window=smoothing_window,
        smoothing_order=smoothing_order,
        center_velocity=center_velocity,
    )
    table = characterize_eigenvectors(
        result,
        components,
        templates,
        artifact=artifact,
    )
    return table, templates


def write_spectral_characterization(table, output, overwrite=True):
    """Write a spectral-characterization table in Astropy ECSV format."""

    output = Path(output)
    table.write(output, format="ascii.ecsv", overwrite=overwrite)
    return output


def _offset_traces(axis, velocity, traces, labels, colors=None):
    offsets = np.arange(len(labels) - 1, -1, -1, dtype=float)
    for index, (trace, label, offset) in enumerate(
        zip(traces.T, labels, offsets)
    ):
        scale = np.nanmax(np.abs(trace))
        normalized = trace if scale == 0.0 else trace / scale
        color = None if colors is None else colors[index]
        axis.plot(velocity, 0.38 * normalized + offset, color=color)
        axis.axhline(offset, color="0.85", linewidth=0.6, zorder=0)
    axis.set_yticks(offsets, labels=labels)
    axis.set_xlim(velocity[0], velocity[-1])
    axis.set_xlabel(r"Velocity [km s$^{-1}$]")


def plot_spectral_characterization(
    result: PCAResult,
    table,
    templates: SpectralTemplates,
    output,
    dpi=200,
    show=False,
):
    """Plot templates, aligned eigenspectra, and their overlap matrix."""

    use_discminer_style()
    components = np.asarray(table["component"], dtype=int)
    _, velocity_order, _ = _sorted_velocity(result)
    eigenvectors = result.eigenvectors[velocity_order][:, components].copy()
    overlap_matrix = np.column_stack(
        [
            np.asarray(table[f"overlap_{name}"], dtype=float)
            for name in TEMPLATE_NAMES
        ]
    )

    for row, component in enumerate(components):
        dominant_name = str(table["dominant_template"][row])
        signed = float(table[f"signed_overlap_{dominant_name}"][row])
        if signed < 0.0:
            eigenvectors[:, row] *= -1.0

    figure_height = max(7.0, 0.48 * len(components) + 4.0)
    fig = plt.figure(figsize=(13.0, figure_height), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    template_axis = fig.add_subplot(grid[0, 0])
    eigenvector_axis = fig.add_subplot(grid[1, 0])
    overlap_axis = fig.add_subplot(grid[:, 1])

    _offset_traces(
        template_axis,
        templates.velocity,
        templates.values,
        [TEMPLATE_LABELS[name] for name in TEMPLATE_NAMES],
    )
    template_axis.axvline(
        templates.center_velocity,
        color="0.5",
        linestyle=":",
    )
    template_axis.set_title("Empirical orthonormal templates")

    component_labels = [
        f"PC {component} ({float(table['variance_percent'][row]):.2f}%)"
        for row, component in enumerate(components)
    ]
    _offset_traces(
        eigenvector_axis,
        templates.velocity,
        eigenvectors,
        component_labels,
    )
    eigenvector_axis.axvline(
        templates.center_velocity,
        color="0.5",
        linestyle=":",
    )
    eigenvector_axis.set_title("Eigenspectra aligned to dominant templates")

    image = overlap_axis.imshow(
        overlap_matrix,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    overlap_axis.set_xticks(
        np.arange(len(TEMPLATE_NAMES)),
        labels=[TEMPLATE_LABELS[name] for name in TEMPLATE_NAMES],
        rotation=35,
        ha="right",
    )
    overlap_axis.set_yticks(
        np.arange(len(components)),
        labels=[f"PC {component}" for component in components],
    )
    overlap_axis.set_title("Absolute template overlap")
    if overlap_matrix.size <= 100:
        for row in range(overlap_matrix.shape[0]):
            for column in range(overlap_matrix.shape[1]):
                value = overlap_matrix[row, column]
                color = "white" if value < 0.45 else "black"
                overlap_axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )
    fig.colorbar(image, ax=overlap_axis, label="Absolute cosine overlap")

    output = Path(output)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output
