"""Plotting utilities for PCA artifacts and reconstructed cubes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from discminer.plottools import get_discminer_cmap, use_discminer_style

from .artifact import PCAResult


def covariance_velocity_window(
    velocity,
    central_fraction=0.4,
    velocity_limit=None,
):
    """Return a central covariance window in physical velocity units."""

    velocity = np.asarray(velocity, dtype=float)
    finite = velocity[np.isfinite(velocity)]
    if finite.size < 2:
        raise ValueError("At least two finite velocity channels are required")

    vmin = np.min(finite)
    vmax = np.max(finite)
    if not 0 < central_fraction <= 1:
        raise ValueError("central_fraction must be greater than 0 and at most 1")

    if vmin <= 0 <= vmax:
        center = finite[np.argmin(np.abs(finite))]
    else:
        center = 0.5 * (vmin + vmax)

    if velocity_limit is None:
        half_width = 0.5 * central_fraction * (vmax - vmin)
    else:
        half_width = float(velocity_limit)
        if not np.isfinite(half_width) or half_width <= 0:
            raise ValueError("velocity_limit must be finite and positive")

    lower = max(vmin, center - half_width)
    upper = min(vmax, center + half_width)
    return lower, upper


def _velocity_edges(velocity):
    velocity = np.asarray(velocity, dtype=float)
    if velocity.size < 2:
        return velocity[0] - 0.5, velocity[0] + 0.5
    first = velocity[0] - 0.5 * (velocity[1] - velocity[0])
    last = velocity[-1] + 0.5 * (velocity[-1] - velocity[-2])
    return first, last


def plot_covariance(
    result: PCAResult,
    output,
    central_fraction=0.4,
    velocity_limit=None,
    dpi=200,
    show=False,
):
    """Plot the covariance matrix on the input cube's velocity axis."""

    use_discminer_style()
    order = np.argsort(result.velocity)
    velocity = result.velocity[order]
    covariance = result.covariance[np.ix_(order, order)]
    lower, upper = covariance_velocity_window(
        velocity,
        central_fraction=central_fraction,
        velocity_limit=velocity_limit,
    )
    edge0, edge1 = _velocity_edges(velocity)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(
        covariance,
        origin="lower",
        interpolation="nearest",
        cmap="cmr.ember",
        extent=[edge0, edge1, edge0, edge1],
        aspect="equal",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"Velocity [km s$^{-1}$]")
    ax.set_ylabel(r"Velocity [km s$^{-1}$]")
    ax.set_title("PCA covariance")
    fig.colorbar(image, ax=ax, label="Covariance")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return Path(output)


def _symmetric_limits(image, robust=False, percentile=99.5):
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return -1.0, 1.0
    if robust:
        maximum = np.nanpercentile(np.abs(finite), percentile)
    else:
        maximum = np.nanmax(np.abs(finite))
    if maximum == 0 or not np.isfinite(maximum):
        maximum = 1.0
    return -maximum, maximum


def plot_components(
    result: PCAResult,
    components,
    output,
    cmap="RdBu_r",
    robust=False,
    percentile=99.5,
    share_scale=False,
    dpi=200,
    show=False,
):
    """Plot selected zero-based PCA component images."""

    use_discminer_style()
    components = result._validate_component_indices(components, "components")
    if cmap == "discminer":
        cmap = get_discminer_cmap("velocity")

    ncols = len(components)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(3.5 * ncols, 3.5),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    if share_scale:
        common_limits = _symmetric_limits(
            result.eigenimages[components],
            robust=robust,
            percentile=percentile,
        )

    for axis, component in zip(axes, components):
        image = result.eigenimages[component]
        limits = (
            common_limits
            if share_scale
            else _symmetric_limits(
                image, robust=robust, percentile=percentile
            )
        )
        plotted = axis.imshow(
            image,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
        )
        axis.set_title(f"PC {component}")
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)

    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return Path(output)


def plot_widths(
    result: PCAResult,
    output,
    dpi=200,
    show=False,
):
    """Plot valid spectral widths against spatial widths."""

    use_discminer_style()
    spatial = result.spatial_width.copy()
    spatial_error = result.spatial_width_error.copy()
    xlabel = "Spatial width [au]"

    if result.outer_radius_au is not None:
        spatial *= 100.0 / result.outer_radius_au
        spatial_error *= 100.0 / result.outer_radius_au
        xlabel = r"Spatial width [% $R_{\rm out}$]"

    valid = (
        np.isfinite(spatial)
        & np.isfinite(spatial_error)
        & np.isfinite(result.spectral_width)
        & np.isfinite(result.spectral_width_error)
        & (spatial > 0)
        & (result.spectral_width > 0)
    )
    if not np.any(valid):
        raise ValueError("The artifact has no valid spatial-spectral widths")

    components = np.arange(result.n_components)[valid]
    spectral = result.spectral_width[valid]
    spectral_error = result.spectral_width_error[valid]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        spatial[valid],
        spectral,
        xerr=spatial_error[valid],
        yerr=spectral_error,
        fmt="o",
        markeredgecolor="k",
        markerfacecolor="tomato",
        capsize=2,
    )
    for xvalue, yvalue, component in zip(
        spatial[valid], spectral, components
    ):
        ax.annotate(
            str(component),
            (xvalue, yvalue),
            xytext=(4, -4),
            textcoords="offset points",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Spectral width [km s$^{-1}$]")
    ax.set_title("PCA spatial and spectral widths")
    ax.grid(alpha=0.3)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return Path(output)


def plot_channels(
    cube_path,
    distance,
    output,
    channel_ids=None,
    step=1,
    n_channels=8,
    systemic_velocity=0.0,
    cmap="inferno",
    vmin=None,
    vmax=None,
    xlim=None,
    dpi=200,
    show=False,
):
    """Plot up to eight channel maps using the current discminer cube API."""

    from discminer.core import Data

    use_discminer_style()
    cube = Data(str(cube_path), distance)
    n_channels = min(int(n_channels), 8)
    if n_channels < 1:
        raise ValueError("n_channels must be positive")

    if channel_ids is None:
        center = int(np.argmin(np.abs(cube.vchannels - systemic_velocity)))
        offsets = step * (
            np.arange(n_channels, dtype=int) - (n_channels - 1) // 2
        )
        channel_ids = center + offsets
    else:
        channel_ids = np.asarray(channel_ids, dtype=int)

    if len(channel_ids) > 8:
        raise ValueError("At most eight channel maps can be plotted")
    if np.any(channel_ids < 0) or np.any(channel_ids >= cube.nchan):
        raise IndexError("Some requested channel indices are out of bounds")

    pixel_arcsec = cube.pix_size.to_value(u.arcsec)
    xhalf = 0.5 * (cube.nx - 1) * pixel_arcsec
    yhalf = 0.5 * (cube.ny - 1) * pixel_arcsec
    extent = [-xhalf, xhalf, -yhalf, yhalf]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    images = []
    for index, axis in enumerate(axes.ravel()):
        if index >= len(channel_ids):
            axis.axis("off")
            continue
        channel = int(channel_ids[index])
        image = axis.imshow(
            cube.data[channel],
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        delta_velocity = cube.vchannels[channel] - systemic_velocity
        axis.text(
            0.04,
            0.92,
            rf"$\Delta v={delta_velocity:.2f}$ km s$^{{-1}}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="w",
            bbox=dict(boxstyle="round", fc="k", ec="none", alpha=0.35),
        )
        if xlim is not None:
            axis.set_xlim(-xlim, xlim)
            axis.set_ylim(-xlim, xlim)

    for axis in axes[1, :]:
        axis.set_xlabel("Offset [arcsec]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Offset [arcsec]")
    if images:
        fig.colorbar(
            images[-1],
            ax=axes.ravel().tolist(),
            fraction=0.025,
            pad=0.02,
            label="Intensity",
        )

    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return Path(output)
