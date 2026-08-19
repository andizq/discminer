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
        raise ValueError(
            "central_fraction must be greater than 0 and at most 1"
        )

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
    """Plot selected zero-based PCA eigenimages and eigenspectra."""

    use_discminer_style()
    components = result._validate_component_indices(components, "components")
    if cmap == "discminer":
        cmap = get_discminer_cmap("velocity")

    ncols = len(components)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(3.5 * ncols, 6.0),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"height_ratios": (1.0, 0.55)},
    )
    image_axes = axes[0]
    spectrum_axes = axes[1]
    velocity_order = np.argsort(result.velocity)
    velocity = result.velocity[velocity_order]

    if share_scale:
        common_limits = _symmetric_limits(
            result.eigenimages[components],
            robust=robust,
            percentile=percentile,
        )

    for column, component in enumerate(components):
        image_axis = image_axes[column]
        spectrum_axis = spectrum_axes[column]
        image = result.eigenimages[component]
        limits = (
            common_limits
            if share_scale
            else _symmetric_limits(
                image, robust=robust, percentile=percentile
            )
        )
        plotted = image_axis.imshow(
            image,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
        )
        image_axis.set_title(f"PC {component}")
        image_axis.set_xticks([])
        image_axis.set_yticks([])
        fig.colorbar(
            plotted,
            ax=image_axis,
            fraction=0.046,
            pad=0.04,
        )

        eigenvector = result.eigenvectors[velocity_order, component]
        spectrum_axis.axhline(0.0, color="0.65", linewidth=0.8)
        spectrum_axis.plot(velocity, eigenvector, color="dodgerblue")
        spectrum_axis.set_xlabel(r"Velocity [km s$^{-1}$]")
        spectrum_axis.margins(x=0.02)
        if column == 0:
            spectrum_axis.set_ylabel("Eigenvector")

    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return Path(output)


def _diagnostic_component_count(result, n_components):
    if not 1 <= n_components <= 9:
        raise ValueError("n_components must be between 1 and 9")
    return min(n_components, result.selected_components)


def _normalized_autocorrelation(values):
    values = np.asarray(values, dtype=float).copy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    peak = np.nanmax(finite)
    if peak != 0:
        values /= peak
    return values


def _lag_axis(size):
    return np.fft.fftshift(np.fft.fftfreq(size) * size)


def _fit_autocorrelation_ellipse(image, xlag, ylag):
    from matplotlib.path import Path as MatplotlibPath
    from skimage.measure import EllipseModel, find_contours

    contours = find_contours(image, np.exp(-1))
    for contour in contours:
        xvalues = np.interp(
            contour[:, 1],
            np.arange(image.shape[1]),
            xlag,
        )
        yvalues = np.interp(
            contour[:, 0],
            np.arange(image.shape[0]),
            ylag,
        )
        points = np.column_stack((xvalues, yvalues))
        if not MatplotlibPath(points).contains_point((0.0, 0.0)):
            continue
        ellipse = EllipseModel()
        if ellipse.estimate(points):
            return ellipse.predict_xy(np.linspace(0.0, 2.0 * np.pi, 200))
    return None


def plot_spatial_width_diagnostics(
    result: PCAResult,
    output,
    n_components=9,
    dpi=200,
    show=False,
):
    """Plot the spatial autocorrelation width diagnostics."""

    use_discminer_style()
    count = _diagnostic_component_count(
        result,
        n_components,
    )
    autocorrelations = result.spatial_autocorrelation
    ylag = _lag_axis(autocorrelations.shape[1])
    xlag = _lag_axis(autocorrelations.shape[2])
    xedge0, xedge1 = _velocity_edges(xlag)
    yedge0, yedge1 = _velocity_edges(ylag)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(9, 9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for component, axis in enumerate(axes.ravel()):
        if component >= count:
            axis.axis("off")
            continue

        image = _normalized_autocorrelation(
            autocorrelations[component]
        )
        axis.imshow(
            image,
            origin="lower",
            interpolation="nearest",
            cmap="afmhot",
            extent=[xedge0, xedge1, yedge0, yedge1],
            aspect="equal",
        )
        finite = image[np.isfinite(image)]
        if (
            finite.size
            and np.nanmin(finite) <= np.exp(-1) <= np.nanmax(finite)
        ):
            axis.contour(
                xlag,
                ylag,
                image,
                levels=[np.exp(-1)],
                colors="cyan",
                linewidths=1.5,
            )
            if result.spatial_method == "contour":
                ellipse = _fit_autocorrelation_ellipse(
                    image,
                    xlag,
                    ylag,
                )
                if ellipse is not None:
                    axis.plot(
                        ellipse[:, 0],
                        ellipse[:, 1],
                        color="limegreen",
                        linestyle="--",
                        linewidth=1.5,
                    )
        axis.set_title(f"PC {component}")

    fig.text(0.5, 0.01, "Spatial lag x [pixels]", ha="center")
    fig.text(
        0.01,
        0.5,
        "Spatial lag y [pixels]",
        ha="center",
        va="center",
        rotation="vertical",
    )
    fig.suptitle(
        rf"Eigenimage spatial ACF: $1/e={np.exp(-1):.3f}$",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return Path(output)


def _spectral_channel_width(velocity):
    differences = np.abs(np.diff(np.asarray(velocity, dtype=float)))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if differences.size == 0:
        raise ValueError(
            "At least two distinct velocity channels are required"
        )
    return np.median(differences)


def plot_spectral_width_diagnostics(
    result: PCAResult,
    output,
    n_components=9,
    max_lag=None,
    dpi=200,
    show=False,
):
    """Plot spectral autocorrelations and their fitted widths."""

    use_discminer_style()
    count = _diagnostic_component_count(
        result,
        n_components,
    )
    autocorrelations = result.spectral_autocorrelation
    lag = np.fft.rfftfreq(result.n_channels) * result.n_channels
    channel_width = _spectral_channel_width(result.velocity)
    fitted_widths = result.spectral_width / channel_width

    if max_lag is None:
        plot_limit = lag[-1]
    else:
        plot_limit = float(max_lag)
        if not np.isfinite(plot_limit) or plot_limit <= 0:
            raise ValueError("max_lag must be finite and positive")
        plot_limit = min(plot_limit, lag[-1])

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(9, 9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for component, axis in enumerate(axes.ravel()):
        if component >= count:
            axis.axis("off")
            continue

        values = _normalized_autocorrelation(
            autocorrelations[:lag.size, component]
        )
        axis.plot(lag, values, color="dodgerblue")
        axis.axhline(
            np.exp(-1),
            color="red",
            linestyle="--",
            label=r"$\exp(-1)$",
        )
        axis.axhline(0.0, color="black", linestyle=":")
        if np.isfinite(fitted_widths[component]):
            axis.axvline(
                fitted_widths[component],
                color="limegreen",
                linestyle="-.",
                label="Fitted width",
            )
        axis.set_title(f"PC {component}")
        axis.set_xlim(0.0, plot_limit)
        if component == 0:
            axis.legend(frameon=False)

    fig.text(0.5, 0.01, "Spectral lag [channels]", ha="center")
    fig.text(
        0.01,
        0.5,
        "Normalized autocorrelation",
        ha="center",
        va="center",
        rotation="vertical",
    )
    fig.suptitle("Eigenvector spectral ACF", y=0.995)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return Path(output)


def weighted_log_width_fit(spatial, spectral, spectral_error):
    """Fit ``log10(spectral) = slope * log10(spatial) + intercept``.

    This reproduces the custom weighted least-squares calculation used by the
    original PCA width script. Only the spectral uncertainties set the
    weights; spatial uncertainties are retained for plotting.
    """

    spatial = np.asarray(spatial, dtype=float)
    spectral = np.asarray(spectral, dtype=float)
    spectral_error = np.asarray(spectral_error, dtype=float)
    if not (
        spatial.shape == spectral.shape == spectral_error.shape
        and spatial.ndim == 1
    ):
        raise ValueError("Fit inputs must be one-dimensional matching arrays")
    if spatial.size < 2:
        raise ValueError(
            "At least two valid components are required for a fit"
        )
    if (
        np.any(~np.isfinite(spatial))
        or np.any(~np.isfinite(spectral))
        or np.any(~np.isfinite(spectral_error))
        or np.any(spatial <= 0)
        or np.any(spectral <= 0)
        or np.any(spectral_error <= 0)
    ):
        raise ValueError("Fit inputs must be finite and strictly positive")

    log_spatial = np.log10(spatial)
    log_spectral = np.log10(spectral)
    log_spectral_error = spectral_error / (
        spectral * np.log(10.0)
    )

    design = np.column_stack((log_spatial, np.ones_like(log_spatial)))
    weights = 1.0 / log_spectral_error**2
    normal = design.T @ (weights[:, np.newaxis] * design)
    if np.linalg.matrix_rank(normal) < 2:
        raise ValueError("The selected spatial widths cannot define a fit")

    covariance = np.linalg.inv(normal)
    parameters = covariance @ (
        design.T @ (weights * log_spectral)
    )
    slope, intercept = parameters
    slope_error, intercept_error = np.sqrt(np.diag(covariance))
    amplitude = 10.0**intercept
    amplitude_error = amplitude * np.log(10.0) * intercept_error

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_error": slope_error,
        "intercept_error": intercept_error,
        "amplitude": amplitude,
        "amplitude_error": amplitude_error,
        "covariance": covariance,
    }


def _beam_size_au(result):
    """Return the major-axis FWHM in au using discminer's convention."""

    bmaj_degrees = result.source_header.get("BMAJ")
    if bmaj_degrees is None:
        return None
    bmaj_arcsec = float(bmaj_degrees) * 3600.0
    return bmaj_arcsec * result.distance_pc


def plot_widths(
    result: PCAResult,
    output,
    beam_multiple=1.0,
    n_fit_components=6,
    spectral_error_scale=0.2,
    dpi=200,
    show=False,
):
    """Plot and fit spectral widths against spatial widths."""

    use_discminer_style()
    if beam_multiple < 0:
        raise ValueError("beam_multiple must be non-negative")
    if n_fit_components < 2:
        raise ValueError("n_fit_components must be at least 2")
    if spectral_error_scale <= 0:
        raise ValueError("spectral_error_scale must be positive")

    spatial = result.spatial_width.copy()
    spatial_error = result.spatial_width_error.copy()
    spectral = result.spectral_width.copy()
    spectral_error = spectral_error_scale * result.spectral_width_error
    components = np.arange(result.n_components)
    xlabel = "Spatial width [au]"
    beam_scale = _beam_size_au(result)

    if result.outer_radius_au is not None:
        spatial *= 100.0 / result.outer_radius_au
        spatial_error *= 100.0 / result.outer_radius_au
        xlabel = r"Spatial width [% $R_{\rm out}$]"
        if beam_scale is not None:
            beam_scale *= 100.0 / result.outer_radius_au

    valid = (
        np.isfinite(spatial)
        & np.isfinite(spatial_error)
        & np.isfinite(spectral)
        & np.isfinite(spectral_error)
        & (spatial > 0)
        & (spatial_error > 0)
        & (spectral > 0)
        & (spectral_error > 0)
    )
    if not np.any(valid):
        raise ValueError("The artifact has no valid spatial-spectral widths")

    spatial = spatial[valid]
    spatial_error = spatial_error[valid]
    spectral = spectral[valid]
    spectral_error = spectral_error[valid]
    components = components[valid]

    fit_mask = components < n_fit_components
    if beam_scale is not None:
        fit_mask &= spatial > beam_multiple * beam_scale
    if np.count_nonzero(fit_mask) < 2:
        raise ValueError(
            "Fewer than two components satisfy the fit selection. Increase "
            "--n-fit-components or reduce --beam-multiple."
        )

    fit = weighted_log_width_fit(
        spatial[fit_mask],
        spectral[fit_mask],
        spectral_error[fit_mask],
    )

    beam_threshold = (
        None if beam_scale is None else beam_multiple * beam_scale
    )
    if beam_scale is not None:
        print(f"Beam size      = {_beam_size_au(result):.1f} au")
        if result.outer_radius_au is not None:
            print(
                f"Frac beam size = {beam_threshold:.1f} % Rout"
            )
    print(f"Number of fitted points = {np.count_nonzero(fit_mask)}")
    print("Best-fit relation in log10 space:")
    print(
        "log10(spectral) = "
        f"({fit['slope']:.4f} ± {fit['slope_error']:.4f}) "
        "log10(spatial) + "
        f"({fit['intercept']:.4f} ± {fit['intercept_error']:.4f})"
    )
    print("\nEquivalent linear-space relation:")
    print(
        "spectral = "
        f"({fit['amplitude']:.4e} ± {fit['amplitude_error']:.4e}) * "
        f"spatial^({fit['slope']:.4f} ± {fit['slope_error']:.4f})"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(
        spatial,
        spectral,
        xerr=spatial_error,
        yerr=spectral_error,
        ecolor="k",
        fmt="o",
        ms=6,
        alpha=0.3,
        markeredgecolor="k",
        markerfacecolor="w",
        capsize=2,
        label="All points",
    )
    ax.errorbar(
        spatial[fit_mask],
        spectral[fit_mask],
        xerr=spatial_error[fit_mask],
        yerr=spectral_error[fit_mask],
        fmt="o",
        ms=9,
        capsize=2,
        markeredgewidth=1.5,
        markeredgecolor="k",
        markerfacecolor="tomato",
        label="Fit sample",
    )
    for xvalue, yvalue, component in zip(
        spatial[fit_mask],
        spectral[fit_mask],
        components[fit_mask],
    ):
        ax.text(
            xvalue + 0.3,
            yvalue,
            str(component),
            ha="left",
            va="top",
            fontsize=15,
        )

    xline = np.linspace(np.min(spatial), np.max(spatial), 500)
    yline = fit["amplitude"] * xline ** fit["slope"]
    ax.plot(
        xline,
        yline,
        lw=3,
        color="tomato",
        alpha=0.3,
        label=(
            rf"Fit: $y = {fit['amplitude']:.2e}\,"
            rf"x^{{{fit['slope']:.2f}}}$"
        ),
    )
    if beam_threshold is not None:
        beam_unit = (
            r"% $R_{\rm out}$"
            if result.outer_radius_au is not None
            else "au"
        )
        ax.axvline(
            beam_threshold,
            ls="--",
            lw=2.0,
            color="magenta",
            label=f"Beam = {beam_threshold:.1f} {beam_unit}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Spectral width [km s$^{-1}$]")
    ax.set_title("PCA spatial and spectral widths")
    ax.set_ylim(
        0.2 * np.min(spectral[fit_mask]),
        1.5 * np.max(spectral[fit_mask]),
    )
    ax.legend(frameon=False)
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
