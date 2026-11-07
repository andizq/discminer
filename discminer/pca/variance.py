"""Variance spectra and exploratory, rank-dependent PCA tail diagnostics."""

import numpy as np
from astropy.table import Table


MAD_SCALE = 1.482602218505602


def _eigenvalues(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be a nonempty finite one-dimensional array")
    tolerance = 1e-12 * np.max(np.abs(values))
    if np.any(values < -tolerance):
        raise ValueError("eigenvalues contain materially negative values")
    values = np.maximum(values, 0.0)
    if np.sum(values) <= 0:
        raise ValueError("eigenvalues must have positive total power")
    if np.any(np.diff(values) > tolerance):
        raise ValueError("eigenvalues must be in descending PCA order")
    return values


def fit_variance_tail(
    eigenvalues, start=10, stop=None, model="log-linear", mad_multiplier=3.0
):
    """Fit a suffix in PCA rank, returning empirical curves and diagnostics.

    ``stop`` is inclusive. Only ranks before ``start`` are eligible for excess
    flags. MAD bounds describe the fitted tail; they are not confidence limits
    or calibrated noise thresholds. No independently measured noise is used.
    """
    values = _eigenvalues(eigenvalues)
    if model not in {"constant", "log-linear"}:
        raise ValueError("tail model must be constant or log-linear")
    if not np.isfinite(mad_multiplier) or mad_multiplier < 0:
        raise ValueError("tail MAD multiplier must be finite and non-negative")
    stop = values.size - 1 if stop is None else stop
    if not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)):
        raise ValueError("tail start and stop must be integer component indices")
    if start < 0 or stop >= values.size or stop - start < 2:
        raise ValueError(
            f"tail range {start}:{stop} must contain at least three components "
            f"within 0:{values.size - 1} (inclusive)"
        )
    ranks = np.arange(values.size)
    tail = values[start:stop + 1]
    slope = intercept = log_mad = np.nan
    median = float(np.median(tail))
    mad = float(MAD_SCALE * np.median(np.abs(tail - median)))
    if median <= 0:
        raise ValueError("selected tail has zero median; choose a positive tail range")
    if model == "constant":
        baseline = np.full(values.size, median)
        lower = np.full(values.size, max(0.0, median - mad_multiplier * mad))
        upper = np.full(values.size, median + mad_multiplier * mad)
    else:
        from scipy.stats import theilslopes

        if np.any(tail <= 0):
            raise ValueError(
                "log-linear tail requires positive eigenvalues throughout the "
                "fit range; use --tail-stop to exclude zero eigenvalues"
            )
        x, y = ranks[start:stop + 1], np.log(tail)
        slope, intercept, _, _ = theilslopes(y, x)
        intercept += np.median(y - (intercept + slope * x))
        residual = y - (intercept + slope * x)
        log_mad = float(MAD_SCALE * np.median(np.abs(residual)))
        with np.errstate(over="ignore", under="ignore"):
            baseline = np.exp(intercept + slope * ranks)
            lower = np.exp(intercept + slope * ranks - mad_multiplier * log_mad)
            upper = np.exp(intercept + slope * ranks + mad_multiplier * log_mad)
        if not np.all(np.isfinite(upper)) or np.any(lower <= 0):
            raise ValueError("tail extrapolation overflows; choose a broader fit range")
    eligible = ranks < start
    return dict(
        model=model, start=int(start), stop=int(stop), n_tail=int(tail.size),
        median=median, mad=mad, slope=float(slope), intercept=float(intercept),
        log_mad=log_mad, first_to_last=float(tail[0] / tail[-1]) if tail[-1] else np.inf,
        baseline=baseline, lower=lower, upper=upper, eligible=eligible,
        above=(values > upper * (1 + 1e-12)) & eligible,
    )


def variance_diagnostics(
    results, labels=None, tail_fit="none", tail_start=10, tail_starts=None,
    tail_stop=None, tail_mad=3.0,
):
    """Return full-spectrum component and tail-fit tables for PCA artifacts.

    Display limits never affect normalization or fitting. All fractions use the
    complete stored spectrum. Several tail starts assess analyst-choice
    sensitivity, not detection probability. A component in a fitted tail is
    excluded from that fit's robustness count.
    """
    results = list(results)
    if not results:
        raise ValueError("at least one PCA result is required")
    labels = [str(i) for i in range(len(results))] if labels is None else list(map(str, labels))
    if len(labels) != len(results) or len(set(labels)) != len(labels):
        raise ValueError("provide one unique label per PCA artifact")
    if tail_fit not in {"none", "constant", "log-linear"}:
        raise ValueError("tail_fit must be none, constant or log-linear")
    if tail_fit == "none" and (
        tail_starts is not None or tail_stop is not None or tail_start != 10 or tail_mad != 3.0
    ):
        raise ValueError("tail settings require --tail-fit constant or log-linear")
    starts = list(dict.fromkeys([tail_start] + list(tail_starts or [])))
    rows, fit_rows = [], []
    for result, label in zip(results, labels):
        values = _eigenvalues(result.eigenvalues)
        n = values.size
        ranks = np.arange(n)
        variance = values / values.sum()
        cumulative = np.cumsum(variance)
        normalized = np.full(n, np.nan)
        if np.sum(values[1:]) > 0:
            normalized[1:] = values[1:] / np.sum(values[1:])
        cumulative_without_pc0 = np.r_[np.nan, np.cumsum(variance[1:])]
        cumulative_normalized = np.r_[np.nan, np.cumsum(normalized[1:])]
        fits = [fit_variance_tail(values, s, tail_stop, tail_fit, tail_mad)
                for s in starts] if tail_fit != "none" else []
        nan = np.full(n, np.nan)
        primary = fits[0] if fits else dict(
            baseline=nan, lower=nan, upper=nan,
            eligible=np.zeros(n, dtype=bool), above=np.zeros(n, dtype=bool),
        )
        eligible = np.sum([f["eligible"] for f in fits], axis=0) if fits else np.zeros(n, int)
        above = np.sum([f["above"] for f in fits], axis=0) if fits else np.zeros(n, int)
        support = np.divide(above, eligible, out=nan.copy(), where=eligible > 0)
        ratio = np.divide(values, primary["upper"], out=nan.copy(), where=primary["upper"] > 0)
        for f in fits:
            fit_rows.append((label, f["model"], f["start"], f["stop"], f["n_tail"],
                f["median"], f["mad"], f["slope"], f["intercept"], f["log_mad"],
                f["first_to_last"], float(tail_mad),
                ",".join(map(str, ranks[f["above"]]))))
        for k in ranks:
            rows.append((label, int(k), values[k], variance[k], 100*variance[k],
                cumulative[k], cumulative_without_pc0[k], normalized[k],
                cumulative_normalized[k], primary["baseline"][k], primary["lower"][k],
                primary["upper"][k], ratio[k], bool(primary["eligible"][k]),
                bool(primary["above"][k]), int(eligible[k]), int(above[k]), support[k],
                bool(eligible[k] >= 3 and above[k] == eligible[k])))
    components = Table(rows=rows, names=(
        "label", "component", "eigenvalue", "variance_fraction", "variance_percent",
        "cumulative_variance", "cumulative_variance_no_pc0",
        "variance_fraction_no_pc0_renormalized", "cumulative_variance_no_pc0_renormalized",
        "tail_baseline", "tail_lower", "tail_upper", "observed_to_tail_upper",
        "primary_tail_eligible", "primary_above_tail", "eligible_tail_count",
        "above_tail_count", "tail_support_fraction", "above_all_eligible_tails",
    ))
    fit_table = Table(rows=fit_rows, names=(
        "label", "tail_model", "tail_start", "tail_stop", "n_tail",
        "tail_median", "tail_mad_scale", "log_slope_per_pc", "log_intercept",
        "log_residual_mad_scale", "tail_first_to_last", "mad_multiplier", "above_components",
    ), dtype=None if fit_rows else (
        "U1", "U16", int, int, int, float, float, float, float, float, float, float, "U1"))
    components.meta.update(
        sources=[result.source_path for result in results], labels=labels,
        mean_sub=[bool(result.mean_sub) for result in results],
        source_intensity_units=[getattr(result, "source_header", {}).get("BUNIT", "unspecified")
                                for result in results],
        eigenvalue_units="Square of each source intensity unit, with the original artifact normalization.",
        tail_fit=tail_fit, primary_tail_start=tail_start if fits else None,
        tail_starts=starts if fits else [], tail_stop=tail_stop, tail_mad=float(tail_mad),
        normalization="Full stored spectrum; PC 0-excluded normalization uses all PCs > 0.",
        caveat="Empirical tail envelopes are not confidence bands or noise detections. "
               "Support fractions measure tail-choice sensitivity, not probability. "
               "All-tail flag requires at least three eligible fits.",
    )
    fit_table.meta.update(components.meta)
    return components, fit_table


def fit_above_tail_components(components):
    """Describe the leading observed eigenvalues above the primary tail bound.

    Fit ln(eigenvalue) versus rank with Theil-Sen, using only pre-tail points
    flagged above the primary upper MAD envelope. PC 0 is always excluded.
    Two points determine a line but do not estimate residual scatter. This
    selection-dependent descriptive fit has no formal parameter uncertainties.
    """
    if components.meta.get("tail_fit", "none") == "none":
        raise ValueError("an above-tail component fit requires a fitted tail")
    from scipy.stats import theilslopes

    rows = []
    for label in dict.fromkeys(str(value) for value in components["label"]):
        subset = components[np.asarray(components["label"] == label)]
        subset = subset[np.argsort(subset["component"])]
        ranks = np.asarray(subset["component"], dtype=int)
        values = np.asarray(subset["eigenvalue"], dtype=float)
        selected = (
            np.asarray(subset["primary_tail_eligible"], dtype=bool)
            & np.asarray(subset["primary_above_tail"], dtype=bool)
            & (ranks > 0) & np.isfinite(values) & (values > 0)
        )
        x, y = ranks[selected], np.log(values[selected])
        slope = intercept = scatter = np.nan
        status = "insufficient_components"
        if len(x) >= 2:
            slope, intercept, _, _ = theilslopes(y, x)
            intercept += np.median(y - (intercept + slope*x))
            if len(x) >= 3:
                residual = y - (intercept + slope*x)
                scatter = float(MAD_SCALE * np.median(np.abs(residual)))
            status = "fitted"
        rows.append((label, ",".join(map(str, x)), len(x), float(slope),
                     float(intercept), scatter, status))
    fits = Table(rows=rows, names=(
        "label", "components", "n_components", "log_slope_per_pc",
        "log_intercept", "log_residual_mad_scale", "status",
    ))
    fits.meta.update(components.meta)
    fits.meta.update(
        fit_equation="ln(eigenvalue) = log_intercept + log_slope_per_pc * component",
        fit_method="Theil-Sen slope, median-centered log intercept",
        selection="PC > 0, before the primary tail, above its upper MAD envelope",
        baseline_subtracted=False,
        caveat="Descriptive fit to selected observed eigenvalues, not noise-subtracted signal. "
               "No formal significance or coefficient uncertainties. Two points provide no scatter estimate.",
    )
    return fits
