"""
Auto-build prepare_data.py and preview suggested spatial and spectral clipping.
Also, provide rough estimates for systemic velocity and orientation parameters.

Conventions:
- data.shape = (nchan, nrow, ncol) = (spectral, y, x)
- North  = increasing rows  (+y index)
- East   = decreasing cols  (-x index)
- PA (E of N) = atan2(E, N) = atan2(-dx, dy) for a direction (dx,dy) in ARRAY basis.
- Rough disc inclination i ≈ arccos(b/a).
"""
from discminer.mining_control import _mining_prepare
from discminer.plottools import use_discminer_style, get_discminer_cmap, make_1d_legend
from discminer.tools.utils import FrontendUtils

import os
import argparse
import termtables as tt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrow

from astropy import units as u
from astropy.stats import SigmaClip
from spectral_cube import SpectralCube

from scipy.ndimage import (
    gaussian_filter,
    binary_opening, binary_closing, generate_binary_structure,
    label,
    distance_transform_edt,
)

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_prepare(None)
    args = parser.parse_args()

_break_line = FrontendUtils._break_line

def _mad(a):
    med = np.nanmedian(a)
    return 1.4826 * np.nanmedian(np.abs(a - med))

def _round_up_to_multiple(x, m):
    return int(np.ceil(float(x) / float(m)) * m) if x > 0 else m

def _round_down_to_multiple(x, m):
    return int(np.floor(max(0.0, float(x)) / float(m)) * m)

# ---------- Robust peak mask & core ----------
def _emission_mask_from_map_robust(img, k_sigma=5.0, min_pixels=50,
                                   smooth_sigma=1.0, open_iter=1, close_iter=1):
    """
    Robust mask from a peak map:
      1) light Gaussian smoothing,
      2) sigma-clipped background stats,
      3) threshold at med + k_sigma*std,
      4) keep largest connected component,
      5) small morphological open/close,
      6) enforce minimum area.
    """
    sm = gaussian_filter(img.astype(float), sigma=float(smooth_sigma)) if smooth_sigma and smooth_sigma>0 else img

    sigclip = SigmaClip(sigma=3.0, maxiters=5)
    clipped = sigclip(sm)
    bkg_med  = np.nanmedian(clipped)
    bkg_std  = np.nanstd(clipped)
    thr = bkg_med + k_sigma * bkg_std

    mask = (sm > thr) & np.isfinite(sm)

    if mask.any():
        lab, nlab = label(mask)
        if nlab > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0  # background
            keep = sizes.argmax()
            mask = (lab == keep)

    if open_iter or close_iter:
        st = generate_binary_structure(2, 2)  # 8-connected
        for _ in range(int(open_iter)):
            mask = binary_opening(mask, st)
        for _ in range(int(close_iter)):
            mask = binary_closing(mask, st)

    if mask.sum() < int(min_pixels):
        mask[:] = False

    return mask

def _core_mask(mask, core_frac=0.90):
    """
    Keep the most interior fraction of mask pixels using Euclidean distance transform.
    Stabilises centroid/orientation by focusing on the coherent core.
    """
    if not mask.any():
        return mask
    edt = distance_transform_edt(mask)
    vals = edt[mask]
    thr = np.percentile(vals, (1.0 - float(core_frac)) * 100.0)
    core = np.zeros_like(mask, dtype=bool)
    core[mask] = vals >= thr
    # Safety: if core got too small, fall back to original mask
    return core if core.sum() >= max(10, 0.1 * mask.sum()) else mask

# ---------- PCA on core; winsorised extents on full mask ----------
def _centroid_weighted(img, mask):
    y, x = np.indices(img.shape)
    w = np.where(mask, img, 0.0)
    W = w.sum()
    if W <= 0 or not np.isfinite(W):
        return img.shape[0] / 2.0, img.shape[1] / 2.0
    yc = float((w * y).sum() / W)
    xc = float((w * x).sum() / W)
    return yc, xc

def _principal_axes_x_y(img, yc, xc):
    """
    Weighted covariance eigenvectors in ARRAY order (x=cols, y=rows).
    Returns eigvals (desc) and eigvecs as columns: v_major = eigvecs[:,0] = [vx, vy].
    """
    y, x = np.indices(img.shape)
    w = np.where(np.isfinite(img), img, 0.0)
    dx = (x - xc).ravel()
    dy = (y - yc).ravel()
    ww = w.ravel()
    m = ww > 0
    if m.sum() < 3:
        return np.array([1.0, 0.5]), np.eye(2)
    X = np.vstack([dx[m], dy[m]])  # (2, N) array basis
    wn = ww[m] / ww[m].sum()
    C = (X * wn) @ X.T
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]

def _robust_axis_extents_winsor(mask, yc, xc, eigvecs_x_y,
                                extent_quantile=97.5, winsor_top_pct=2.0):
    """
    Quantile-based semi-axes along principal axes, with upper-tail winsorisation.
    Reduces sensitivity to scattered outliers that survive masking.
    """
    yy, xx = np.where(mask)
    if yy.size == 0:
        return 0.0, 0.0
    dy = yy - yc
    dx = xx - xc
    vmaj = eigvecs_x_y[:, 0]
    vmin = eigvecs_x_y[:, 1]
    proj_maj = np.abs(dx * vmaj[0] + dy * vmaj[1])
    proj_min = np.abs(dx * vmin[0] + dy * vmin[1])

    def _winsorize_top(a, pct):
        if pct <= 0:
            return a
        hi = np.percentile(a, 100.0 - pct)
        return np.minimum(a, hi)

    proj_maj = _winsorize_top(proj_maj, winsor_top_pct)
    proj_min = _winsorize_top(proj_min, winsor_top_pct)

    q = float(extent_quantile)
    Rmaj_raw = float(np.percentile(proj_maj, q))
    Rmin_raw = float(np.percentile(proj_min, q))
    return Rmaj_raw, Rmin_raw

# ---------- Spectral window detection ----------
def _find_spectral_window_in_box(data, y0, y1, x0, x1, snr_min=3.0, smooth=5, empty_pad=10):
    """
    Decide [kmin,kmax] using ONLY the given spatial box, across ALL channels.
    ch_sig: 99th-percentile; ch_rms: MAD; optional boxcar smoothing.
    """
    nchan = data.shape[0]
    ch_sig = np.empty(nchan, dtype=float)
    ch_rms = np.empty(nchan, dtype=float)
    for k in range(nchan):
        sl = data[k, y0:y1, x0:x1]
        ch_sig[k] = np.nanpercentile(sl, 99.0)
        ch_rms[k] = _mad(sl) or np.nanstd(sl)
    if smooth > 1:
        ker = np.ones(smooth) / float(smooth)
        ch_sig = np.convolve(ch_sig, ker, mode="same")
        ch_rms = np.convolve(ch_rms, ker, mode="same")
    is_emiss = ch_sig > (snr_min * ch_rms)
    if np.any(is_emiss):
        idx = np.where(is_emiss)[0]
        kmin_tight, kmax_tight = int(idx[0]), int(idx[-1])
    else:
        kmin_tight, kmax_tight = nchan // 4, 3 * nchan // 4
    kmin = max(0, kmin_tight - empty_pad)
    kmax = min(nchan - 1, kmax_tight + empty_pad)
    return kmin, kmax, (ch_sig, ch_rms)

# ---------- Spectral axis, vsys, and red side ----------
def _safe_velocity_axis(sc):
    """
    Return spectral axis in km/s (radio). Else return native + a kind tag.
    """
    try:
        sc_vel = sc.with_spectral_unit(u.km/u.s, velocity_convention='radio')
        v = sc_vel.spectral_axis.to_value(u.km/u.s)
        return v, 'velocity'
    except Exception:
        pass
    try:
        ax = sc.spectral_axis
        unit = ax.unit
        pt = getattr(unit, "physical_type", "")
        vals = ax.to_value(unit)
        if pt == 'frequency':
            return vals, 'frequency'
        if pt == 'length':
            return vals, 'wavelength'
        if pt in ('speed', 'velocity'):
            return vals, 'velocity'
        return np.arange(ax.size, dtype=float), 'index'
    except Exception:
        return np.arange(sc.shape[0], dtype=float), 'index'

def _centroid_coord(weights, coord):
    w = np.clip(weights, a_min=0, a_max=None).astype(float)
    if not np.isfinite(w).any() or np.sum(w) <= 0:
        return np.nan
    return float(np.sum(coord * w) / np.sum(w))

def _estimate_vsys_from_clipped(sc, data, y0, y1, x0, x1, kmin, kmax):
    coord, kind = _safe_velocity_axis(sc)
    sub = data[kmin:kmax+1, y0:y1, x0:x1]
    spec = np.nanmean(sub, axis=(1, 2))
    coord_sub = coord[kmin:kmax+1]
    vsys = _centroid_coord(spec, coord_sub)
    return vsys, coord, kind

def _red_mask(coord, kind, vsys):
    if kind == 'velocity':
        return coord > vsys
    if kind == 'frequency':
        return coord < vsys
    if kind == 'wavelength':
        return coord > vsys
    return coord > vsys  # index fallback

def _estimate_red_side_pa_from_redmask(data, yc, xc, vmaj, redmask, r_test, box_hw=3):
    """
    Decide which end of 'vmaj' is redshifted using only red channels.
    Basis:
      vmaj = [vx, vy] in ARRAY basis (x=cols, y=rows)
      N = +vy, E = -vx, hence PA = atan2(E, N) = atan2(-vx, vy)
    """
    nchan, ny, nx = data.shape
    xA = xc + r_test * vmaj[0]; yA = yc + r_test * vmaj[1]
    xB = xc - r_test * vmaj[0]; yB = yc - r_test * vmaj[1]

    def box_sum_red(xc_pix, yc_pix):
        x0 = int(np.clip(np.floor(xc_pix - box_hw), 0, nx-1))
        x1 = int(np.clip(np.ceil (xc_pix + box_hw),  0, nx))
        y0 = int(np.clip(np.floor(yc_pix - box_hw), 0, ny-1))
        y1 = int(np.clip(np.ceil (yc_pix + box_hw),  0, ny))
        cube = data[:, y0:y1, x0:x1]
        if redmask is None or not np.any(redmask):
            return np.nan
        return float(np.nansum(cube[redmask]))

    IA = box_sum_red(xA, yA)
    IB = box_sum_red(xB, yB)
    if not np.isfinite(IA) and not np.isfinite(IB):
        return None

    red_is_plus = (IA >= IB)
    v_red = vmaj if red_is_plus else -vmaj
    N = v_red[1]; E = -v_red[0]
    pa_red_deg = (np.degrees(np.arctan2(E, N)) % 360.0)
    return pa_red_deg

# ---------------- Core logic ----------------
def suggest_clipping_with_diag(
    fits_path,
    dpc=100,
    clip_factor=1.5,
    spatial_multiple=30,
    k_sigma_peak=5.0,
    min_pixels=50,
    extent_quantile=97.5,
    winsor_top_pct=2.0,
    core_frac=0.90,
    spectral_box_frac=0.25,
    snr_min_spec=3.0,
    smooth_spec=5,
    empty_pad=10,
):
    """
    Spatial from PEAK map using robust mask + core-first PCA + winsorised extents.
    Spectral from a central box that is a fraction of the crop width.
    Estimate inclination and PA (red semimajor axis).
    """

    sc = SpectralCube.read(fits_path)    
    #sc.allow_huge_operations = True
    data = sc.filled_data[:].value if hasattr(sc.filled_data[:], "value") else np.asarray(sc.filled_data[:])
    nchan, ny, nx = np.shape(data)

    # ---- PEAK map across ALL channels ----
    peak = np.nanmax(data, axis=0)
    
    # ---- Robust mask & core-first geometry ----
    mask = _emission_mask_from_map_robust(
        peak, k_sigma=k_sigma_peak, min_pixels=min_pixels,
        smooth_sigma=1.0, open_iter=1, close_iter=1
    )
    mask_core = _core_mask(mask, core_frac=core_frac)

    # Centroid and axes from the CORE for stability
    yc, xc = _centroid_weighted(peak, mask_core)
    eigvals, eigvecs = _principal_axes_x_y(peak * mask_core, yc, xc)

    # Semi-axes from FULL mask but winsorized for robustness
    Rmaj_raw, Rmin_raw = _robust_axis_extents_winsor(
        mask, yc, xc, eigvecs,
        extent_quantile=extent_quantile, winsor_top_pct=winsor_top_pct
    )

    # Scale to requested clip factor
    Rmaj = clip_factor * Rmaj_raw
    Rmin = clip_factor * Rmin_raw
    Rmax_used = max(Rmaj_raw, Rmin_raw)

    # npix from robust size, rounded to multiple and capped by bounds
    desired_half = clip_factor * Rmax_used
    allowed_half = float(min(yc, ny - 1 - yc, xc, nx - 1 - xc))
    desired_up   = _round_up_to_multiple(desired_half, spatial_multiple)
    allowed_down = _round_down_to_multiple(allowed_half, spatial_multiple)
    if allowed_down > 0:
        npix = min(desired_up, allowed_down)
    else:
        npix = int(max(1, np.floor(allowed_half)))

    # Crop box corners (array indices)
    y0 = int(np.floor(yc - npix)); y1 = int(np.ceil(yc + npix))
    x0 = int(np.floor(xc - npix)); x1 = int(np.ceil(xc + npix))

    # ---- Spectral: central box as a fraction of the clipped window ----
    box_w = max(4, int(np.round((2 * npix) * float(spectral_box_frac))))
    half = max(2, box_w // 2)
    sy0 = int(max(0, np.floor(yc - half))); sy1 = int(min(ny, np.ceil(yc + half)))
    sx0 = int(max(0, np.floor(xc - half))); sx1 = int(min(nx, np.ceil(xc + half)))

    kmin, kmax, (ch_sig, ch_rms) = _find_spectral_window_in_box(
        data, sy0, sy1, sx0, sx1, snr_min=snr_min_spec, smooth=smooth_spec, empty_pad=empty_pad
    )

    # ---- vsys from spectrum in the clipped spatial window ----
    vsys, coord_full, coord_kind = _estimate_vsys_from_clipped(sc, data, y0, y1, x0, x1, kmin, kmax)
    if not np.isfinite(vsys):
        vsys = 0.5 * (kmin + kmax) if coord_kind == 'index' else 0.5 * (coord_full[kmin] + coord_full[kmax])
    redmask = _red_mask(coord_full, coord_kind, vsys)

    # ---- Inclination and PA_red ----
    ba = float(Rmin_raw / Rmaj_raw) if Rmaj_raw > 0 else np.nan
    ba = float(np.clip(ba, 0.0, 1.0))
    inc_deg = float(np.degrees(np.arccos(ba))) if np.isfinite(ba) else np.nan

    r_test = 0.6 * min(npix, Rmaj)
    vmaj = eigvecs[:, 0]
    pa_red_deg = _estimate_red_side_pa_from_redmask(data, yc, xc, vmaj, redmask, r_test, box_hw=3)

    diag = {
        # Spatial
        "peak": peak, "yc": yc, "xc": xc, "vmaj": vmaj, "vmin": eigvecs[:,1],
        "Rmaj_raw": float(Rmaj_raw), "Rmin_raw": float(Rmin_raw),
        "Rmaj": float(Rmaj), "Rmin": float(Rmin),
        "npix_half": int(npix), "box": (y0, y1, x0, x1),
        "mask": mask, "mask_core": mask_core,
        # Spectral
        "spectral_interval": (kmin, kmax),
        "spectral_box_coords": (sy0, sy1, sx0, sx1),
        "vsys": vsys, "coord_kind": coord_kind,
        # Derived orientation
        "inc_deg": inc_deg,
        "pa_red_deg": pa_red_deg,
        # Meta
        "map_label": "Peak intensity",
    }
    return int(npix), int(kmin), int(kmax), diag


def _divisors(n: int):
    n = int(abs(n))
    if n == 0:
        return []
    small, large = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]  # ascending order


def pick_downsample_factor(npix: int, min_side=50, max_side=100, choice="largest"):
    """
    Pick integer ds | npix so that side_after = (2*npix)/ds is in [min_side, max_side].
    choice: 'largest' (default) or 'smallest' among the valid ds.
    If no ds yields a side in-range, fall back to the divisor giving side
    closest to the midpoint, tie-breaking toward larger ds.
    Returns (ds, side_after).
    """
    npix = int(npix)
    full_side = 2 * npix
    divs = [d for d in _divisors(npix) if d > 0]

    valid = []
    for ds in divs:
        side = full_side / ds
        if side.is_integer() and (min_side <= side <= max_side):
            valid.append((ds, int(side)))

    if valid:
        if choice == "smallest":
            ds, side = min(valid, key=lambda t: t[0])
        else:  # "largest"
            ds, side = max(valid, key=lambda t: t[0])
        return ds, side

    # Fallback: closest to midpoint, tie-break toward larger ds
    target = 0.5 * (min_side + max_side)
    best = None
    best_key = None
    for ds in divs:
        side = full_side / ds
        if not side.is_integer():
            continue
        side = int(side)
        key = (abs(side - target), -ds)  # prefer larger ds on ties
        if (best is None) or (key < best_key):
            best = (ds, side)
            best_key = key
    return best if best is not None else (1, full_side)

def write_prepare_data_py_autods(
    outfile, file_data, dpc_pc, npix, kmin, kmax,
    min_side=50, max_side=100, ds_choice="largest"
):
    """
    Writes prepare_data.py per your template:
      - Always writes the clipping step with npix and [kmin,kmax]
      - Writes analysis downsample(2) only if 2*npix >= 250
      - Picks MCMC downsample factor automatically (ds | npix and final side in [50,100])
        using ds_choice: 'largest' or 'smallest'
    """
    ds_mcmc, side_after = pick_downsample_factor(npix, min_side=min_side, max_side=max_side, choice=ds_choice)
    tag_mcmc = f"_{ds_mcmc}pix"

    include_analysis_ds2 = (2 * int(npix) >= 250)
    tag_analysis = "_2pix"
    base = file_data.split(".fits")[0]

    lines = []
    lines.append("#Command-line generated: discminer prepare")
    lines.append("from discminer.core import Data")
    lines.append("from astropy import units as u")
    lines.append("")
    lines.append(f"file_data = '{base}'")
    lines.append(f"dpc = {dpc_pc}*u.pc")
    lines.append("")
    lines.append("#**********************")
    lines.append("#DATACUBE FOR ANALYSIS")
    lines.append("#**********************")
    lines.append("datacube = Data(file_data+'.fits', dpc)")
    lines.append(f"datacube.clip(npix={int(npix)}, channels={{\"interval\": [{int(kmin)}, {int(kmax)}]}}, overwrite=True)")
    if include_analysis_ds2:
        lines.append(f"datacube.downsample(2, tag='{tag_analysis}')")
    lines.append("")
    lines.append("#**********************")
    lines.append("#DATACUBE FOR MCMC FIT")
    lines.append("#**********************")
    lines.append("datacube = Data(file_data+'_clipped.fits', dpc)")
    lines.append(f"datacube.downsample({ds_mcmc}, tag='{tag_mcmc}')")
    lines.append("")

    content = "\n".join(lines)
    with open(outfile, "w") as f:
        f.write(content)
    return outfile, ds_mcmc, side_after


# ---------------- Plotting ----------------
def _plot_preview(diag, outpng, outshow, title=None):
    peak = diag["peak"]
    yc, xc = diag["yc"], diag["xc"]
    vsys = diag["vsys"]
    vmaj = diag["vmaj"]; vmin = diag["vmin"]
    y0, y1, x0, x1 = diag["box"]
    Rmaj, Rmin = diag["Rmaj"], diag["Rmin"]
    npix_half = diag["npix_half"]
    sy0, sy1, sx0, sx1 = diag["spectral_box_coords"]
    inc_deg = diag["inc_deg"]
    pa_red_deg = diag["pa_red_deg"]

    fig, ax = plt.subplots(figsize=(8.8, 7.8))
    im = ax.imshow(peak, origin="lower", interpolation="nearest", cmap=get_discminer_cmap("peakintensity"))
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(diag.get("map_label", "Peak (arb.)"))

    # Centroid
    ax.plot(xc, yc, marker="x", ms=8, mew=2)

    # Axes (ARRAY basis)
    ax.plot([xc - Rmaj * vmaj[0], xc + Rmaj * vmaj[0]],
            [yc - Rmaj * vmaj[1], yc + Rmaj * vmaj[1]], lw=2, ls='-')
    ax.plot([xc - Rmin * vmin[0], xc + Rmin * vmin[0]],
            [yc - Rmin * vmin[1], yc + Rmin * vmin[1]], lw=2, ls='--')

    # Ellipse overlay: angle from +x(cols) toward +y(rows)
    theta_deg = float(np.degrees(np.arctan2(vmaj[1], vmaj[0])))
    ell = Ellipse((xc, yc), width=2*Rmaj, height=2*Rmin, angle=theta_deg,
                  fill=False, linewidth=2)
    ax.add_patch(ell)

    # Clipping box
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color='red', lw=2, label='Suggested Field of View size')

    # Spectral box (dotted)
    ax.plot([sx0, sx1, sx1, sx0, sx0], [sy0, sy0, sy1, sy1, sy0], color='red',
            lw=2.0, dash_capstyle='round', dashes=(0.5, 1.5), alpha=0.95, label='Box for spectral clipping estimate')

    # Redshifted major-axis arrow (if available)
    if pa_red_deg is not None:
        # Convert PA to ARRAY deltas: rows += cos(PA); cols += -sin(PA)
        theta = np.radians(pa_red_deg)
        N = np.cos(theta); E = np.sin(theta)
        L = 0.8 * Rmaj
        dx = -E * L
        dy = +N * L
        ax.add_patch(FancyArrow(xc, yc, dx, dy, width=0.0,
                                length_includes_head=True, head_width=6, head_length=8, zorder=50))

    ax.set_xlim(0, peak.shape[1]); ax.set_ylim(0, peak.shape[0]); ax.set_aspect("equal")
    if title:
        pa_txt = f"{pa_red_deg:.1f}°" if pa_red_deg is not None else "n/a"
        ax.set_title(title + f" | inc≈{inc_deg:.1f}° | PA_red≈{pa_txt} | vsys={vsys:.1f}km/s |", fontsize=9)
    ax.set_xlabel("x [pix] (→ West, ← East)"); ax.set_ylabel("y [pix] (→ North)")

    make_1d_legend(ax, ncol=1, fontsize=10, loc='upper left', bbox_to_anchor=(0.03,0.97))
    
    plt.savefig(outpng, dpi=200, bbox_inches='tight')
    if outshow:
        plt.show()
    plt.close()

# ---------------- RUN ----------------
def main():    
    fits_path = args.file_data
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    npix, kmin, kmax, diag = suggest_clipping_with_diag(
        fits_path,
        dpc=args.dpc,
        clip_factor=args.clip_factor,
        spatial_multiple=args.spatial_multiple,
        k_sigma_peak=args.k_sigma_peak,
        min_pixels=args.min_pixels,
        extent_quantile=args.extent_quantile,
        winsor_top_pct=args.winsor_top_pct,
        core_frac=args.core_frac,
        spectral_box_frac=args.spectral_box_frac,
        snr_min_spec=args.snr_min_spec,
        smooth_spec=args.smooth_spec,
        empty_pad=args.empty_pad,
    )

    out_py, ds_mcmc, side_after = write_prepare_data_py_autods(
        args.prepare_data, args.file_data, args.dpc, npix, kmin, kmax,
        min_side=50, max_side=100, ds_choice=args.ds_choice
    )

    _break_line()

    title = f"| npix={npix} | ch=[{kmin},{kmax}]"
    _plot_preview(diag, args.preview, args.show_output, title=title)
    print(f"Saved preview: {args.preview}")
    
    print(f"Wrote {out_py} with:")
    choice_str = "aggressive binning (faster)" if args.ds_choice=='largest' else "minimal binning (more detail preserved)"
    print(f"  npix = {npix}  → MCMC downsampling={ds_mcmc} → side {side_after} px (choice: {args.ds_choice} → {choice_str})")
    print('''  and channels={"interval": [%d, %d]}'''%(kmin, kmax))
        
    # --- Compute diagnostics ---
    inc = diag['inc_deg']
    pa_red = diag['pa_red_deg']
    vsys = diag['vsys']
    
    pa_str = f"{pa_red:.1f}° E of N (red)" if pa_red is not None else "n/a"
    incsign_str = "POSITIVE" if args.incl_sign > 0 else "NEGATIVE"
    velsign_str = "POSITIVE" if args.vel_sign > 0 else "NEGATIVE"

    if args.vel_sign * args.incl_sign < 0:
        dPA = 90
        diff_str = "-"
    else:
        dPA = -90
        diff_str = "+"
        
    pa_dm = np.radians(pa_red - dPA)
    inc_rad = np.radians(args.incl_sign * inc)

    # --- table 1: observational diagnostics ---
    table1 = [
        ["Window FOV half-size (pix)", f"{npix}"],
        ["Spectral clipping (Channel ids)", f"[{kmin}, {kmax}]"],
        ["Downsampled size for fit (pix)", ds_mcmc],
        ["Inclination (deg)", f"{args.incl_sign * inc:.1f}"],        
        ["PA_redshifted_axis (deg)", f"{pa_red:.1f}" if pa_red is not None else "n/a"],
    ]

    tt.print(
        table1,
        header=["Derived quantity", "Value"],
        style=tt.styles.rounded,
        alignment="rr"
    )

    print(f"(mask pixels: {int(diag['mask'].sum())}, core pixels: {int(diag['mask_core'].sum())})")        

    print(f"If these settings look good, run: python {args.prepare_data}\n"
          "You can also adjust values manually in that script before running it.\n")
    
    # --- table 2: suggested initial guesses ---

    print("Rough orientation parameters and systemic velocity,\n"
          "converted to discminer convention...")
    
    table2 = [
        ["Inclination (rad)", f"{inc_rad:.1f}"],
        ["PA_discminer (rad)", f"{pa_dm:.1f}"],
        ["vsys (km/s)", f"{vsys:.1f}"],
        #["vel_sign", f"{args.vel_sign} (fixed user input)"],        
    ]

    tt.print(
        table2,
        header=["Parameter", "Suggested guess for fit"],
        style=tt.styles.rounded,
        alignment="rr"
    )

    print(f"⚠️  Using {incsign_str} inclination and {velsign_str} rotation for PA correction.")
    print("   You can adjust via --incl_sign and --vel_sign.")
    print(f"   (PA_discminer computed as {pa_red:.1f} {diff_str} {abs(dPA)}°)")
    _break_line()
    
main()
