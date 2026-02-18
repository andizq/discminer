from discminer.mining_control import _mining_mirror_spectra
from discminer.tools.fit_kernel import _gauss
from discminer.grid import GridTools
from discminer.plottools import (MEDIUM_SIZE,
                                 make_round_map,
                                 make_up_ax,
                                 mod_major_ticks,
                                 use_discminer_style)

from discminer.mining_utils import (get_2d_plot_decorators,
                                    get_noise_mask,
                                    load_moments,
                                    load_disc_grid,
                                    init_data_and_model,
                                    make_1d_legend,
                                    show_output)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy import units as u
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import json
import warnings

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_mirror_spectra(None)
    args = parser.parse_args()

integrate = 0

#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

vsys = best['velocity']['vsys']
Rout = best['intensity']['Rout']
incl = best['orientation']['incl']
PA = best['orientation']['PA']
xc = best['orientation']['xc']
yc = best['orientation']['yc']
z_pars = best['height_upper']

gaps = custom['gaps']
rings = custom['rings']

try:
    kinks = custom['kinks']
except KeyError:
    kinks = []
    
ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, unit_simple=True, fmt_vertical=True)

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.1*Rout*u.au #Max model radius, 10% larger than disc Rout

with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.15*np.min([xmax, Rmax.value])
extent= np.array([-xmax, xmax, -xmax, xmax])

#********************
#LOAD DATA AND GRID
#********************
datacube, model = init_data_and_model()
vchannels = datacube.vchannels
model.make_model()

z_func = model.z_upper_func
vel2d, int2d, linew2d, lineb2d = model.props

R, phi, z = load_disc_grid()
noise_mean, mask = get_noise_mask(datacube, thres=2,
                                  mask_phi={'map2d': np.degrees(phi['upper']),
                                            'lims': args.mask_phi},
                                  mask_R={'map2d': R['upper']/au_to_m,
                                          'lims': args.mask_R}
)

Xproj = R[args.surface]*np.cos(phi[args.surface])
Yproj = R[args.surface]*np.sin(phi[args.surface])

X = Xproj/au_to_m
Y = Yproj/au_to_m

#**************************
#LOAD AND CLIP MOMENT MAPS
#**************************
moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask,
    clip_Rmin=0.0*u.au,
    clip_Rmax=args.Router*Rout*u.au,
    clip_Rgrid=R[args.surface]*u.m
)
    
datacube.convert_to_tb(writefits=False)

#********************************
#FOLD SPECTRA TO GET CPD SIGNAL?
#********************************
R_au = R[args.surface]/au_to_m
R_au_flat = R_au.flatten()
phi_rad = phi[args.surface]
indices = np.arange(len(R_au_flat))

def _wrap_dphi(dphi):
    """Wrap angle differences to [-pi, pi] elementwise."""
    return np.arctan2(np.sin(dphi), np.cos(dphi))

def get_pixel_from_Rphi(
    Rp_au,
    phip_rad,
    tol=datacube.beam_size.to('au').value/4,
    require_within_tol=True,
):
    """
    Infer (i,j) pixel indices corresponding to disc-plane (Rp, phip),
    using the same selection strategy as get_mirror_pixel below:
      - arc-length constraint: Rp * |dphi| < tol
      - radial constraint: |R - Rp| < tol
    Then pick the best candidate by minimising hypot(dR, Rp*|dphi|).
    """
    #Flattened diffs (with 2pi wrapping)
    dphi_flat = np.abs(_wrap_dphi(phi_rad - phip_rad)).flatten()
    dR_flat = np.abs(R_au_flat - Rp_au)

    #Candidate selection
    pind = (Rp_au * dphi_flat) < tol
    rind = dR_flat[pind] < tol

    if np.any(pind) and np.any(rind):
        #Evaluate best among candidates
        Rcand = R_au_flat[pind][rind]
        dphicand = dphi_flat[pind][rind]
        totdiff = np.hypot(Rcand - Rp_au, Rp_au * dphicand)
        indbest = indices[pind][rind][np.argmin(totdiff)]
        return np.unravel_index(indbest, R_au.shape)

    #Global nearest in the same metric
    totdiff_all = np.hypot(R_au_flat - Rp_au, Rp_au * dphi_flat)
    indbest = indices[np.argmin(totdiff_all)]
    ij = np.unravel_index(indbest, R_au.shape)

    if require_within_tol:
        #Warning if diff is beyond tolerance.
        i, j = ij
        if (np.abs(R_au[i, j] - Rp_au) > tol) or (Rp_au * np.abs(_wrap_dphi(phi_rad[i, j] - phip_rad)) > tol):
            warnings.warn(
                f"Requested (Rp,phip)=({Rp_au:.2f} au, {np.degrees(phip_rad):.2f} deg) "
                f"has no candidates within tol={tol:.2f} au; using global nearest pixel {ij}."
            )

    return ij

def build_ij_window(ipix, jpix, dpix, shape):
    """Return a clipped list of (i,j) indices for a (2*dpix+1)^2 window."""
    ny, nx = shape
    i0 = max(ipix - dpix, 0)
    i1 = min(ipix + dpix, ny - 1)
    j0 = max(jpix - dpix, 0)
    j1 = min(jpix + dpix, nx - 1)

    ilist, jlist = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    return list(zip(ilist.ravel(), jlist.ravel()))

def cli_coords_to_disc_Rphi(args, dpc, z_func, z_pars, incl, PA, xc=0.0, yc=0.0):
    """
    Returns (Rp_au, phip_rad) always in disc frame.

    - If args.input_coords == 'disc':
        rp   : au
        phip : deg (disc azimuth)
    - If args.input_coords == 'sky':
        rp   : arcsec
        phip : deg (PA from North)

    If multiple rp/phip values are passed, only the first one
    is used and a warning is printed.
    """

    rp_list = args.rp
    phip_list = args.phip

    if not isinstance(rp_list, (list, tuple, np.ndarray)):
        rp_list = [rp_list]

    if not isinstance(phip_list, (list, tuple, np.ndarray)):
        phip_list = [phip_list]

    # Print warning if more than one entry
    if len(rp_list) > 1 or len(phip_list) > 1:
        warnings.warn(
            "Multiple -rp/-phip values received. "
            "Only the first entry will be used as the reference pixel."
        )

    rp0 = float(rp_list[0])
    phip0 = float(phip_list[0])

    if args.input_coords == "disc":
        Rp_au = rp0
        phip_rad = np.deg2rad(phip0)
        return Rp_au, phip_rad

    if args.input_coords == "sky":
        rsky_au = rp0 * (dpc.to("pc").value) #arcsec to au
        PAsky = np.deg2rad(phip0)  #PA measured from North

        #Sky plane Cartesian (North up, East left)
        xsky = -rsky_au * np.sin(PAsky)
        ysky =  rsky_au * np.cos(PAsky)

        #Transform to disc frame
        xdisc, ydisc = GridTools.get_disc_from_sky_coords(
            xsky, ysky, z_func, z_pars, incl, PA, xc=xc, yc=yc
        )

        Rp_au = np.hypot(xdisc, ydisc)
        phip_rad = np.arctan2(ydisc, xdisc)

        return Rp_au, phip_rad

    raise ValueError(
        f"Unknown --input_coords '{args.input_coords}' "
        "(expected 'disc', 'disk', or 'sky')."
    )

def get_mirror_pixel(i, j, surface=args.surface, tol=datacube.beam_size.to('au').value/4):

    Rij = R_au[i,j]
    phiij = phi_rad[i,j]

    if phiij>0:
        diff = 0.5*np.pi - phiij
        phidiff = phi_rad - (0.5*np.pi + diff) #phi2d minus perfect mirror values, to be minimised along with dR
    else: 
        diff = -0.5*np.pi - phiij
        phidiff = phi_rad - (-0.5*np.pi + diff)        
        
    phidiff = np.abs(phidiff).flatten()

    pind = Rij * phidiff < tol #arc length
    rind = np.abs(Rij - R_au_flat[pind]) < tol #dR

    Rmir = R_au_flat[pind][rind]
    phimir_diff = phidiff[pind][rind]
    totdiff = np.hypot(Rmir-Rij, Rmir*phimir_diff)

    indbest = indices[pind][rind][np.argmin(totdiff)]

    return np.unravel_index(indbest, R_au.shape)

def get_mirror_channel_index(vchannels, channel_id, vsys):
    """
    Return the index of the channel mirrored around vsys.
    """
    v0 = vchannels[channel_id]
    v_mirror = 2.0 * vsys - v0

    idx_mirror = np.argmin(np.abs(vchannels - v_mirror))
    return idx_mirror

def get_spectra(cube, ij, ijmirr, vsys=vsys):

    i, j = ij
    imirr, jmirr = ijmirr

    spec = cube.data[:, i, j]
    vel_spec = vchannels - vsys
    
    spec_mirr = cube.data[:, imirr, jmirr]
    vel_mirr = vsys - vchannels

    return spec, vel_spec, spec_mirr, vel_mirr

def median_and_uncertainty(values):
    peak = np.max(values)    
    median = np.median(values)
    lower = np.percentile(values, 16)
    upper = np.percentile(values, 84)
    return peak, median, lower, upper

def fold_and_plot_spectra(cube, ijlist, vel_interp=np.linspace(-5.0, 5.0, 101), ntop=10, vsys=vsys, integrate=integrate):
    
    residuals_spec = []
    spec_arr = []
    mirr_arr = []
    
    for ij in ijlist:

        print ('Pixel, mirror:', ij, get_mirror_pixel(*ij))
        spec, vel_spec, spec_mirr, vel_mirr = get_spectra(cube, ij, get_mirror_pixel(*ij), vsys=vsys)

        if np.max(vel_interp)>np.max(vel_spec): #Fix if spectrum velocity falls out of interpolation range
            vel_interp = np.linspace(-np.max(vel_spec), np.max(vel_spec), 101)

        peakind = np.argmax(spec)
        f_spec = interpolate.interp1d(vel_spec, spec, kind='cubic')
        f_mirr = interpolate.interp1d(vel_mirr, spec_mirr, kind='cubic')

        spec_arr.append(spec)
        mirr_arr.append(spec_mirr)
        
        residuals_spec.append(f_spec(vel_interp) - f_mirr(vel_interp))

    residuals_spec = np.asarray(residuals_spec)

    #Initial guess for gaussian fits
    p0 = [0.5*np.nanmax(residuals_spec), vchannels[args.channel_id]-vsys, 0.3, 0.0] #Amplitude, centroid, sigma, offset    
    
    if integrate:
        spec_mean = np.sum(spec_arr, axis=0)
        mirr_mean = np.sum(mirr_arr, axis=0)
        #vel_spec is the same for all spectra (spec) and equally spaced
        fs = interpolate.interp1d(vel_spec, spec_mean, kind='cubic') 
        fm = interpolate.interp1d(vel_mirr, mirr_mean, kind='cubic')         
        #residuals_mean = fs(vel_interp) - fm(vel_interp)
        
    #**********************
    #PLOT RESIDUAL SPECTRA
    #**********************     
    fig, ax = plt.subplots(figsize=(14,5))

    if args.units=='Jy':
        residuals_spec*=1000
        ax.set_ylabel('Folded Residuals [mJy/beam]', fontsize=MEDIUM_SIZE+2)
    else:
        ax.set_ylabel('Folded Residuals [K]', fontsize=MEDIUM_SIZE+2)
        
    ax.set_xlabel(r'Velocity relative to CSD $\upsilon_{\rm sys}$ [km/s]', fontsize=MEDIUM_SIZE+2)
    
    for resspec in residuals_spec:
        ax.plot(vel_interp, resspec, color='k', lw=0.6, alpha=0.1)
                    
    max_curve = np.max(residuals_spec, axis=0)
    min_curve = np.min(residuals_spec, axis=0)
    
    ymax = savgol_filter(max_curve, 10, 2)      
    ymin = savgol_filter(min_curve, 10, 2)

    res_mean = np.mean(residuals_spec)
    res_std = np.std(residuals_spec)
    ax.axhline(res_mean, ls='--', lw=1.2, color='k', label='Mean (%.1f)'%res_mean)
    ax.axhline(res_mean-res_std, ls='--', lw=0.6, color='k', label=r'1$\sigma$ (%.1f)'%res_std)
    ax.axhline(res_mean+res_std, ls='--', lw=0.6, color='k')
    
    make_up_ax(ax, labelbottom=True, labeltop=False, labelsize=MEDIUM_SIZE+1, xlims=(-4.5, 4.5), ylims=(-np.max(ymax)*0.6, 1.4*np.max(ymax)))
    leg = make_1d_legend(ax, fontsize=MEDIUM_SIZE+2, handlelength=1.5, loc='lower left', bbox_to_anchor=(0.0, 1.0),
                         #frameon=True, fancybox=True, shadow=False, framealpha=1, facecolor='w', edgecolor='0.8', borderpad=1.0
    )
    #leg.get_frame().set_linewidth(3.0)
    mod_major_ticks(ax, axis='x', nbins=10)

    ax.yaxis.get_major_locator().set_params(integer=True)

    #**************************
    #GAUSSIAN FIT TO RESIDUALS
    #**************************
    amplitudes = []
    centers = []
    sigmas = []
    offsets = []
    Rlist = []
    philist = []
    rskylist = []
    PAlist = []
    
    for i, profile in enumerate(residuals_spec):
        try:
            popt, _ = curve_fit(_gauss, vel_interp, profile, p0=p0)
            A_fit, mu_fit, sigma_fit, off_fit = popt
            amplitudes.append(A_fit)
            centers.append(mu_fit)
            sigmas.append(sigma_fit)
            offsets.append(off_fit)
            ip, jp = ijlist[i]
            Rtmp = np.hypot(X[ip,jp], Y[ip,jp]) #This is correct when model parfile z=0
            phitmp = np.arctan2(Y[ip,jp], X[ip,jp])
            ztmp = model.z_upper_func({'R': Rtmp*au_to_m}, **z_pars)/au_to_m
            Rlist.append(Rtmp)
            philist.append(np.degrees(phitmp))
            xs, ys, zs = GridTools.get_sky_from_disc_coords(Rtmp, phitmp, z=ztmp, incl=incl, PA=PA, xc=xc, yc=yc)
            rskylist.append(np.hypot(xs, ys)/dpc.value) #arcsecs
            PAlist.append(270 + np.degrees(np.arctan2(ys, xs)))
            
        except RuntimeError:
            # Skip fits that fail
            continue

    amplitudes = np.array(amplitudes)
    centers = np.array(centers)
    sigmas = np.array(sigmas)
    offsets = np.array(offsets)    
    Rlist = np.array(Rlist)
    philist = np.array(philist)
    rskylist = np.array(rskylist)
    PAlist = np.array(PAlist)
    
    top_idx = np.argsort(amplitudes)[-ntop:][::-1]
    print ('R[au], phi[deg] of peak spectrum', Rlist[top_idx[0]], philist[top_idx[0]])
    print ('R[au], phi[deg] median of top %d spectra'%ntop, np.median(Rlist[top_idx]), np.median(philist[top_idx]))    
    print ('R[au], phi[deg] stddev of top %d spectra'%ntop, np.std(Rlist[top_idx]), np.std(philist[top_idx]))
    print ('Rsky[''], PA[deg] median of top %d spectra'%ntop, np.median(rskylist[top_idx]), np.median(PAlist[top_idx]))    
    print ('Rsky[''], PA[deg] stddev of top %d spectra'%ntop, np.std(rskylist[top_idx]), np.std(PAlist[top_idx])) 
    
    _, A_median, A_16, A_84 = median_and_uncertainty(amplitudes[top_idx])
    _, mu_median, mu_16, mu_84 = median_and_uncertainty(centers[top_idx])
    _, sigma_median, sigma_16, sigma_84 = median_and_uncertainty(sigmas[top_idx])
    _, off_median, off_16, off_84 = median_and_uncertainty(offsets[top_idx])    

    A_peak, mu_peak, sigma_peak, off_peak = amplitudes[top_idx[0]], centers[top_idx[0]], sigmas[top_idx[0]], offsets[top_idx[0]]
    
    print("Median Amplitude: {:.2f} (+{:.2f}/-{:.2f})".format(A_median, A_84 - A_median, A_median - A_16))
    print("Median Centroid:    {:.2f} (+{:.2f}/-{:.2f})".format(mu_median, mu_84 - mu_median, mu_median - mu_16))
    print("Median Sigma:     {:.2f} (+{:.2f}/-{:.2f})".format(sigma_median, sigma_84 - sigma_median, sigma_median - sigma_16))

    gaussian_median = _gauss(vel_interp, A_median, mu_median, sigma_median, off_median)
    vel_fit = np.linspace(vel_interp[0], vel_interp[-1], 400)

    for idx in top_idx:
        A, mu, sigma, offset = amplitudes[idx], centers[idx], sigmas[idx], offsets[idx]
        g = _gauss(vel_fit, A, mu, sigma, offset)
        ax.plot(vel_fit, g, lw=4, color='tomato', alpha=0.15, zorder=0)        
            
    print ('Peak Gaussian fit properties (A, center, sigma [FWHM], offset):', A_peak, mu_peak, sigma_peak, [2.35482*sigma_peak], off_peak)
    
    return fig, ax, residuals_spec

#*****************************************
#GET REFERENCE PIXEL, WINDOW, and MIRRORS
#*****************************************
Rp_au, phip_rad = cli_coords_to_disc_Rphi(args, dpc, z_func, z_pars, incl, PA, xc=xc, yc=yc)

ipix, jpix = get_pixel_from_Rphi(Rp_au, phip_rad) #Reference pixel
ijlist = build_ij_window(ipix, jpix, args.dpix, R_au.shape)
imir, jmir = get_mirror_pixel(ipix, jpix) #Mirror pixel

print(f"CLI coords: input-->{args.input_coords} rp={args.rp[0]} phip={args.phip[0]}")
print(f"Disc-frame target: R={Rp_au:.2f} au, phi={np.degrees(phip_rad):.2f} deg")
print(f"Reference pixel: (i,j)=({ipix},{jpix})  ->  R={R_au[ipix,jpix]:.2f} au, phi={np.degrees(phi_rad[ipix,jpix]):.2f} deg")
print(f"Mirror pixel: (i,j)=({imir},{jmir})  ->  R={R_au[imir,jmir]:.2f} au, phi={np.degrees(phi_rad[imir,jmir]):.2f} deg")

#***********************************
#SPECTRUM FROM REF PIXEL AND MIRROR
#***********************************
vchannels_fold = vchannels - vsys
        
mpl.rcParams['hatch.linewidth'] = 0.1

fig, ax = plt.subplots(figsize=(14,5))
data = datacube.data

if integrate:    
    stat_func = np.mean
    npix = args.dpix

    data_slice = data[:, ipix-npix:ipix+npix+1, jpix-npix:jpix+npix+1]
    data_spec = np.array([stat_func(chan) for chan in data_slice])

    mirr_slice = data[:, imir-npix:imir+npix+1, jmir-npix:jmir+npix+1]
    mirr_spec = np.array([stat_func(chan) for chan in mirr_slice])

    ax.step(vchannels_fold, data_spec, where='mid', lw=3.3, color='tomato')
    ax.fill_between(vchannels_fold, data_spec, step='mid', facecolor='tomato', alpha=0.1)
    
    ax.step(vchannels_fold, mirr_spec, where='mid', lw=3.3, color='dodgerblue')
    ax.fill_between(vchannels_fold, mirr_spec, step='mid', facecolor='dodgerblue', alpha=0.1)
    
    ax.step(-vchannels_fold, mirr_spec, where='mid', lw=2.2, color='k')
    ax.fill_between(-vchannels_fold, mirr_spec, step='mid', facecolor='0.5', alpha=0.6, zorder=20)
    ax.fill_between(-vchannels_fold, mirr_spec, step='mid', facecolor='none', edgecolor='k', lw=0, hatch='X', zorder=21)

else:
    ax.step(vchannels_fold, data[:, ipix, jpix], where='mid', lw=3.3, color='tomato', label='Reference')
    ax.fill_between(vchannels_fold, data[:, ipix, jpix], step='mid', facecolor='tomato', alpha=0.1)

    ax.step(vchannels_fold, data[:, imir, jmir], where='mid', lw=3.3, color='dodgerblue', label='Mirror')
    ax.fill_between(vchannels_fold, data[:, imir, jmir], step='mid', facecolor='dodgerblue', alpha=0.1)

    ax.step(-vchannels_fold, data[:, imir, jmir], where='mid', lw=2.2, color='k', label='Folded mirror')
    ax.fill_between(-vchannels_fold, data[:, imir, jmir], step='mid', facecolor='0.5', alpha=0.6, zorder=20)
    ax.fill_between(-vchannels_fold, data[:, imir, jmir], step='mid', facecolor='none', edgecolor='k', lw=0, hatch='X', zorder=21)

ax.axvline(vchannels[args.channel_id]-vsys, lw=2.0, dash_capstyle='round', dashes=(2.0, 2.5), color='magenta', label=r'$\upsilon_{\rm ref}$')
make_up_ax(ax, labelbottom=True, labeltop=False, labelsize=MEDIUM_SIZE+1, xlims=(-4.5, 4.5), ylims=(-2, 1.2*np.nanmax(data[:, ipix, jpix])))
make_1d_legend(ax, fontsize=MEDIUM_SIZE+2, handlelength=1.5, ncols=4)
mod_major_ticks(ax, axis='x', nbins=10)

if args.units=='Jy':
    ax.set_ylabel('Intensity [mJy/beam]', fontsize=MEDIUM_SIZE+2)
else:
    ax.set_ylabel('Brightness temperature [K]', fontsize=MEDIUM_SIZE+2)
ax.set_xlabel(r'Velocity relative to CSD $\upsilon_{\rm sys}$ [km/s]', fontsize=MEDIUM_SIZE+2)

plt.savefig('spectra_reference_and_mirror.png', bbox_inches='tight', dpi=200)
show_output(args)


#**************
#FOLD AND PLOT
#**************
fig, ax, residuals_spec = fold_and_plot_spectra(datacube, ijlist, vsys=vsys)

ax.axvline(vchannels[args.channel_id] - vsys, ls='--', lw=2.0, dash_capstyle='round', dashes=(2.0, 2.5), color='magenta')

plt.savefig('spectra_folded_residuals_%dpix.png'%(args.dpix), bbox_inches='tight', dpi=200)
show_output(args)


#*****************************
#REFERENCE CHANNEL AND PIXELS
#*****************************
def plot_projected_channel(chanid, tag=''):
    fig, ax = make_round_map(moment_data, levels_im, Xproj*u.m, Yproj*u.m, args.Router*Rout*u.au,
                         z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                             cmap=cmap_mom, clabel=unit, fmt=cfmt, 
                             gaps=gaps, rings=rings, kinks=kinks,
                             make_contourf=False,
                             make_contour=False,
                             make_radial_grid=False,
                             make_azimuthal_grid=True,                             
                             make_cbar=False,
                             #ticks_phi=[],                             
                             #mask_inner=args.Rinner*datacube.beam_size
    )
    
    cf_map = ax.contourf(X, Y, datacube.data[chanid], levels=levels_im, cmap=cmap_mom)
        
    for ij in ijlist:
        i,j = ij
        ax.scatter(X[i,j], Y[i,j], marker='o', fc='none', ec='tomato', lw=1.5, s=100)

    ax.scatter(Rp_au*np.cos(phip_rad), Rp_au*np.sin(phip_rad), marker='X', fc='none', ec='k', lw=2.5, s=350)
    ax.scatter(X[imir,jmir], Y[imir,jmir], marker='P', fc='none', ec='k', lw=2.5, s=350)

    ax.set_title('Projected channel map (%.1f km/s)'%vchannels[chanid], fontsize=MEDIUM_SIZE+2)
    ax.set_xlim(-Rout, Rout)
    ax.set_ylim(-Rout, Rout)

    plt.savefig('spectra_mappixels_%dpix_%s.png'%(args.dpix, tag), bbox_inches='tight', dpi=200)    
    show_output(args)

plot_projected_channel(args.channel_id, tag='reference')
plot_projected_channel(get_mirror_channel_index(vchannels, args.channel_id, vsys), tag='mirror')

