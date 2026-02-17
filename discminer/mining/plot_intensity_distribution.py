from discminer.mining_control import _mining_intensdistrib
from discminer.core import Data

from discminer.mining_utils import (_get_mask_tuples,
                                    _merge_R_phi_mask,
                                    make_masks,
                                    get_noise_mask,
                                    load_disc_grid,
                                    get_2d_plot_decorators,
                                    init_data_and_model,
                                    load_moments,
                                    mark_planet_location,
                                    show_output)

from discminer.plottools import (make_up_ax,
                                 make_round_map,
                                 use_discminer_style)

import json
import random
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_intensdistrib(None)
    args = parser.parse_args()

#*******************
#FILL UP EMPTY ARGS
#*******************
nmasks = int(round(0.5*len(args.annuli)))

if len(args.r0)==nmasks:
    rcenters = args.r0    
else:
    nmissing = nmasks - len(args.r0)
    rcenters = np.append(args.r0, [0]*nmissing)
    
if len(args.phi0)==nmasks:
    phicenters = np.radians(args.phi0)
else:
    nmissing = nmasks - len(args.phi0)
    phicenters = np.radians(np.append(args.phi0, [0]*nmissing))

if len(args.colors)==nmasks:
    colors = args.colors
else:
    nmissing = nmasks - len(args.colors)
    colors = np.append(args.colors, ['black']*nmissing)

if len(args.linewidths)==nmasks:
    lws = args.linewidths
else:
    nmissing = nmasks - len(args.linewidths)
    lws = np.append(args.linewidths, [1.5]*nmissing)
    
#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

vel_sign = best['velocity']['vel_sign']
vsys = best['velocity']['vsys']
Rout = best['intensity']['Rout']
incl = best['orientation']['incl']
PA = best['orientation']['PA']
xc = best['orientation']['xc']
yc = best['orientation']['yc']

gaps = custom['gaps']
rings = custom['rings']
try:
    kinks = custom['kinks']
except KeyError:
    kinks = []

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment)

if args.clim!=0:
    clim = args.clim
    
#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc

R_lev = np.arange(25, 0.98*Rout, 50)*au_to_m

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model(Rmin=0, Rmax=1.6)
noise_mean, mask = get_noise_mask(datacube, thres=3)
modelcube = Data('cube_model_%s.fits'%tag, dpc)

datapdf = np.where(mask[None, :, :], np.nan, datacube.data)

#Definitions for plots
beam_au = datacube.beam_size.to('au').value
if args.absolute_Rinner>=0:
    Rmod_in = args.absolute_Rinner
else:
    Rmod_in = args.Rinner*beam_au

if args.absolute_Router>=0:
    Rmod_out = args.absolute_Router
else:
    Rmod_out = args.Router*Rout

xmax = model.skygrid['xmax'] 
xlim = 1.1*Rout
extent= np.array([-xmax, xmax, -xmax, xmax])/au_to_m
#plot_title = ctitle + r' $-$ ' + clabel.split('[')[0]

#****************************
#LOAD DISC GEOMETRY AND MASK
#****************************
R, phi, z = load_disc_grid()

R_s = R[args.surface]*u.m.to('au')
phi_s = np.degrees(phi[args.surface])

Xproj = R_s*np.cos(phi[args.surface])
Yproj = R_s*np.sin(phi[args.surface])

#*************************
#LOAD AND CLIP MOMENT MAPS
#Clip radially
# Masked sections are visually drawn at the end
moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask, #fill masked cells with nans
    clip_Rmin=0.0*u.au,
    clip_Rmax=Rmod_out*u.au,
    clip_Rgrid=R_s*u.au
)

#***********
#MAKE MASKS
#***********
masktuples_R = _get_mask_tuples(args.annuli)
masktuples_phi = _get_mask_tuples(args.wedges)

if len(masktuples_phi)==nmasks:
    pass
else:
    nmissing = nmasks - len(masktuples_phi)
    masktuples_phi = masktuples_phi + [[]]*nmissing
    
masks = []
xi, yi = [], []

for i in range(nmasks):

    if rcenters[i] != 0:
        x0 = rcenters[i]*np.cos(phicenters[i])
        y0 = rcenters[i]*np.sin(phicenters[i])
        Rgrid = np.hypot(Xproj - x0, Yproj-y0)
        phigrid = np.arctan2(Yproj - y0, Xproj - x0)
        xi.append(x0)
        yi.append(y0)

    else:
        Rgrid = R_s
        phigrid = phi_s
        xi.append(0)
        yi.append(0)

    mask_R={'map2d': Rgrid, 'lims': masktuples_R[i]}        
    mask_phi={'map2d': phigrid, 'lims': masktuples_phi[i]}

    mask0 = _merge_R_phi_mask(mask_R, mask_phi)
    if i>0:
        maski = mask0 & (~masks[i-1]) 
    else:
        maski = mask0
        
    masks.append(maski)

#*********
#MAKE PDF
#*********
chanwidth = np.abs(np.median(np.diff(datacube.vchannels)))
dv_native = 0.5*chanwidth

vmin = -5*dv_native + np.min(datacube.vchannels)   # generous margins
vmax =  5*dv_native + np.max(datacube.vchannels)
bins   = np.arange(vmin, vmax + dv_native, dv_native)
vcenters = 0.5*(bins[:-1] + bins[1:])

def thick_gaussian(v, sigma, tau=0.0, noise_mean=0.05, seed=10): #SNR 20 at planet location
    """
    Compute a thick (flat-topped) Gaussian intensity profile,
    optionally adding random Gaussian noise.

    Parameters
    ----------
    v : array-like
        Velocity or x-grid (centered at zero recommended).
    sigma : float
        Standard deviation of the underlying Gaussian (same units as v).
    tau : float, optional
        Optical depth factor. tau=0 gives a normal Gaussian,
        larger tau flattens the peak (optically thick limit).
    noise_mean : float, optional
        RMS of additive Gaussian noise (default 0 = no noise).
    rng : np.random.Generator, optional
        Random number generator for reproducibility (e.g., np.random.default_rng(42)).

    Returns
    -------
    I : ndarray
        Normalized intensity profile (max=1 if noise_mean=0).
    """
    G = np.exp(-0.5 * (v / sigma)**2)

    if tau == 0.0:
        I = G.copy()
    else:
        # Radiative transfer form: (1 - exp(-tau(v))) normalized to the line center
        I = (1.0 - np.exp(-tau * G)) / (1.0 - np.exp(-tau))

    # Add Gaussian-distributed random noise
    if noise_mean > 0.0:
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
            
        I += rng.normal(0.0, noise_mean, size=I.shape)

    # Clip to non-negative and renormalize if no saturation wanted
    I = np.clip(I, 0.0, None)
    if I.max() > 0:
        I /= I.max()

    return I


def thick_gaussian_random(vmin, vmax, sigma, tau=0.0, noise_mean=0.05, nsamples=50000,
                          seed=10, normalize_after_noise=True):
    """
    Generate a thick (flat-topped) Gaussian intensity distribution
    by sampling random velocities between vmin and vmax.

    Parameters
    ----------
    vmin, vmax : float
        Velocity range over which to draw random samples.
    sigma : float
        Standard deviation of the underlying Gaussian (same units as vmin/vmax).
    tau : float, optional
        Optical-depth-like factor. tau=0 -> pure Gaussian.
    noise_mean : float, optional
        RMS of additive Gaussian noise (0 = no noise).
    nsamples : int, optional
        Number of random velocity samples to draw.
    seed : int or None
        RNG seed for reproducibility.
    normalize_after_noise : bool, optional
        If True, renormalize max(I)=1 after adding noise.
        If False, keep the pre-noise peak normalized to 1.

    Returns
    -------
    v : ndarray
        Random velocity samples (uniformly distributed between vmin and vmax).
    I : ndarray
        Corresponding intensity samples, normalized to max=1.
    """
    rng = np.random.default_rng(seed)
    v = rng.uniform(vmin, vmax, nsamples)

    G = np.exp(-0.5 * (v / sigma)**2)

    if tau == 0.0:
        I = G
    else:
        # Use expm1 for numerical stability
        num = -np.expm1(-tau * G)
        den = -np.expm1(-tau)
        I = num / den

    # Add Gaussian-distributed noise
    if noise_mean > 0.0:
        I_noisy = I + rng.normal(0.0, noise_mean, size=I.shape)
        I_noisy = np.clip(I_noisy, 0.0, None)
        if normalize_after_noise and I_noisy.max() > 0:
            I_noisy /= I_noisy.max()
        I = I_noisy
    else:
        if I.max() > 0:
            I /= I.max()

    return v, I

def gaussian_reference_pdf(sigma, tau):
    vmin = -4*sigma
    vmax = 4*sigma
    v = np.arange(vmin, vmax+chanwidth, chanwidth)
    I = thick_gaussian(v, sigma, tau)
    I = I[np.abs(v)<4*sigma]

    #v, I = thick_gaussian_random(vmin, vmax, sigma, tau=tau)

    kde = gaussian_kde(I)
    Igrid = np.linspace(I.min(), I.max(), 300)
    pdf = kde(Igrid)
    return Igrid, pdf

def bin_add(v_dep, y, perbin):
    # v_dep: deprojected velocity axis for a spectrum (datacube.vchannels - vcent)
    # y: normalized intensities
    idx = np.digitize(v_dep, bins) - 1   # bin indices
    good = (idx >= 0) & (idx < len(vcenters)) & np.isfinite(y)
    for k in np.where(good)[0]:
        perbin[idx[k]].append(y[k])
        
def make_pdf(maski):
    intensities = datapdf[:, maski].ravel()
    intensities = intensities[intensities>args.sigma*noise_mean]
    intensities = intensities/np.nanmax(intensities)
    kde = gaussian_kde(intensities) #Normalise to peak intensity
    Igrid = np.linspace(np.nanmin(intensities), np.nanmax(intensities), 300)
    pdf = kde(Igrid)
    return Igrid, pdf

def make_spectra(maski):
    intensities = datapdf[:, maski]
    return intensities.T

def select_spectra(spectra, n):
    inds = random.sample(range(0, len(spectra)), n)
    return spectra[inds]

#**************
#MAKE PLT AXES
#**************
figx, figy = 12, 4
x2y = figx/figy

fig = plt.figure(figsize=(figx,figy))

axr = fig.add_axes([0.05, 0.05, 0.9/x2y, 0.9])
axp0 = 0.05+0.9/x2y+0.03
axp = fig.add_axes([axp0, 0.14, 0.90-axp0, 0.72])

if len(args.spectra)>0:
    axs = fig.add_axes([0.94, 0.22, 0.4, 0.56])
    spectra, draws = [], []
    zup, vphi, vcent = [], [], []

sigma_ref = 4 * dv_native #0.3 km/s
tau_ref = 20
Igrid_ref, pdf_ref = gaussian_reference_pdf(sigma_ref, tau_ref)
axp.plot(Igrid_ref, pdf_ref, ls='--', color='k', lw=1.0,
         label=fr'Gaussian ref ($\sigma={sigma_ref:.2f}$ km/s)', zorder=5)
print ('Thick-Gaussian skewness:', skew(pdf_ref))

Igrid_ref, pdf_ref = gaussian_reference_pdf(sigma_ref, tau_ref*0)
axp.plot(Igrid_ref, pdf_ref, ls=':', color='k', lw=1.0,
         label=fr'Gaussian ref ($\sigma={sigma_ref:.2f}$ km/s)', zorder=5)
print ('Gaussian skewness:', skew(pdf_ref))

#Ig = thick_gaussian(vcenters, sigma_ref, tau_ref*0)
#axs.plot(vcenters, Ig, 'tomato', lw=5, zorder=100)

ns = 0 #spectra counter
for i in range(nmasks):
    Igrid, pdf = make_pdf(masks[i])
    axp.plot(Igrid, pdf, color=colors[i], lw=lws[i], zorder=50-i)
    axp.fill_between(Igrid, pdf, color='k', lw=0, alpha=0.05)    

    #skewi = skew(Igrid*pdf)
    #kurti = kurtosis(Igrid*pdf)

    skewi = skew(pdf)
    kurti = kurtosis(pdf)
    
    """
    nsamples = 5000
    samples = np.random.choice(Igrid, size=nsamples, replace=True, p=pdf/np.sum(pdf))

    # Compute skewness and kurtosis of the intensity distribution
    skew_samples = skew(samples)
    kurt_samples = kurtosis(samples)

    print(f"Mask {i}: skew(PDF)={skew(pdf):.3f}, kurt(PDF)={kurtosis(pdf):.3f}, "
          f"skew(samples)={skew_samples:.3f}, kurt(samples)={kurt_samples:.3f}")
    """
    #print ('PDF i skew kurt:', i, skew(pdf), kurtosis(pdf))

    if len(args.spectra)>0:
        
        if i in args.spectra:
            
            perbin = [[] for _ in range(len(vcenters))]

            spectrai = make_spectra(masks[i])
            spectra.append(spectrai)
            
            if len(spectrai) > args.ndraws:
                drawsi = random.sample(range(0, len(spectrai)), args.ndraws)
            else:
                drawsi = range(0, len(spectrai))

            draws.append(drawsi)
            zupi = model.z_upper_func({'R': Rgrid[masks[i]]*au_to_m}, **best['height_upper'])
            vphii = model.velocity_func({'R': Rgrid[masks[i]]*au_to_m, 'z': zupi}, **best['velocity'])
            vcenti = vsys + vphii * np.sin(incl) * np.cos(np.radians(phigrid[masks[i]]))            
            zup.append(zupi)
            vphi.append(vphii)
            vcent.append(vcenti)
            peaki = np.nanmax(spectrai[drawsi]) #Global peak

            skewall = []
            kurtall = []            
            for j in drawsi:
                #peaki = np.nanmax(spectrai[j]) #Peak per spectrum
                spec = spectrai[j]/peaki
                v_dep = datacube.vchannels-vcenti[j]                
                bin_add(v_dep, spec, perbin)
                axs.plot(v_dep, spec, lw=0.1, color=colors[i], alpha=0.2, zorder=50-i)

                skewall.append(skew(spec))
                kurtall.append(kurtosis(spec))

            print ('Spectra i med(skew), med(kurt)', i, np.median(skewall), np.median(kurtall))
            
            med_i = np.empty(len(vcenters)); med_i[:] = np.nan
            p16_i = np.empty(len(vcenters)); p16_i[:] = np.nan
            p84_i = np.empty(len(vcenters)); p84_i[:] = np.nan
            cnt_i = np.zeros(len(vcenters), dtype=int)

            for b in range(len(vcenters)):
                if perbin[b]:
                    arr = np.asarray(perbin[b], float)
                    cnt_i[b] = arr.size
                    med_i[b] = np.median(arr)
                    p16_i[b] = np.percentile(arr, 16)
                    p84_i[b] = np.percentile(arr, 84)

            # (optional) mask bins with too few samples
            min_count = 3
            sparse = cnt_i < min_count
            med_i[sparse] = np.nan; p16_i[sparse] = np.nan; p84_i[sparse] = np.nan

            # ----- plot per-i median + spread -----
            if colors[i] in ['k', 'black']:
                fci = '0.9'
            else:
                fci = colors[i]
                
            axs.scatter(vcenters, med_i, lw=1.5, ec='k', color=fci, s=30, zorder=100-i)
            
            #axs.plot(vcenters, med_i, lw=4.0, color=colors[i], alpha=0.4, zorder=99-i)        
            #axs.fill_between(vcenters, p16_i, p84_i, alpha=0.15, color=colors[i], linewidth=0, zorder=90-i)

            #axs.text(0.20, 0.85-0.12*ns, '%.2f'%skewi, fontsize=12, ha='right', va='top', color=colors[i], transform=axs.transAxes)
            ns += 1        

            """
            good = np.isfinite(med_i)
            v = vcenters[good]
            I = med_i[good]

            if I.size == 0 or np.all(I <= 0):
                print(f"[warn] Empty or zero med_i for region {i}")
            else:
                I = np.clip(I, 0, None)
                w_sum = np.sum(I)
                mu_v = np.sum(I * v) / w_sum
                var_v = np.sum(I * (v - mu_v)**2) / w_sum
                std_v = np.sqrt(var_v)
                mu3_v = np.sum(I * (v - mu_v)**3) / w_sum
                skew_weighted_velocity = mu3_v / (std_v**3)
                
            print(f"Region {i}: intensity-weighted velocity skewness = {skew_weighted_velocity:.3f}")
            """
    axp.text(0.2, 0.85-0.09*i, '%.2f'%skewi, fontsize=args.fontsize-1, ha='right', va='top', color=colors[i], transform=axp.transAxes)

axp.text(0.2, 0.92, 'Skewness', weight='bold', fontsize=args.fontsize-1, ha='right', va='top', color='0.6', transform=axp.transAxes)

make_up_ax(axp, labelleft=False, labelright=True, labeltop=False, labelbottom=True, labelsize=16)
axp.set_xticks([0.0, 0.5, 1.0])
axp.set_yticks([0, 1, 2])
axp.set_xticklabels([0, r'0.5I$_{\rm peak}$', r'I$_{\rm peak}$'])

#axp.text(0, 1.02, 'Intensity Distribution of Velocity Channels', fontsize=12, transform=axp.transAxes)
if args.show_title:
    axp.text(0.5, 1.06, 'Probability Density of Intensities across Velocity Channels', fontsize=args.fontsize-2.5, transform=axp.transAxes, ha='center', va='center')

axp.yaxis.set_label_position("right")
#axp.set_ylabel('Probability density', fontsize=14, labelpad=15)
axp.set_xlim(0, 1)
axp.set_ylim(0, None)

if len(args.spectra)>0:    
    make_up_ax(axs, labelleft=False, labelright=True, labeltop=False, labelbottom=True, labelsize=14)
    axs.tick_params(axis='y', pad=-80, labelcolor='k')
    axs.axvline(0, dash_capstyle='round', dashes=(3.0, 2.5), color='k', zorder=0)
    axs.axhline(0.5, xmin=0.5, dash_capstyle='round', dashes=(0.5, 1.5), color='k', zorder=0)
    axs.axhline(1.0, xmin=0.5, dash_capstyle='round', dashes=(0.5, 1.5), color='k', zorder=0)
    axs.set_yticks([0.5, 1.0])
    axs.set_yticklabels([r'0.5I$_{\rm peak}$', r'I$_{\rm peak}$'])

    #axs.text(0.20, 0.95, 'Skewness', fontsize=12, ha='right', va='top', color='0.6', transform=axs.transAxes)

    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1)
    plt.setp(axs.get_yticklabels(), bbox=bbox)

    axs.set_xticks([-0.5, 0.0, 0.5])
    axs.set_xlabel('Stacked velocities [km/s]')
    axs.set_xlim(-1.1, 1.1)
    axs.set_ylim(0, 1.07)

#***************
#MAKE ROUND MAP
#***************
kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=levels_im)

if args.surface in ['up', 'upper']:
    z_func = model.z_upper_func
    z_pars = best['height_upper']

elif args.surface in ['low', 'lower']:
    z_func = model.z_lower_func
    z_pars = best['height_lower']

levels_resid = np.linspace(-clim, clim, 32)
        
fig, axr = make_round_map(1000*residuals, levels_resid, Xproj*u.au, Yproj*u.au, Rmod_out*u.au,
                          fig=fig, ax=axr,
                          kwargs_contourf=dict(cmap=cmap_res), #Skip set_under, set_over by overwriting cmap
                          z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                          cmap=cmap_res, clabel='m/s', fmt='%d',
                          gaps=gaps, rings=rings, kinks=kinks,
                          make_cbar=args.colorbar,                         
                          mask_inner=Rmod_in*u.au,
                          fontsize_azimuthal_grid=args.fontsize-4,
                          fontsize_radial_grid=args.fontsize-3, 
                          fontsize_cbar=args.fontsize-2,
                          fontsize_xaxis=args.fontsize-4,
                          fontsize_nskyaxis=args.fontsize-2,
                          make_radial_lines=False,
                          make_nskyaxis=args.show_nsky,
                          make_Rout_proj=args.show_xaxis,
                          make_xaxis=args.show_xaxis,                          
)

#Filled regions
for i in np.arange(0, len(masks))[::-1]:
    
    if rcenters[i] != 0:
        colorcirc = 'k'
        alpha = 0.5
        lw = 2
    else:
        colorcirc = colors[i]
        alpha = 0.3
        lw = 1

    if np.any(masks[i]):
        axr.contourf(
            Xproj, Yproj, masks[i].astype(float),
            levels=[0.5, 1.5],
            colors=[colors[i]],
            alpha=alpha,
            #hatches=['///'],
            #zorder=30 - i,
            antialiased=True
        )

    circle = plt.Circle((xi[i], yi[i]), masktuples_R[i][-1], color=colorcirc, lw=lw, fill=False)#, zorder=10-i)
    axr.add_patch(circle)
    circle = plt.Circle((xi[i], yi[i]), masktuples_R[i][0], color=colorcirc, lw=lw, fill=False)#, zorder=10-i)
    axr.add_patch(circle)

if len(args.mask_R)>0 or len(args.mask_phi)>0:
    make_masks(axr, args.mask_R, args.mask_phi, Rmax=Rmod_out)
    
mark_planet_location(axr, args, edgecolors='k', lw=0, s=550, coords='disc', model=model)    

plt.savefig('intensity_distribution_%s_%s.png'%(meta['disc'], meta['mol']), bbox_inches='tight', dpi=200)    
show_output(args)

