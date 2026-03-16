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

import sys
import json
import random
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_intensdistrib(None)
    args = parser.parse_args()

stat_func = getattr(np, args.stat)

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

R_prof = np.arange(Rmod_in, Rmod_out, beam_au/args.binsperbeam)

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
masktuples_R = _get_mask_tuples(R_prof, consecutive=True)
masktuples_phi = _get_mask_tuples([])

bincenters = np.mean(masktuples_R, axis=1)
bincenters_arcsec = bincenters/dpc.value

#*******************
#FILL UP EMPTY ARGS
#*******************
nmasks = len(masktuples_R)

cmap = plt.get_cmap(args.cmap)
colors = cmap(np.linspace(0.15, 0.85, nmasks))
lws = [1.2] * nmasks
    
if len(masktuples_phi)==nmasks:
    pass
else:
    nmissing = nmasks - len(masktuples_phi)
    masktuples_phi = masktuples_phi + [[]]*nmissing
    
masks = []
xi, yi = [], []

for i in range(nmasks):

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

#**********************************
#LOAD EXTERNAL RADIAL vphi PROFILE
#**********************************
vphi_filename = 'radial_profile_velocity_data.dat'

if not args.keplerian:
    try:
        vphi_file = np.loadtxt(vphi_filename, comments='#')
    except Exception:
        sys.exit(
            f"\nERROR: Could not load '{vphi_filename}'.\n\n"
            "Run:\n"
            "    discminer radprof\n"
            "to generate a radial velocity profile of your data, or use:\n"
            "    discminer stack -keplerian 1\n"
            "to fall back to a pure Keplerian profile for the stacking.\n"
        )

    R_vphi_au = vphi_file[:, 0] # first column: radius [au]
    vphi_prof = vphi_file[:, 1] # second column: vphi profile
    vphi_std  = vphi_file[:, 2] # optional third column: stddev

    isort = np.argsort(R_vphi_au)
    R_vphi_au = R_vphi_au[isort]
    vphi_prof = vphi_prof[isort]
    vphi_std  = vphi_std[isort]

    vphi_interp = interp1d(
        R_vphi_au,
        vphi_prof,
        bounds_error=False,
        fill_value=np.nan
    )

#************
#GET SPECTRA
#************
chanwidth = np.abs(np.median(np.diff(datacube.vchannels)))
dv_native = chanwidth/args.binsperchan

vmin = -10*dv_native + np.min(datacube.vchannels)   # generous margins
vmax =  10*dv_native + np.max(datacube.vchannels)
bins   = np.arange(vmin, vmax + dv_native, dv_native)
vcenters = 0.5*(bins[:-1] + bins[1:])

def bin_add(v_dep, y, perbin):
    # v_dep: deprojected velocity axis for a spectrum (datacube.vchannels - vcent)
    # y: intensities
    idx = np.digitize(v_dep, bins) - 1   # bin indices
    good = (idx >= 0) & (idx < len(vcenters)) & np.isfinite(y)
    for k in np.where(good)[0]:
        perbin[idx[k]].append(y[k])
        
def make_spectra(maski):
    intensities = datapdf[:, maski]
    return intensities.T

#**************
#MAKE PLT AXES
#**************
figx, figy = 14, 6
fig = plt.figure(figsize=(figx, figy))

gs = GridSpec(
    nrows=1,
    ncols=2,
    figure=fig,
    width_ratios=[1.0, 2.2],
    wspace=0.18
)

axr = fig.add_subplot(gs[0, 0])

ncols_spec = int(np.ceil(np.sqrt(nmasks)))
nrows_spec = int(np.ceil(nmasks / ncols_spec))

gs_spec = gs[0, 1].subgridspec(
    nrows=nrows_spec,
    ncols=ncols_spec,
    hspace=0.12,
    wspace=0.12
)

axs_all = [fig.add_subplot(gs_spec[r, c]) for r in range(nrows_spec) for c in range(ncols_spec)]
axs = axs_all[:nmasks]

for ax in axs_all[nmasks:]:
    ax.set_visible(False)

#MAKE SPECTRA
stacked_profiles = []
spectra, draws = [], []
zup, vphi, vcent = [], [], []

for i in range(nmasks):

    perbin = [[] for _ in range(len(vcenters))]

    spectrai = make_spectra(masks[i])
    spectra.append(spectrai)

    if (args.ndraws < 0) or (args.ndraws > len(spectrai)):
        drawsi = range(0, len(spectrai))
    else:
        drawsi = random.sample(range(0, len(spectrai)), args.ndraws)

    draws.append(drawsi)

    Rmask_au = Rgrid[masks[i]]
    zupi = model.z_upper_func({'R': Rmask_au*au_to_m}, **best['height_upper'])

    if args.keplerian:
        vphii = model.velocity_func({'R': Rmask_au*au_to_m, 'z': zupi}, **best['velocity'])
    else:
        vphii = vphi_interp(Rmask_au)
        
    vcenti = vsys + vphii * np.sin(incl) * np.cos(np.radians(phigrid[masks[i]])) #Line centroid
    
    zup.append(zupi)
    vphi.append(vphii)
    vcent.append(vcenti)
    peaki = np.nanmax(spectrai[drawsi])

    for j in drawsi:
        spec = 1e3 * spectrai[j]
        v_dep = datacube.vchannels - vcenti[j]
        bin_add(v_dep, spec, perbin)
        axs[i].plot(v_dep, spec, lw=0.1, color='k', alpha=0.15, zorder=50-i)

    stat_i = np.empty(len(vcenters)); stat_i[:] = np.nan
    p16_i = np.empty(len(vcenters)); p16_i[:] = np.nan
    p84_i = np.empty(len(vcenters)); p84_i[:] = np.nan
    cnt_i = np.zeros(len(vcenters), dtype=int)
    
    for b in range(len(vcenters)):
        if perbin[b]:
            arr = np.asarray(perbin[b], float)
            cnt_i[b] = arr.size
            stat_i[b] = stat_func(arr)
            p16_i[b] = np.percentile(arr, 16)
            p84_i[b] = np.percentile(arr, 84)

    min_count = 3
    sparse = cnt_i < min_count
    stat_i[sparse] = np.nan
    p16_i[sparse] = np.nan
    p84_i[sparse] = np.nan

    stacked_profiles.append(stat_i)
    
    axs[i].scatter(vcenters, stat_i, lw=1.0, ec='k', color=colors[i], s=15, zorder=100-i)
    axs[i].text(0.07, 0.93, '%d au'%bincenters[i], fontsize=10, ha='left', va='top', color=colors[i], transform=axs[i].transAxes)
    axs[i].text(0.07, 0.81, "$%.2f''$"%bincenters_arcsec[i], fontsize=10, ha='left', va='top', color=colors[i], transform=axs[i].transAxes)     

#***************
#IMPROVE LAYOUT
#***************
for i, axi in enumerate(axs):
    row = i // ncols_spec
    col = i % ncols_spec

    is_bottom = (row == nrows_spec - 1)
    is_right  = (col == ncols_spec - 1)

    make_up_ax(
        axi,
        labelleft=False,
        labelright=is_right,
        labeltop=False,
        labelbottom=is_bottom,
        labelsize=10
    )

    axi.axvline(0, dash_capstyle='round', dashes=(3.0, 2.5), color='k', zorder=0)
    axi.tick_params(axis='y', pad=5, labelcolor='k')

    axi.set_xticks(np.linspace(-np.floor(args.vlim), np.floor(args.vlim), 5).astype(int))
    axi.set_xlim(-args.vlim, args.vlim)

    if not is_bottom:
        axi.tick_params(labelbottom=False)
        
    if not is_right:
        axi.tick_params(labelright=False, right=True)

axs[-1].set_xlabel('Stacked velocities [km/s]', fontsize=12)
axs[-3].yaxis.set_label_position("right")
axs[-3].set_ylabel('[mJy/beam]', fontsize=12, labelpad=15, rotation=270)

#*************
#WRITE OUTPUT
#*************
stacked_profiles = np.array(stacked_profiles).T
out = np.column_stack([vcenters, stacked_profiles])
header = " vel_bin, " + ", ".join(f"{r:.1f} au" for r in bincenters)

np.savetxt(
    "stacked_spectra_%s_%s_kepler%d_%s.txt"%(meta['disc'], meta['mol'], args.keplerian, args.stat),
    out,
    header=header,
    fmt="%.3f"
)

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
                          fontsize_azimuthal_grid=args.fontsize-6,
                          fontsize_radial_grid=args.fontsize-5, 
                          fontsize_cbar=args.fontsize-4,
                          fontsize_xaxis=args.fontsize-6,
                          fontsize_nskyaxis=args.fontsize-4,
                          make_radial_lines=False,
                          make_nskyaxis=args.show_nsky,
                          make_Rout_proj=args.show_xaxis,
                          make_xaxis=args.show_xaxis,                          
)

#Filled regions
for i in np.arange(0, len(masks))[::-1]:
    
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

    circle = plt.Circle((xi[i], yi[i]), masktuples_R[i][-1], color=colorcirc, lw=lw, fill=False)
    axr.add_patch(circle)
    circle = plt.Circle((xi[i], yi[i]), masktuples_R[i][0], color=colorcirc, lw=lw, fill=False)
    axr.add_patch(circle)

if len(args.mask_R)>0 or len(args.mask_phi)>0:
    make_masks(axr, args.mask_R, args.mask_phi, Rmax=Rmod_out)
    
mark_planet_location(axr, args, edgecolors='k', lw=0, s=550, coords='disc', model=model)    

plt.savefig('line_stacking_%s_%s_kepler%d_%s.png'%(meta['disc'], meta['mol'], args.keplerian, args.stat), bbox_inches='tight', dpi=200)    
show_output(args)

