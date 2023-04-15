from discminer.plottools import (get_discminer_cmap, append_sigma_panel,
                                 make_up_ax, mod_minor_ticks, mod_major_ticks,
                                 use_discminer_style, mod_nticks_cbars,
                                 make_substructures, make_round_map)
from discminer.pick import Pick
from discminer.rail import Contours

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
import json
import copy

from utils import (init_data_and_model,
                   get_noise_mask,
                   add_parser_args,
                   get_2d_plot_decorators,
                   load_moments)

from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='Identify and show peak residuals', description='Identify peak residuals in folded maps.')
parser.add_argument('-c', '--clean_thres', default=np.inf, type=float, help="Threshold above which peak residuals will be rejected.")
args = add_parser_args(parser, moment=True, kind=True, surface=True, fold=True, projection=True, Rinner=True, Router=True)

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

gaps = custom['gaps']
rings = custom['rings']

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, unit_simple=True, fmt_vertical=True)

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.1*Rout*u.au #Max model radius, 10% larger than disc Rout

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model()

noise_mean, mask = get_noise_mask(datacube, thres=2)
vchannels = datacube.vchannels
model.make_model()

#*************************
#LOAD MOMENT MAPS
moment_data, moment_model, mtags = load_moments(args)

#**************************
#MASK AND COMPUTE RESIDUALS
moment_data = np.where(mask, np.nan, moment_data)
moment_model = np.where(mask, np.nan, moment_model)
moment_residuals = moment_data - moment_model
    
if args.moment=='velocity' and args.fold=='absolute':
    moment_residuals = np.abs(moment_data-vsys) - np.abs(moment_model-vsys)

#*******************
#FIND PEAK RESIDUALS
#*******************
beam_au = datacube.beam_size.to('au').value
R_prof = np.arange(args.Rinner*beam_au, args.Router*Rout, beam_au/5)
xlim0, xlim1 = 0.5*R_prof[0], 1.05*R_prof[-1]

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(11,5))    
ax_c = fig.add_axes([0.8,0.65,0.23*5/11,0.23]) #AxesSubplot for showing contour colours 

pick = Pick(model, moment_residuals, R_prof, fold=True, color_bounds = np.array([0.33, 0.66, 1.0])*Rout, ax2=ax_c)
folded_map = pick.make_2d_map() #Map where peaks will be picked from
pick.find_peaks(clean_thres=args.clean_thres)

lev = pick.lev_list
color = pick.color_list
peak_resid = pick.peak_resid
peak_angle = pick.peak_angle
peak_resid = pick.peak_resid

model.make_emission_surface(ax_c)
model.make_disc_axes(ax_c)
ax_c.axis('off')

#*******************
#SHOW PEAK RESIDUALS
#*******************
kwargs_sc = dict(facecolors=color, edgecolors='0.1', lw=0.7, alpha=0.7, zorder=10)

ax[0].set_xlabel(r'Azimuth [deg]')
ax[0].set_ylabel('Peak residual [%s]'%unit)
ax[1].set_xlabel('Radius [au]')

for axi in ax: 
    axi.tick_params(labelbottom=True, top=True, right=True, which='both', direction='in')    
    mod_major_ticks(axi, nbins=8)
    mod_minor_ticks(axi)
ax[0].set_xlim(-95,95)
ax[0].set_xticks(np.arange(-90,90+1,30))    
ax[1].tick_params(labelleft=False)

ax[0].scatter(peak_angle, peak_resid, **kwargs_sc)
ax[1].scatter(lev, peak_resid, **kwargs_sc)

ax[0].axvline(pick.peak_global_angle, lw=3, c='k', label='global peak', zorder=11)
ax[1].axvline(pick.peak_global_radius, lw=3, c='k', zorder=11) 
ax[0].legend(frameon=False, fontsize=15, handlelength=1.0, loc='lower left', bbox_to_anchor=(-0.04, 0.98))

make_substructures(ax[1], gaps=gaps, rings=rings)
append_sigma_panel(fig, ax, peak_resid, weights=pick.peak_weight, hist=True)
plt.savefig('peak_residuals_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)
plt.close()

#***************
#SHOW FOLDED MAP
#***************
R = model.projected_coords['R']
phi = model.projected_coords['phi']

Xproj = R[args.surface]*np.cos(phi[args.surface])
Yproj = R[args.surface]*np.sin(phi[args.surface])

cos_peak = np.cos(np.radians(peak_angle))
sin_peak = np.sin(np.radians(peak_angle))

if args.projection=='cartesian':
    levels_resid = np.linspace(-clim, clim, 32)
    
    if args.surface in ['up', 'upper']:
        z_func = model.z_upper_func
        z_pars = best['height_upper']

    elif args.surface in ['low', 'lower']:
        z_func = model.z_lower_func
        z_pars = best['height_lower']
    
    fig, ax = make_round_map(folded_map, levels_resid, pick.X*u.au, pick.Y*u.au, R_prof[-1]*u.au,
                             z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                             cmap=cmap_res, clabel=unit, fmt=cfmt, 
                             rings=rings,
                             mask_wedge=(90, 270)*u.deg,
                             mask_inner=R_prof[0]*u.au)
    ax.scatter(lev*cos_peak, lev*sin_peak, edgecolors='none', facecolors=color, alpha=0.2, s=100)
    ax.scatter(lev*cos_peak, lev*sin_peak, edgecolors='none', facecolors=color, alpha=1.0, s=10)
    ax.scatter(lev*cos_peak, lev*sin_peak, edgecolors='0.3', facecolors='none', alpha=1.0, s=100)
    ax.set_title('%s, folded map'%ctitle, fontsize=16, color='k')
    

plt.savefig('folded_residuals_deproj_%s_%s.png'%(mtags['base'], args.projection), bbox_inches='tight', dpi=200)
plt.close()

#*************
#FIND CLUSTERS
#*************
pick.find_clusters(n_phi=8, n_R=8)

#*************
#PLOT CLUSTERS
#*************
