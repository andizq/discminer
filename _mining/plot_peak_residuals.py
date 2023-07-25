from discminer.plottools import (get_discminer_cmap, append_sigma_panel,
                                 make_up_ax, mod_minor_ticks, mod_major_ticks,
                                 use_discminer_style, mod_nticks_cbars,
                                 make_substructures, make_round_map,
                                 get_cmap_from_color, make_clusters_1d)
from discminer.pick import Pick
from utils import (init_data_and_model,
                   get_noise_mask,
                   add_parser_args,
                   get_2d_plot_decorators,
                   load_moments)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import json
from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='Identify and show peak residuals', description='Identify peak residuals in folded maps.')
parser.add_argument('-c', '--clean_thres', default=np.inf, type=float, help="Threshold above which peak residuals will be rejected.")
parser.add_argument('-np', '--nphi', default=6, type=int, help="Number of azimuthal clusters.")
parser.add_argument('-nr', '--nr', default=6, type=int, help="Number of radial clusters.")
parser.add_argument('-sp', '--show_peaks', default=1, type=int, help="Show peak residuals on 2D maps.")
args = add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, fold=True, projection=True, Rinner=1.5, Router=0.95, smooth=True)

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
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)
    
if args.moment=='velocity' and args.fold=='absolute':
    residuals = np.abs(moment_data-vsys) - np.abs(moment_model-vsys)

#*******************
#FIND PEAK RESIDUALS
#*******************
beam_au = datacube.beam_size.to('au').value
R_prof = np.arange(args.Rinner*beam_au, args.Router*Rout, beam_au/5)
xlim0, xlim1 = 0.5*R_prof[0], 1.05*R_prof[-1]

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(11,5))    
ax_c = fig.add_axes([0.8,0.65,0.23*5/11,0.23]) #AxesSubplot for showing contour colours 

pick = Pick(model, residuals, R_prof, fold=True, color_bounds = np.array([0.33, 0.66, 1.0])*Rout, ax2=ax_c)
folded_map = pick.make_2d_map() #Map where peaks will be picked from
pick.find_peaks(clean_thres=args.clean_thres)

lev = pick.lev_list
color = pick.color_list
peak_resid = pick.peak_resid
peak_angle = pick.peak_angle

model.make_emission_surface(ax_c)
model.make_disc_axes(ax_c)
ax_c.axis('off')

#*******************
#SHOW PEAK RESIDUALS
#*******************
kwargs_sc = dict(facecolors=color, edgecolors='0.1', lw=0.7, alpha=0.7, zorder=10)
color_global_peak = 'lime'
color_cluster_peak = 'lime'

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

ax[0].axvline(pick.peak_global_angle, lw=3.5, c=color_global_peak, label='global peak', zorder=0)
ax[1].axvline(pick.peak_global_radius, lw=3.5, c=color_global_peak, zorder=0) 
ax[0].legend(frameon=False, fontsize=15, handlelength=1.0, loc='lower left', bbox_to_anchor=(-0.04, 0.98))

make_substructures(ax[1], gaps=gaps, rings=rings)
append_sigma_panel(fig, ax, peak_resid, weights=pick.peak_weight, hist=True)
plt.savefig('peak_residuals_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)
#plt.show()
plt.close()

#sys.exit()

#*************
#FIND CLUSTERS
#*************
try:
    pick.find_clusters(n_phi=args.nphi, n_R=args.nr)
    found_clusters = True    
except np.linalg.LinAlgError as e:
    print(30*'*')    
    print(repr(e))
    print('Cluster finder algorithm failed due to the low number of input points. Try changing the number of clusters to use or the disc extent...')
    print(30*'*')        
    found_clusters = False
    
#*************
#PLOT CLUSTERS
#*************
if found_clusters:
    fig, ax = make_clusters_1d(pick, which='phi')
    plt.savefig('clusters_phi_peak_residuals_%s_%dclust.png'%(mtags['base'], args.nphi), bbox_inches='tight', dpi=200)
    plt.close()
    #plt.show()

    fig, ax = make_clusters_1d(pick, which='r')
    plt.savefig('clusters_r_peak_residuals_%s_%dclust.png'%(mtags['base'], args.nr), bbox_inches='tight', dpi=200)
    plt.close()
    #plt.show()

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
    make_substructures(
        ax, gaps=gaps, rings=rings, twodim=True, label_rings=True,
    )#kwargs_rings={'lw': 0.6})

    if args.show_peaks:
        ax.scatter(lev*cos_peak, lev*sin_peak, edgecolors='none', facecolors=color, alpha=0.2, s=100, zorder=20)
        ax.scatter(lev*cos_peak, lev*sin_peak, edgecolors='none', facecolors=color, alpha=1.0, s=10, zorder=20)
        ax.scatter(lev*cos_peak, lev*sin_peak, edgecolors='0.3', facecolors='none', alpha=1.0, s=100, zorder=20)

    #Show global peak
    ax.scatter(pick.peak_global_radius*np.cos(np.radians(pick.peak_global_angle)),
               pick.peak_global_radius*np.sin(np.radians(pick.peak_global_angle)),
               edgecolors='k', facecolors=color_global_peak,
               marker='o', lw=2.5, alpha=0.7, s=450, zorder=21)

    print ('Global peak residual found at R=%.1f au, phi=%.1f deg, in disc frame coordinates...'%(pick.peak_global_radius, pick.peak_global_angle))
    
    if found_clusters:
        
        if len(pick.acc_peaks_phi)>0 and len(pick.acc_peaks_R)>0:
            #Show cluster peak
            ax.scatter(pick.acc_R*np.cos(np.radians(pick.acc_phi)),
                       pick.acc_R*np.sin(np.radians(pick.acc_phi)),
                       edgecolors='k', facecolors=color_cluster_peak,
                       marker='X', lw=2.5, alpha=0.7, s=450, zorder=21)
        
            print ('Weighted centre of accepted clusters found at R=%.1f au, phi=%.1f deg, in disc frame coordinates...'%(pick.acc_R, pick.acc_phi))
            
    ax.set_title('%s, folded map'%ctitle, fontsize=16, color='k')

pick.writetxt(filename='pick_summary_%s.txt'%mtags['base'])

plt.savefig('folded_residuals_deproj_%s_%s.png'%(mtags['base'], args.projection), bbox_inches='tight', dpi=200)
#plt.show()
plt.close()
