from discminer.mining_control import _mining_pick
from discminer.mining_utils import (init_data_and_model,
                                    get_noise_mask,
                                    get_2d_plot_decorators,
                                    load_moments,
                                    load_disc_grid,
                                    mark_planet_location,
                                    make_masks,
                                    format_sky_coords,
                                    show_output)
from discminer.pick import Pick
from discminer.grid import GridTools

from discminer.plottools import (get_discminer_cmap,
                                 append_sigma_panel,
                                 mod_minor_ticks,
                                 mod_major_ticks,
                                 use_discminer_style,
                                 make_substructures,
                                 make_round_map,
                                 make_polar_map,
                                 get_cmap_from_color,
                                 make_clusters_1d,
                                 make_1d_legend)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits

import json
import os

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_pick(None)
    args = parser.parse_args()

#**********************
#JSON AND PARSER STUFF
#**********************
if args.moment=='continuum':
    parfile = 'parfile_band7.json'
else:
    parfile = 'parfile.json'
    
with open(parfile) as json_file:
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

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, parfile=parfile, unit_simple=True, fmt_vertical=True)

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
if args.moment=='continuum':
    datacube, model = init_data_and_model(parfile=parfile, twodim=True, write_extent=True)
    datacube.convert_to_tb(writefits=False, planck=False)
else:
    datacube, model = init_data_and_model(parfile=parfile)
    
vchannels = datacube.vchannels
model.make_model()

beam_au = datacube.beam_size.to('au').value
if args.absolute_Rinner>=0:
    Rmod_in = args.absolute_Rinner
else:
    Rmod_in = args.Rinner*beam_au

if args.absolute_Router>=0:
    Rmod_out = args.absolute_Router
else:
    Rmod_out = args.Router*Rout

gaps = np.array(custom['gaps'])
rings = np.array(custom['rings'])
gaps = gaps[gaps<Rmod_out]
rings = rings[rings<Rmod_out]

Rmax = 1.1*Rmod_out*u.au #Max window extent, 10% larger than disc Rout

R, phi, z = load_disc_grid()
noise_mean, mask = get_noise_mask(datacube, thres=2,
                                  mask_phi={'map2d': np.degrees(phi['upper']),
                                            'lims': args.mask_phi},
                                  mask_R={'map2d': R['upper']/au_to_m,
                                          'lims': args.mask_R}
)

#noise_mean, mask = get_noise_mask(datacube, thres=2)

if args.surface in ['up', 'upper']:
    z_func = model.z_upper_func
    z_pars = best['height_upper']

elif args.surface in ['low', 'lower']:
    z_func = model.z_lower_func
    z_pars = best['height_lower']

#*************************
#LOAD MOMENT MAPS
if args.moment=='continuum':
    residuals = datacube.data[0]
    mtags = dict(dir_model=meta['dir_model'], base=f'{args.moment}')
else:
    moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)
    ref_surf = mtags['ref_surf']

if args.moment=='velocity' and args.fold=='absolute':
    residuals = np.abs(moment_data-vsys) - np.abs(moment_model-vsys)
    residuals[mask] = np.nan
    
if args.moment=='velocity' and args.percentage_kepler:
    coords = {'R': model.projected_coords['R']['upper']}
    velocity_kepler = model.get_attribute_map(coords, 'velocity', surface=ref_surf) * vel_sign    
    residuals = residuals/velocity_kepler
    
#********************

#*******************
#FIND PEAK RESIDUALS
#*******************
R_prof = np.arange(Rmod_in, Rmod_out, beam_au/5)
xlim0, xlim1 = 0.5*R_prof[0], 1.05*R_prof[-1]

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(11,5))    
ax_c = fig.add_axes([0.8,0.65,0.23*5/11,0.23]) #AxesSubplot for showing contour colours 

pick = Pick(model, residuals, R_prof, fold=True, fold_func=args.fold_func, color_bounds = np.array([0.33, 0.66, 1.0])*Rmod_out, ax2=ax_c)
xf, yf, folded_orig, folded_map = pick.make_2d_map(return_coords=True) #Map where peaks will be picked from
        
figh, axh = plt.subplots(ncols=1, nrows=1, figsize=(9,6))    
pick.find_peaks(clean_thres=args.clean_thres, phi_min=args.phimin, phi_max=args.phimax, fig_ax_histogram=(figh, axh), clean_histogram=True)
figh.savefig('histogram_peak_residuals_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)

lev = pick.lev_list
color = pick.color_list
peak_resid = pick.peak_resid
peak_angle = pick.peak_angle

model.make_emission_surface(ax_c)
model.make_disc_axes(ax_c)
ax_c.axis('off')

def mad_sigma(x):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

median_bg = np.median(peak_resid)
sigma_bg  = mad_sigma(peak_resid)

global_peak = np.max(peak_resid)
significance = (global_peak - median_bg) / sigma_bg

print ('MAD:', median_bg)
print ('sigma:', sigma_bg)
print ('Global peak:', global_peak)
print ('Significance:', significance)

#*******************
#SHOW PEAK RESIDUALS
#*******************
kwargs_sc = dict(facecolors=color, edgecolors='0.1', lw=1.2, alpha=0.7, s=70, zorder=10)
color_global_peak = 'lime'
color_cluster_peak = 'lime'

ax[0].set_xlabel(r'Azimuth [deg]', fontsize=args.fontsize-4)
ax[0].set_ylabel('Peak residual [%s]'%unit, fontsize=args.fontsize-4)
ax[1].set_xlabel('Radius [au]', fontsize=args.fontsize-4)

for axi in ax: 
    axi.tick_params(labelbottom=True, top=True, right=True, which='both', direction='in', labelsize=args.fontsize-6)    
    mod_major_ticks(axi, nbins=8)
    mod_minor_ticks(axi)
ax[0].set_xlim(-95,95)
ax[0].set_xticks(np.arange(-90,90+1,30))    
ax[1].tick_params(labelleft=False)

ax[0].scatter(peak_angle, peak_resid, **kwargs_sc)
ax[1].scatter(lev, peak_resid, **kwargs_sc)

ax[0].axvline(pick.peak_global_angle, lw=3.5, c=color_global_peak, label='global', zorder=0) #label = 'global peak'
ax[1].axvline(pick.peak_global_radius, lw=3.5, c=color_global_peak, zorder=0) 

for axi in ax:
    axi.axhline(median_bg+3*sigma_bg, color='magenta', ls='--', lw=2.2, dash_capstyle='round', dashes=(3.0, 2.5), alpha=1.0)

ax[0].legend(frameon=False, fontsize=args.fontsize-2, handlelength=0.8, columnspacing=1.0, handletextpad=0.35, loc='lower left', bbox_to_anchor=(-0.04, 0.98), ncols=2)

make_substructures(ax[1], gaps=gaps, rings=rings)
append_sigma_panel(fig, ax, peak_resid, weights=pick.peak_weight, hist=True, linecolor='0.8')
fig.savefig('peak_residuals_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)
#plt.show()
plt.close()


#*************
#FIND CLUSTERS
#*************
if args.clusters:
    try:
        pick.find_clusters(n_phi=args.nphi, n_R=args.nr)
        found_clusters = True    
    except np.linalg.LinAlgError as e:
        print(30*'*')    
        print(repr(e))
        print('Cluster finder algorithm failed due to the low number of input points. Try changing the number of clusters to use or the disc extent...')
        print(30*'*')        
        found_clusters = False
else:
    found_clusters = False    
    
#*************
#PLOT CLUSTERS
#*************
if found_clusters and args.clusters:
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
    qcbar = None if args.quadrant_cbar==0 else args.quadrant_cbar
        
    fig, ax = make_round_map(folded_map, levels_resid, pick.X*u.au, pick.Y*u.au, R_prof[-1]*u.au,
                             z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                             cmap=cmap_res, clabel=unit, fmt=cfmt,
                             make_cbar=args.colorbar, 
                             rings=rings,                             
                             mask_wedge=(90, 270)*u.deg,
                             mask_inner=R_prof[0]*u.au,
                             fontsize_azimuthal_grid=args.azlabels*args.fontsize,
                             fontsize_radial_grid=args.rlabels*(args.fontsize+3), 
                             fontsize_cbar=args.fontsize+0,
                             fontsize_xaxis=args.fontsize+3,
                             fontsize_nskyaxis=args.fontsize+5,
                             make_nskyaxis=args.show_nsky,
                             make_Rout_proj=args.show_xaxis,
                             make_xaxis=args.show_xaxis,
                             quadrant=qcbar
    )
    
    make_substructures(
        ax, gaps=gaps, rings=rings, twodim=True, label_rings=True,
    )#kwargs_rings={'lw': 0.6})

    if args.show_peaks:
        ax.scatter(lev*cos_peak, lev*sin_peak, edgecolors='none', facecolors=color, alpha=0.2, s=100+150, zorder=20)
        ax.scatter(lev*cos_peak, lev*sin_peak, edgecolors='none', facecolors=color, alpha=1.0, s=10+30, zorder=20)
        ax.scatter(lev*cos_peak, lev*sin_peak, edgecolors='0.3', facecolors='none', alpha=1.0, s=100+150, zorder=20)

    #Show global peak
    sb = 450
    kwargs_global = dict(edgecolors='k', facecolors=color_global_peak, marker='o', lw=2.5, alpha=0.7)
    
    if args.show_global:
        ax.scatter(pick.peak_global_radius*np.cos(np.radians(pick.peak_global_angle)),
                   pick.peak_global_radius*np.sin(np.radians(pick.peak_global_angle)),
                   s=sb+350, zorder=21, **kwargs_global)
               
        ax.scatter(None, None, label='Global peak', s=sb, **kwargs_global) #for legend

    peak_global_rsky, peak_global_PAsky = model.get_sky_offset(pick.peak_global_radius*u.au, pick.peak_global_angle*u.deg, relative_to='disc', midplane=True)        
    print ('Global peak residual found at R=%.1f au, phi=%.1f deg, in disc frame coordinates...'%(pick.peak_global_radius, pick.peak_global_angle))
    print ('Global peak in sky coords,', format_sky_coords(peak_global_rsky, peak_global_PAsky))
    
    if found_clusters and args.clusters:
        
        if len(pick.acc_peaks_phi)>0 and len(pick.acc_peaks_R)>0:
            #Show cluster peak
            kwargs_cluster = dict(edgecolors='k', facecolors=color_cluster_peak, marker='X', lw=2.5, alpha=0.7)
            ax.scatter(pick.acc_R*np.cos(np.radians(pick.acc_phi)),
                       pick.acc_R*np.sin(np.radians(pick.acc_phi)),                       
                       s=sb+350, zorder=21, **kwargs_cluster)

            ax.scatter(None, None, label='Cluster peak', s=sb, **kwargs_cluster) #for legend

            acc_R_rsky,  acc_phi_PAsky = model.get_sky_offset(pick.acc_R*u.au, pick.acc_phi*u.deg, relative_to='disc', midplane=True) 
            print ('Weighted centre of accepted clusters found at R=%.1f au, phi=%.1f deg, in disc frame coordinates...'%(pick.acc_R, pick.acc_phi))
            print ('Cluster in sky coords,', format_sky_coords(acc_R_rsky, acc_phi_PAsky))
            
    #Mark planet location if passed as an arg
    kwargs_planet = dict(edgecolors='k', facecolors='none', marker='o', lw=4.5, alpha=1.0, zorder=22)    

    mark_planet_location(ax, args, s=sb+300, coords='disc', zfunc=z_func, zpars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc, dpc=dpc, **kwargs_planet)    
    
    #mark_planet_location(ax, args, s=sb+200, **kwargs_planet)
    ax.scatter(None, None, label='Planet', s=sb, **kwargs_planet) #for legend
            
    if len(args.rp)>0 and args.show_legend:
        make_1d_legend(ax, handlelength=1.5, loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=args.fontsize)

    if args.show_title:
        ax.set_title('%s, folded map'%ctitle, fontsize=args.fontsize+1, color='k')

    
elif args.projection=='polar': #Currently displaying folded map only
    levels_resid = np.linspace(-clim, clim, 48)

    Rf = np.hypot(pick.X, pick.Y)
    phif = np.arctan2(pick.Y, pick.X)
    
    fig, ax, cbar = make_polar_map(folded_map, levels_resid,
                                   Rf*u.au, phif*u.rad, R_prof[-1]*u.au,
                                   cmap=cmap_res, clabel=unit, fmt=cfmt)
    ax.fill_between([-180, -90], R_prof[-1], color='k', alpha=0.5)
    ax.fill_between([90, 180], R_prof[-1], color='k', alpha=0.5)
    
    make_substructures(
        ax, gaps=gaps, rings=rings, twodim=True, label_rings=True, polar=True
    )

    
if len(args.mask_R)>0 or len(args.mask_phi)>0:
    make_masks(ax, args.mask_R, args.mask_phi, Rmax=Rmod_out)

    
symm_mean = np.nanmean(peak_resid)
symm_std = np.nanstd(peak_resid)
symm = symm_mean + symm_std
print ('Non-axisymmetric factor from peak folded residuals mean, std, (mean+1std): %.3f, %.3f, %.3f km/s'%(symm_mean, symm_std, symm))
    
#*********************
#MAKE FILES AND PLOTS
#*********************
#Write Pick summary file
if args.writetxt:    
    pick.writetxt(filename='pick_summary_%s.txt'%mtags['base'])

#Write interpolated folded_map into file
if args.writefits:
    fits.writeto(os.path.join(mtags['dir_model'], mtags['base']+'_foldedresiduals.fits'), folded_map, header=datacube.header, overwrite=True)    

#Write folded values into file, along with original x,y coordinates:
if args.writetxt:
    tmp = np.asarray([xf, yf, folded_orig]).T
    np.savetxt(os.path.join(mtags['dir_model'], mtags['base']+'_foldedresiduals.txt'), tmp, fmt='%.3f')
    
plt.savefig('folded_residuals_deproj_%s_%s.png'%(mtags['base'], args.projection), bbox_inches='tight', dpi=200)
show_output(args)
