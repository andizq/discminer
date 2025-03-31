from discminer.mining_control import _mining_pick
from discminer.mining_utils import (init_data_and_model,
                                    get_noise_mask,
                                    get_2d_plot_decorators,
                                    load_moments,
                                    load_disc_grid,
                                    mark_planet_location,
                                    make_masks,
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

import json

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_pick(None)
    args = parser.parse_args()

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

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, unit_simple=True, fmt_vertical=True)

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
datacube, model = init_data_and_model()
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
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)
    
if args.moment=='velocity' and args.fold=='absolute':
    residuals = np.abs(moment_data-vsys) - np.abs(moment_model-vsys)
    residuals[mask] = np.nan
    if args.percentage_kepler:
        ref_surf = mtags['ref_surf']
        coords = {'R': model.projected_coords['R']['upper']}
        velocity_kepler = model.get_attribute_map(coords, 'velocity', surface=ref_surf) * vel_sign    
        residuals = residuals/velocity_kepler

#********************
def get_sky_coords(rp, phip, midplane=True):
    phii = np.radians(phip)
    zp = z_func({'R': rp*u.au.to('m')}, **z_pars)*u.m.to('au')
    if midplane:
        zp *= 0
    xi,yi,zi = GridTools.get_sky_from_disc_coords(rp, phii, zp, incl, PA, xc, yc)
    PAsky = np.arctan2(yi, xi) - np.pi/2
    rsky = np.hypot(xi-xc, yi-yc)/dpc.value
    return rsky, np.degrees(PAsky)

#*******************
#FIND PEAK RESIDUALS
#*******************
R_prof = np.arange(Rmod_in, Rmod_out, beam_au/5)
xlim0, xlim1 = 0.5*R_prof[0], 1.05*R_prof[-1]

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(11,5))    
ax_c = fig.add_axes([0.8,0.65,0.23*5/11,0.23]) #AxesSubplot for showing contour colours 

pick = Pick(model, residuals, R_prof, fold=True, color_bounds = np.array([0.33, 0.66, 1.0])*Rmod_out, ax2=ax_c)
folded_map = pick.make_2d_map() #Map where peaks will be picked from

figh, axh = plt.subplots(ncols=1, nrows=1, figsize=(9,6))    
pick.find_peaks(clean_thres=args.clean_thres, fig_ax_histogram=(figh, axh), clean_histogram=True)
figh.savefig('histogram_peak_residuals_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)

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
fig.savefig('peak_residuals_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)
#plt.show()
plt.close()

#sys.exit()

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
        
    fig, ax = make_round_map(folded_map, levels_resid, pick.X*u.au, pick.Y*u.au, R_prof[-1]*u.au,
                             z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                             cmap=cmap_res, clabel=unit, fmt=cfmt,
                             make_cbar=args.colorbar, 
                             rings=rings,                             
                             mask_wedge=(90, 270)*u.deg,
                             mask_inner=R_prof[0]*u.au)
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
                   s=sb+250, zorder=21, **kwargs_global)
               
        ax.scatter(None, None, label='Global peak', s=sb, **kwargs_global) #for legend
    
    print ('Global peak residual found at R=%.1f au, phi=%.1f deg, in disc frame coordinates...'%(pick.peak_global_radius, pick.peak_global_angle))
    print ('Global peak in sky coords, R=%.3f arcsec, PA=%.1f deg...'%get_sky_coords(pick.peak_global_radius, pick.peak_global_angle))
    
    if found_clusters and args.clusters:
        
        if len(pick.acc_peaks_phi)>0 and len(pick.acc_peaks_R)>0:
            #Show cluster peak
            kwargs_cluster = dict(edgecolors='k', facecolors=color_cluster_peak, marker='X', lw=2.5, alpha=0.7)
            ax.scatter(pick.acc_R*np.cos(np.radians(pick.acc_phi)),
                       pick.acc_R*np.sin(np.radians(pick.acc_phi)),                       
                       s=sb+250, zorder=21, **kwargs_cluster)

            ax.scatter(None, None, label='Cluster peak', s=sb, **kwargs_cluster) #for legend
            
            print ('Weighted centre of accepted clusters found at R=%.1f au, phi=%.1f deg, in disc frame coordinates...'%(pick.acc_R, pick.acc_phi))
            print ('Cluster in sky coords, R=%.3f arcsec, PA=%.1f deg...'%get_sky_coords(pick.acc_R, pick.acc_phi))
            
    #Mark planet location if passed as an arg
    kwargs_planet = dict(edgecolors='gold', facecolors='none', marker='o', lw=4.5, alpha=1.0, zorder=22)    

    mark_planet_location(ax, args, edgecolor='k', lw=3.5, s=sb+200, coords='disc', zfunc=z_func, zpars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc, dpc=dpc)    
    
    #mark_planet_location(ax, args, s=sb+200, **kwargs_planet)
    ax.scatter(None, None, label='Planet', s=sb, **kwargs_planet) #for legend
            
    if len(args.rp)>0 and args.show_legend:
        make_1d_legend(ax, handlelength=1.5, loc='lower center', bbox_to_anchor=(0.5, 1.02))
        
    ax.set_title('%s, folded map'%ctitle, fontsize=16, color='k')

    
elif args.projection=='polar':
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

if args.clusters:    
    pick.writetxt(filename='pick_summary_%s.txt'%mtags['base'])

plt.savefig('folded_residuals_deproj_%s_%s.png'%(mtags['base'], args.projection), bbox_inches='tight', dpi=200)
show_output(args)

symm_mean = np.nanmean(peak_resid)
symm_std = np.nanstd(peak_resid)
symm = symm_mean + symm_std
print ('Non-axisymmetric factor from peak folded residuals mean, std, (mean+1std): %.3f, %.3f, %.3f km/s'%(symm_mean, symm_std, symm))
