from discminer.mining_control import _mining_residuals_all
from discminer.mining_utils import (init_data_and_model,
                                    get_2d_plot_decorators,
                                    get_noise_mask,
                                    load_moments,
                                    load_disc_grid,
                                    mark_planet_location,
                                    show_output)

from discminer.rail import Contours
from discminer.plottools import (add_cbar_ax,
                                 make_substructures,
                                 _make_radial_grid_2D,
                                 make_up_ax,
                                 mod_minor_ticks,
                                 mod_major_ticks,
                                 use_discminer_style)

import json
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_residuals_all(None)
    args = parser.parse_args()

if args.coords=='disk':
    args.coords = 'disc'
    
#**************************
#JSON AND SOME DEFINITIONS
#**************************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

Rout = best['intensity']['Rout']
incl = best['orientation']['incl']
PA = best['orientation']['PA']
xc = best['orientation']['xc']
yc = best['orientation']['yc']

gaps = custom['gaps']
rings = custom['rings']

file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.1*args.Router*Rout*u.au

#*******************
#LOAD DATA AND GRID
#*******************
#datacube = Data(file_data, dpc) # Read data and convert to Cube object
datacube, model = init_data_and_model(Rmin=0, Rmax=Rmax)
noise_mean, mask = get_noise_mask(datacube)

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.05*np.min([xmax, Rmax.value]) 
extent= np.array([-xmax, xmax, -xmax, xmax])

R, phi, z = load_disc_grid()

R_nonan_au = np.nan_to_num(R[args.surface])/au_to_m
z_nonan_au = np.nan_to_num(z[args.surface])/au_to_m

Xproj = R_nonan_au*np.cos(phi[args.surface])
Yproj = R_nonan_au*np.sin(phi[args.surface])

nh = int(0.5*datacube.nx)

#***********************************
#LOAD MOMENT MAPS AND GET RESIDUALS
#***********************************
moments_data = {}
moments_model = {}
residuals = {}
mtags = {}
mtypes = ['linewidth', 'velocity', 'peakintensity']

for moment in mtypes:
    if args.coords=='sky':
        md, mm, rr, mt = load_moments(args, moment=moment, mask=mask)
    elif args.coords=='disc':
        md, mm, rr, mt = load_moments(
            args,
            moment=moment,
            mask=mask,
            clip_Rmin=args.Rinner*datacube.beam_size,
            clip_Rmax=args.Router*Rout*u.au,
            clip_Rgrid=R[args.surface]*u.m
        )        
    mtags[moment] = mt
    moments_data[moment] = md 
    moments_model[moment] = mm 
    residuals[moment] = rr
         
#*****************
#HELPER FUNCTIONS
#*****************
def decorate_ax_res_2D(ax, cbar, lim=xlim):
    ax.set_aspect(1)
    make_up_ax(ax, xlims=(-lim,lim), ylims=(-lim,lim), labelsize=13)
    ax.set_xlabel('Offset [au]')
    ax.xaxis.set_label_position('top')    
    cbar.ax.tick_params(which='major', direction='in', width=2.7, size=4.8, pad=4, labelsize=12)
    cbar.ax.tick_params(which='minor', direction='in', width=2.3, size=3.3)
    mod_minor_ticks(cbar.ax)

#***************
#PLOT RESIDUALS
#***************    
fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(17,6))
cbar_axes = [add_cbar_ax(fig, axi, perc=5, pad=0) for axi in ax]

clabels = {
    'linewidth': r'$\Delta$ Linewidth [km s$^{-1}$]',
    'velocity': r'$\Delta$ Centroid Velocity [km s$^{-1}$]',
    'peakintensity': r'$\Delta$ Peak Intensity [K]'
}
fill_angs_2pi = np.linspace(0, 2*np.pi, 100)

for i, (axi, moment) in enumerate(zip(ax, mtypes)):
    ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(moment)
    levels_resid = np.linspace(-clim, clim, 32)
    levels_cbar = np.linspace(-clim, clim, 5)

    chot = '0.5'
    cmap_mom.set_under(chot)
    cmap_mom.set_over(chot)
    
    cmap_res.set_under(chot)
    cmap_res.set_over(chot)
    
    kwargs_im = dict(cmap=cmap_res, levels=levels_resid, extend='both', origin='lower')
    kwargs_cbar = dict(orientation='horizontal', format=cfmt, ticks=levels_cbar, pad=0.03, shrink=0.95, aspect=15)    
    
    if args.coords=='sky':
        im = axi.contourf(residuals[moment], extent=extent, **kwargs_im)
        Contours.emission_surface(axi, R, phi, extent=extent,
                                  R_lev=np.linspace(0.1, 1.0, 10)*Rout*au_to_m,
                                  which=mtags['velocity']['surf']
        )
        Contours.disc_axes(axi,
                           R[args.surface][nh]/au_to_m,
                           z[args.surface][nh]/au_to_m,
                           incl, PA, xc=xc, yc=yc)

        Contours.emission_surface(
            axi, R, phi, extent=extent,
            R_lev=np.array(rings)*u.au.to('m'), which='upper',
            kwargs_R={'linestyles': '-', 'linewidths': 1.2, 'colors': 'k'},
            kwargs_phi={'colors': 'none'}
        )
        
    elif args.coords=='disc':
        im = axi.contourf(Xproj, Yproj, residuals[moment], **kwargs_im)
        #make_substructures(axi, gaps=gaps, rings=rings, twodim=True)
        if Rout>700: lf = 4
        else: lf = 2
        _make_radial_grid_2D(axi, args.Router*Rout, gaps=gaps, make_labels=True, label_freq=lf)
        
    cbar = plt.colorbar(im, cax=cbar_axes[i], **kwargs_cbar)
    cbar.set_label(clabels[moment], fontsize=14)

    decorate_ax_res_2D(axi, cbar)
    datacube.plot_beam(axi, fc='lime')

ax[0].set_ylabel('Offset [au]')

tag_base = mtags['velocity']['base'].split('velocity_')[-1]

#*************
#MARK PLANETS
#*************
kwargs_sc = dict(s=150, lw=2.0, edgecolors='k')
for axi in ax:
    mark_planet_location(axi, args, dpc=dpc, coords='sky', zfunc=model.z_upper_func, zpars=best['height_upper'], **best['orientation'], **kwargs_sc)

plt.savefig('residuals_all_%s_%sframe.png'%(tag_base, args.coords), bbox_inches='tight', dpi=200)
show_output(args)
