from discminer.mining_control import _mining_isovelocities
from discminer.mining_utils import (get_2d_plot_decorators,
                                    get_noise_mask,
                                    load_moments,
                                    load_disc_grid,
                                    overlay_continuum,
                                    overlay_filaments,
                                    overlay_spirals,
                                    mark_planet_location,
                                    show_output)

from discminer.core import Data
import discminer.cart as cart
from discminer.plottools import (make_round_map,
                                 make_polar_map,
                                 make_substructures,
                                 make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 use_discminer_style,
                                 get_cmap_from_color,
                                 get_continuous_cmap)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import copy
import json

import warnings

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_isovelocities(None)
    args = parser.parse_args()

spids = np.asarray(args.spiral_ids)
spirals_pos = spids[spids>0]
spirals_neg = spids[spids<0]

#**************************
#JSON AND SOME DEFINITIONS
#**************************
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
try:
    kinks = custom['kinks']
except KeyError:
    kinks = []
    
ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, unit_simple=True, fmt_vertical=True)

chan_step = custom['chan_step']
nchans = custom['nchans']

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
tag_figure = args.projection

#********************
#LOAD DATA AND GRID
#********************
datacube = Data(file_data, dpc) # Read data and convert to Cube object

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

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
    
xmax = grid['xsky'] 
xlim = 1.15*np.min([xmax, Rmax.value])
extent= np.array([-xmax, xmax, -xmax, xmax])

#****************************
#LOAD DISC GEOMETRY AND MASK
R, phi, z = load_disc_grid()

noise_mean, mask = get_noise_mask(datacube, thres=2,
                                  mask_phi={'map2d': np.degrees(phi['upper']),
                                            'lims': args.mask_phi},
                                  mask_R={'map2d': R['upper']/au_to_m,
                                          'lims': args.mask_R}
)

Xproj = R[args.surface]*np.cos(phi[args.surface])
Yproj = R[args.surface]*np.sin(phi[args.surface])

#*************************
#LOAD AND CLIP MOMENT MAPS    
moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask,
    clip_Rmin=0.0*u.au,
    clip_Rmax=Rmod_out*u.au,
    clip_Rgrid=R[args.surface]*u.m
)

#********************
#SURFACE PROPERTIES
if args.surface in ['up', 'upper']:
    if 'surf2pwl' in meta['kind']:
        z_func = cart.z_upper_powerlaw
    elif 'surfirregular' in meta['kind']:
        z_func = cart.z_upper_irregular
    else:
        z_func = cart.z_upper_exp_tapered
    z_pars = best['height_upper']
        
elif args.surface in ['low', 'lower']:
    if 'surf2pwl' in meta['kind']:
        z_func = cart.z_lower_powerlaw
    elif 'surfirregular' in meta['kind']:
        z_func = cart.z_lower_irregular            
    else:
        z_func = cart.z_lower_exp_tapered
    z_pars = best['height_lower']

#***********
#MAKE PLOT
clabels = {
    'linewidth': r'$\Delta$ Line width [km s$^{-1}$]',
    'lineslope': r'$\Delta$ Line slope',
    'velocity': r'$\Delta$ Centroid [km s$^{-1}$]',
    'peakintensity': r'$\Delta$ Peak Int. [K]'
}

idlim = int(0.5*chan_step*(nchans-1))
plot_channels = np.linspace(-idlim,idlim,nchans) + np.argmin(np.abs(datacube.vchannels-best['velocity']['vsys']))
levels = np.sort(datacube.vchannels[plot_channels.astype(int)])

if args.projection=='cartesian':
    #levels_resid = np.linspace(-clim, clim, 32)
        
    fig, ax = make_round_map(moment_data, levels, Xproj*u.m, Yproj*u.m, Rmod_out*u.au,
                             z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                             cmap=cmap_res, clabel=unit, fmt=cfmt, 
                             gaps=gaps, rings=rings, kinks=kinks,
                             make_contourf=False, make_contour=True,
                             make_radial_grid=False, make_azimuthal_grid=True,
                             make_cbar=False,
                             mask_inner=args.Rinner*datacube.beam_size)

    X = Xproj/au_to_m
    Y = Yproj/au_to_m
    cmodel = ax.contour(X, Y, moment_model, levels=levels, colors='k', linestyles='--', linewidths=0.7)

    #SHOW FILAMENTS?
    if args.show_filaments:
        overlay_filaments(ax, Xproj/au_to_m, Yproj/au_to_m,
                          residuals, projection=args.projection, writefits=False)

    #OVERLAY FIT SPIRALS?
    if len(args.spiral_ids)>0:
        overlay_spirals(ax, args, mtags, Rmin=0, Rmax=Rmod_out)
        tag_figure += '_' + args.spiral_type + '_spirals_' + args.spiral_moment
            
    #SHOW CONTINUUM?        
    if args.show_continuum in ['all', 'scattered']:
        try:
            import cmasher as cmr
            cmap = plt.get_cmap('cmr.watermelon')
            overlay_continuum(ax, parfile='parfile_scattered.json', cmap=cmap, vmax=0.8, surface=args.surface, extend='max')
        except FileNotFoundError as e:
            warnings.warn('Unable to load parfile for scattered light image...', Warning)

    if args.show_continuum in ['all', 'band7']:
        try:
            cmap = get_cmap_from_color('orange', lev=32, vmin=0.0, vmax=0.7)
            overlay_continuum(ax, parfile='parfile_band7.json', lev=5, contours=True, cmap=cmap, surface=args.surface, zorder=20)
        except FileNotFoundError as e:
            warnings.warn('Unable to load parfile for band7 image...', Warning)

    if args.show_continuum=='none':
        make_substructures(ax, gaps=gaps, rings=rings, kinks=kinks, twodim=True, label_rings=True, label_kinks=True)
    else:
        make_substructures(ax, rings=rings, kinks=kinks, twodim=True, label_rings=True, kwargs_rings={'color': 'k', 'alpha': 0})

        
elif args.projection=='polar':
    #levels_resid = np.linspace(-clim, clim, 48)
    fig, ax, cbar = make_polar_map(moment_data, levels,
                                   R[args.surface]*u.m, phi[args.surface]*u.rad, Rmod_out*u.au,
                                   Rin=args.Rinner*datacube.beam_size,
                                   make_contourf=False, make_contour=True,                                   
                                   cmap=cmap_res, fmt=cfmt, clabel=clabels[args.moment])

    fig, ax, cbar = make_polar_map(moment_model, levels,
                                   R[args.surface]*u.m, phi[args.surface]*u.rad, Rmod_out*u.au,
                                   fig=fig, ax=ax,                                   
                                   Rin=args.Rinner*datacube.beam_size,
                                   make_contourf=False, make_contour=True,
                                   kwargs_contour={'linestyles': '--'},
                                   cmap=cmap_res, fmt=cfmt, clabel=clabels[args.moment])
    
    if args.show_filaments:
        overlay_filaments(ax, np.degrees(phi[args.surface]), R[args.surface]/au_to_m,
                          residuals, projection=args.projection, writefits=False)
        
    make_substructures(ax, gaps=gaps, rings=rings, twodim=True, polar=True, label_rings=True)

mark_planet_location(ax, args, edgecolor='limegreen', lw=4.5, s=650, alpha=0.8, coords='disc', zfunc=z_func, zpars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc, dpc=dpc)        
ax.set_title(ctitle, fontsize=16, color='k')

plt.savefig('isovelocities_deproj_%s_%s.png'%(mtags['base'], tag_figure), bbox_inches='tight', dpi=200)
show_output(args)
