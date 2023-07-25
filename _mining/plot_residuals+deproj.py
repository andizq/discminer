from discminer.core import Data
import discminer.cart as cart
from discminer.plottools import (make_round_map,
                                 make_polar_map,
                                 make_substructures,
                                 make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 use_discminer_style)

from utils import (make_and_save_filaments,
                   init_data_and_model,
                   get_2d_plot_decorators,
                   get_noise_mask,
                   load_moments,
                   load_disc_grid,
                   add_parser_args)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import json
from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='plot residual maps', description='Plot residual maps')

parser.add_argument('-sppos', '--spirals_pos', nargs='*', default=[], type=int, help="Positive spiral ids to overlay fitted curve and save fit parameters into txt file. FORMAT: -sp 0 2 3")
parser.add_argument('-spneg', '--spirals_neg', nargs='*', default=[], type=int, help="Negative spiral ids to overlay fitted curve and save fit parameters into txt file. FORMAT: -sn 0 2 3")
parser.add_argument('-sptype', '--spiral_type', default='linear', choices=['linear', 'log'], help="Type of spiral fit to be shown and saved into file.")
parser.add_argument('-spmom', '--spiral_moment', default='velocity', choices=['velocity', 'linewidth', 'peakintensity'], help="Moment map utilised to extract and fit the spirals")

parser.add_argument('-f', '--filaments', default=0, type=int, choices=[0, 1], help="Make filaments")

args = add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, projection=True, Rinner=True, Router=0.95, smooth=True)

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

tag_figure = args.projection

#********************
#LOAD DATA AND GRID
#********************
datacube = Data(file_data, dpc) # Read data and convert to Cube object
noise_mean, mask = get_noise_mask(datacube, thres=2)

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.15*np.min([xmax, Rmax.value])
extent= np.array([-xmax, xmax, -xmax, xmax])

#*************************
#LOAD DISC GEOMETRY
R, phi, z = load_disc_grid()

Xproj = R[args.surface]*np.cos(phi[args.surface])
Yproj = R[args.surface]*np.sin(phi[args.surface])

#*************************
#LOAD AND CLIP MOMENT MAPS    
moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask,
    clip_Rmin=0.0*u.au,
    clip_Rmax=args.Router*Rout*u.au,
    clip_Rgrid=R[args.surface]*u.m
)

#***********
#MAKE PLOTS
clabels = {
    'linewidth': r'$\Delta$ Line width [km s$^{-1}$]',
    'lineslope': r'$\Delta$ Line slope',
    'velocity': r'$\Delta$ Centroid [km s$^{-1}$]',
    'peakintensity': r'$\Delta$ Peak Int. [K]'
}

#SPIRAL PRESCRIPTIONS
sp_lin = lambda x, a, b: a + b*x
sp_log = lambda x, a, b: a*np.exp(b*x)

if args.projection=='cartesian':
    levels_resid = np.linspace(-clim, clim, 32)
    
    if args.surface in ['up', 'upper']:
        z_func = cart.z_upper_exp_tapered
        z_pars = best['height_upper']

    elif args.surface in ['low', 'lower']:
        z_func = cart.z_lower_exp_tapered
        z_pars = best['height_lower']
    
    fig, ax = make_round_map(residuals, levels_resid, Xproj*u.m, Yproj*u.m, args.Router*Rout*u.au,
                             z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                             cmap=cmap_res, clabel=unit, fmt=cfmt, 
                             gaps=gaps, rings=rings,
                             mask_inner=args.Rinner*datacube.beam_size)
    
    make_substructures(ax, gaps=gaps, rings=rings, twodim=True, label_rings=True)
    
    if len(args.spirals_pos)+len(args.spirals_neg)>0:
        arr_read = np.loadtxt(
            'spirals_fit_parameters_%s_%s.txt'%(mtags['base'].replace(args.moment, args.spiral_moment), args.spiral_type),
            dtype = {
                'names': ('a', 'b', 'color', 'sign', 'id'),
                'formats': (float, float, 'U10', 'U10', int)
            },
            skiprows = 1,
            comments = None,
        )

        arr_read = np.atleast_1d(arr_read)
        phi_ext = 2*np.linspace(-360, 360, 500)
        phi_ext_rad = np.radians(phi_ext)

        if args.spiral_type == 'linear':
            sp_func = sp_lin
        else:
            sp_func = sp_log

        for arr in arr_read:
            if (
                    (arr[3]=='pos' and arr[4] in args.spirals_pos)
                    or
                    (arr[3]=='neg' and arr[4] in args.spirals_neg)
            ):                
                R_ext = sp_func(phi_ext_rad, *tuple(arr)[:2])
                ind = (R_ext > 0) & (R_ext < args.Router*Rout)
                xplot = R_ext[ind]*np.cos(phi_ext_rad[ind])
                yplot = R_ext[ind]*np.sin(phi_ext_rad[ind])            
                ax.plot(xplot, yplot, lw=4,  color=arr[2], zorder=20)
                ax.plot(xplot, yplot, lw=7, color='k', zorder=19)

        tag_figure += '_' + args.spiral_type + '_spirals_' + args.spiral_moment
            
    if args.filaments:
        _, model = init_data_and_model()
        model.make_model()
        fil_pos_obj, fil_neg_obj, _, _, _ = make_and_save_filaments(model, residuals, tag=mtags['base']+'_'+args.projection, return_all=True)        

        ax.contour(Xproj/au_to_m, Yproj/au_to_m, fil_pos_obj.skeleton, linewidths=0.2, colors='darkred', alpha=1, zorder=11)
        ax.contour(Xproj/au_to_m, Yproj/au_to_m, fil_neg_obj.skeleton, linewidths=0.2, colors='navy', alpha=1, zorder=11)
    
elif args.projection=='polar':
    levels_resid = np.linspace(-clim, clim, 48)    #levels_im #
    fig, ax, cbar = make_polar_map(residuals, levels_resid,
                                   R[args.surface]*u.m, phi[args.surface]*u.rad, args.Router*Rout*u.au,
                                   Rin=args.Rinner*datacube.beam_size,
                                   cmap=cmap_res, fmt=cfmt, clabel=clabels[args.moment])
                                   
    make_substructures(ax, gaps=gaps, rings=rings, twodim=True, polar=True, label_rings=True)

ax.set_title(ctitle, fontsize=16, color='k')

plt.savefig('residuals_deproj_%s_%s.png'%(mtags['base'], tag_figure), bbox_inches='tight', dpi=200)
plt.show()
plt.close()
