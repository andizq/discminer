from discminer.core import Data
from discminer.rail import Contours
import discminer.cart as cart
from discminer.plottools import (make_round_map,
                                 make_polar_map,
                                 make_substructures,
                                 make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 use_discminer_style)

from utils import (get_2d_plot_decorators,
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
args = add_parser_args(parser, moment=True, kind=True, surface=True, projection=True)

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
#LOAD MOMENT MAPS    
moment_data, moment_model, mtags = load_moments(args)

#****************
#USEFUL FUNCTIONS
#****************
def clip_prop_radially(prop2d, Rmin=datacube.beam_size, Rmax=np.inf, Rgrid=R[args.surface]*u.m):
    Rmin = Rmin.to('au').value
    Rgrid = np.nan_to_num(Rgrid).to('au').value
    try:
        Rmax = Rmax.to('au').value
    except AttributeError:
        Rmax = Rmax
    return np.where((Rgrid<Rmin) | (Rgrid>Rmax), np.nan, prop2d)

#**************************
#MASK AND COMPUTE RESIDUALS
moment_data = np.where(mask, np.nan, moment_data)
moment_model = np.where(mask, np.nan, moment_model)
residuals = clip_prop_radially(moment_data - moment_model, Rmax=Rout)

clabels = {
    'linewidth': r'$\Delta$ Line width [km s$^{-1}$]',
    'lineslope': r'$\Delta$ Line slope',
    'velocity': r'$\Delta$ Centroid [km s$^{-1}$]',
    'peakintensity': r'$\Delta$ Peak Int. [K]'
}

if args.projection=='cartesian':
    levels_resid = np.linspace(-clim, clim, 32)
    
    if args.surface in ['up', 'upper']:
        z_func = cart.z_upper_exp_tapered
        z_pars = best['height_upper']

    elif args.surface in ['low', 'lower']:
        z_func = cart.z_lower_exp_tapered
        z_pars = best['height_lower']
    
    fig, ax = make_round_map(residuals, levels_resid, Xproj*u.m, Yproj*u.m, Rout*u.au,
                             z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                             cmap=cmap_res, clabel=unit, fmt=cfmt, 
                             gaps=gaps, rings=rings)
    
    make_substructures(ax, gaps=gaps, rings=rings, twodim=True, label_rings=True)

elif args.projection=='polar':
    levels_resid = np.linspace(-clim, clim, 48)    
    fig, ax, cbar = make_polar_map(residuals, levels_resid,
                                   R[args.surface]*u.m, phi[args.surface]*u.rad, Rout*u.au,
                                   Rin = datacube.beam_size,
                                   cmap=cmap_res, fmt=cfmt, clabel=clabels[args.moment])
                                   
    make_substructures(ax, gaps=gaps, rings=rings, twodim=True, polar=True, label_rings=True)
    
ax.set_title(ctitle, fontsize=16, color='k')

plt.savefig('residuals_deproj_%s_%s.png'%(mtags['base'], args.projection), bbox_inches='tight', dpi=200)
plt.show()
plt.close()
