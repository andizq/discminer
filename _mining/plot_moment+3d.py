from discminer.core import Data
from discminer.rail import Contours
from discminer.plottools import (make_up_ax,
                                 mod_major_ticks,
                                 mod_minor_ticks,
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

parser = ArgumentParser(prog='plot moment or residual maps in 3D', description='')
parser.add_argument('-t', '--type', default='residuals', type=str,
                    choices=['data', 'model', 'residuals'],
                    help="Visualise main observables in 3D. DEFAULTS to 'residuals'")
parser.add_argument('-d', '--density', default=1, type=int,
                    help="Total points shown: ntot/args.density. DEFAULTS to 1")
args = add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, smooth=True)
     
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

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment)

if args.type=='residuals':
    clim0 = -clim
    clim1 = clim
else:    
    clim0 = levels_im[0]    
    clim1 = levels_im[-1]
    
#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']

dpc = meta['dpc']*u.pc
Rmax = 1.1*Rout*u.au #Max model radius, 10% larger than disc Rout
au_to_m = u.au.to('m')

#********************
#LOAD DATA AND GRID
#********************
datacube = Data(file_data, dpc) # Read data and convert to Cube object
noise_mean, mask = get_noise_mask(datacube)

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.15*np.min([xmax, Rmax.value])
extent= np.array([-xmax, xmax, -xmax, xmax])

#*************************
#LOAD DISC GEOMETRY
R, phi, z = load_disc_grid()

#*************************
#LOAD MOMENT MAPS    
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)

cmap = cmap_mom
if args.type=='residuals':
    map2d = residuals
    cmap = cmap_res
elif args.type=='data':
    
    map2d = moment_data
elif args.type=='model':
    map2d = moment_model

#**************************
#MAKE PLOT
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')

xu = (R['upper']*np.cos(phi['upper'])).flatten()/au_to_m
yu = (R['upper']*np.sin(phi['upper'])).flatten()/au_to_m
zu = z['upper'].flatten()/au_to_m

xl = (R['lower']*np.cos(phi['lower'])).flatten()/au_to_m
yl = (R['lower']*np.sin(phi['lower'])).flatten()/au_to_m
zl = z['lower'].flatten()/au_to_m

di = args.density

ax.scatter3D(xu[::di], yu[::di], zu[::di], c=map2d.flatten()[::di], s=2.5, cmap=cmap, vmin=clim0, vmax=clim1)
ax.set_box_aspect([4,4,0.7])
ax.set_title(ctitle + r' $-$ ' + clabel + r' $-$ ' + args.type.capitalize(), pad=5, fontsize=15)
plt.show()
#plt.savefig('moment+residuals_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)


