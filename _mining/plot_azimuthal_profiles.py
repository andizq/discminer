from discminer.core import Data
from discminer.plottools import (get_discminer_cmap,
                                 make_up_ax, mod_major_ticks,
                                 use_discminer_style, mod_nticks_cbars)
from discminer.rail import Rail, Contours
from discminer.disc2d import General2d
import discminer.cart as cart

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits

import json
import copy
import sys

from argparse import ArgumentParser
from utils import get_1d_plot_decorators

use_discminer_style()

parser = ArgumentParser(prog='plot azimuthal contours', description='Plot azimuthal contours from a given moment map [velocity, linewidth, [peakintensity, peakint]?')
parser.add_argument('-m', '--moment', default='velocity', type=str, choices=['velocity', 'linewidth', 'lineslope', 'peakint', 'peakintensity'], help="velocity, linewidth or peakintensity")
parser.add_argument('-k', '--kind', default='residuals', type=str, choices=['data', 'model', 'residuals'], help="data, model or residuals")
args = parser.parse_args()

if args.moment=='peakint':
     args.moment = 'peakintensity'

#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

vsys = best['velocity']['vsys']
incl = best['orientation']['incl']

clabel, clabel_res, clim0, clim0_res, clim1, clim1_res, unit = get_1d_plot_decorators(args.moment, tag=args.kind)

if args.kind=='residuals':
    clim0 = clim0_res
    clim1 = clim1_res
    clabel = clabel_res

if args.moment=='velocity':
    if args.kind=='residuals':
        clabel = r'Velocity %s %s'%(args.kind, unit)
    else:
        clabel = r'Deprojected Velocity %s %s'%(args.kind, unit)        
    
#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rout = best['intensity']['Rout']
Rmax = 1.1*Rout*u.au #Max model radius, 10% larger than disc Rout

#**********
#LOAD DATA
#**********
datacube = Data(file_data, dpc) # Read data and convert to Cube object
noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0)
mask = np.max(datacube.data, axis=0) < 4*np.mean(noise)
vchannels = datacube.vchannels

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = General2d(datacube, Rmax, Rmin=0, prototype=True)

model.z_upper_func = cart.z_upper_exp_tapered
model.z_lower_func = cart.z_lower_exp_tapered
model.velocity_func = model.keplerian_vertical
model.line_profile = model.line_profile_bell

if 'I2pwl' in meta['kind']:
    model.intensity_func = cart.intensity_powerlaw_rbreak
elif 'I2pwlnosurf' in meta['kind']:
    model.intensity_func = cart.intensity_powerlaw_rbreak_nosurf    
else:
    model.intensity_func = cart.intensity_powerlaw_rout
  
#**************
#PROTOTYPE PARS
#**************
model.params = copy.copy(best)
model.params['intensity']['I0'] /= meta['downsamp_factor']

#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
model.make_model() #Make model, just needed to load disc geometry

#*************************
#LOAD MOMENT MAPS
moment_data = fits.getdata('%s_gaussian_data.fits'%args.moment)
moment_model = fits.getdata('%s_gaussian_model.fits'%args.moment) 

#**************************
#MASK AND COMPUTE RESIDUALS
moment_data = np.where(mask, np.nan, moment_data)
moment_model = np.where(mask, np.nan, moment_model)
moment_residuals = moment_data - moment_model

if args.kind=='residuals': map2d = moment_residuals
elif args.kind=='data': map2d = moment_data
elif args.kind=='model': map2d = moment_model

if args.kind!='residuals' and args.moment=='velocity': #deproject velocity field
    map2d = (map2d-vsys)/(np.cos(model.projected_coords['phi']['upper'])*np.sin(incl))
    
#**************************
#MAKE PLOT

beam_au = datacube.beam_size.to('au').value
R_prof = np.arange(2*beam_au, 0.8*Rout, beam_au/4)

color_bounds = np.array([0.33, 0.66, 1.0])*Rout

rail = Rail(model, map2d, R_prof)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,6))
ax2 = fig.add_axes([0.85,0.6,0.3*6/14,0.3])

R_ref = 100
R_list, phi_list, resid_list, color_list = rail.prop_along_coords(coord_ref=R_ref,
                                                                  color_bounds=color_bounds,
                                                                  ax=ax, ax2=ax2)

tick_angles = np.arange(-150, 180, 30)
ax.set_xticks(tick_angles)
ax.set_xlabel(r'Azimuth [deg]')
ax.set_ylabel(clabel)
ax.set_xlim(-180,180)
ax.set_ylim(clim0, clim1)
ax.grid()

model.make_emission_surface(ax2)
model.make_disc_axes(ax2)
make_up_ax(ax, labeltop=False)
make_up_ax(ax2, labelbottom=False, labelleft=False, labeltop=True)
ax.tick_params(labelbottom=True, top=True, right=True, which='both', direction='in')

plt.savefig('azimuthal_%s_%s.png'%(args.moment, args.kind), bbox_inches='tight', dpi=200)
plt.show()
plt.close()
