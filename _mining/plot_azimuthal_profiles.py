from discminer.rail import Rail, Contours
from discminer.plottools import (get_discminer_cmap,
                                 make_up_ax, mod_major_ticks,
                                 use_discminer_style, mod_nticks_cbars)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits

import json
import copy
import sys

from argparse import ArgumentParser
from utils import init_data_and_model, get_noise_mask, get_1d_plot_decorators

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
Rout = best['intensity']['Rout']

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

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model()

noise_mean, mask = get_noise_mask(datacube, thres=2)
vchannels = datacube.vchannels
model.make_model()

#*************************
#LOAD MOMENT MAPS
moment_data = fits.getdata('%s_gaussian_data.fits'%args.moment)
moment_model = fits.getdata('%s_gaussian_model.fits'%args.moment) 
#moment_data = fits.getdata('%s_up_doublebell_mask_data.fits'%args.moment)
#moment_model = fits.getdata('%s_up_doublebell_mask_model.fits'%args.moment) 

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

color_bounds = np.array([0.5, 1.0])*Rout

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
