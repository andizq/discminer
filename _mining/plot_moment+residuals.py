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

parser = ArgumentParser(prog='plot moment maps', description='Plot moment map [velocity, linewidth, [peakintensity, peakint]?')
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

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']

dpc = meta['dpc']*u.pc
Rmax = 1.1*Rout*u.au #Max model radius, 10% larger than disc Rout

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
        
#**************************
#MAKE PLOT

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15,6))
ax_cbar0 = fig.add_axes([0.15, 0.14, 0.450, 0.04])
ax_cbar2 = fig.add_axes([0.68, 0.14, 0.212, 0.04])
    
kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=levels_im)
kwargs_cc = dict(colors='k', linestyles='-', extent=extent, levels=levels_cc, linewidths=0.4)
kwargs_cbar = dict(orientation='horizontal', pad=0.03, shrink=0.95, aspect=15)

im0 = ax[0].contourf(moment_data, extend='both', **kwargs_im)
im1 = ax[1].contourf(moment_model, extend='both', **kwargs_im)
im2 = ax[2].contourf(residuals, cmap=cmap_res, origin='lower', extend='both', extent=extent, levels=np.linspace(-1.01*clim, 1.01*clim, 32))

cc0 = ax[0].contour(moment_data, **kwargs_cc)
cc1 = ax[1].contour(moment_model, **kwargs_cc)

cbar0 = plt.colorbar(im0, cax=ax_cbar0, format='%.1f', **kwargs_cbar)
cbar0.ax.tick_params(labelsize=12) 
cbar2 = plt.colorbar(im2, cax=ax_cbar2, format=cfmt, **kwargs_cbar)
cbar2.ax.tick_params(labelsize=12) 

mod_nticks_cbars([cbar0], nbins=10)
mod_nticks_cbars([cbar2], nbins=5)

ax[0].set_ylabel('Offset [au]', fontsize=15)
ax[0].set_title(ctitle, pad=40, fontsize=17)
ax[1].set_title('Discminer Model', pad=40, fontsize=17)
ax[2].set_title('Residuals', pad=40, fontsize=17)

cbar0.set_label(clabel, fontsize=14)
cbar2.set_label(r'Residuals %s'%unit, fontsize=14)

for axi in ax:
    #axi.scatter(best['orientation']['xc'], best['orientation']['yc'], c='k', marker='X', lw=0.5, s=30) #use +offset.py script
    make_up_ax(axi, xlims=(-xlim, xlim), ylims=(-xlim, xlim), labelsize=11)
    mod_major_ticks(axi, axis='both', nbins=8)
    axi.set_aspect(1)
    
for axi in ax[1:]: axi.tick_params(labelleft=False)

for i,axi in enumerate(ax):
    Contours.emission_surface(axi, R, phi, extent=extent, R_lev=np.linspace(0.1, 1.0, 10)*Rout*u.au.to('m'), which=mtags['surf'])
                              
datacube.plot_beam(ax[0], fc='0.8')

#####
#Planet candidate (Wagner+2023)
if meta['disc']=='mwc758':
    Rp = 0.617*dpc.value
    PAp = np.radians(225)
    phi = PAp + np.pi/2

    for axi in ax:
        axi.scatter(Rp*np.cos(phi), Rp*np.sin(phi), marker='X', s=80, fc='lime', ec='k', lw=1.5, alpha=0.6)
#####

plt.savefig('moment+residuals_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)
plt.show()
plt.close()
