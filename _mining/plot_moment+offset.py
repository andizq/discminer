from discminer.core import Data
from discminer.rail import Contours
from discminer.plottools import (make_up_ax,
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
from matplotlib.patches import Rectangle
from astropy import units as u

import json
from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='plot moment maps', description='Plot moment map from data and zoom-in around central region')
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

rings = custom['rings']
gaps = custom['gaps']
kinks = []

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
#MAKE PLOT + ZOOM-IN PANEL
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,8))
ax_cbar0 = fig.add_axes([0.13, 0.09, 0.77, 0.05])

kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=levels_im)
kwargs_cc = dict(colors='k', linestyles='-', extent=extent, levels=levels_cc, linewidths=0.4)
kwargs_cbar = dict(orientation='horizontal', pad=0.03, shrink=0.95, aspect=15)
zoomcolor = '0.3'
zoomwidth = 90
        
def make_plot(ax, xlim=xlim, color='k', labelcolor='k'):
    im = ax.contourf(moment_data, extend='both', **kwargs_im)
    if args.surface!='lower':
        cc = ax.contour(moment_data, **kwargs_cc)
    make_up_ax(axi, xlims=(-xlim, xlim), ylims=(-xlim, xlim), labelsize=17, color=color, labelcolor=labelcolor)
    return im

for i,axi in enumerate(ax):
    if i==1: im = make_plot(axi, xlim=zoomwidth, color=zoomcolor, labelcolor=zoomcolor)
    else: im = make_plot(axi, xlim=xlim)
    axi.scatter(best['orientation']['xc'], best['orientation']['yc'],
                ec='k', fc='w', marker='X', lw=0.5+i, s=60*(i+1), zorder=20)        
        
    mod_major_ticks(axi, axis='both', nbins=8)
    datacube.plot_beam(axi, fc='lime')
    axi.set_aspect(1)
    #model.disc_axes(axi)
    
for axi in ax[1:]:
    axi.tick_params(labelleft=False)
    for side in ['top','bottom','left','right']:
        axi.spines[side].set_linewidth(4.0)
        axi.spines[side].set_color(zoomcolor)
        axi.spines[side].set_linestyle((0, (1,1.5)))
        axi.spines[side].set_capstyle('round')
    axi.grid(color='k', ls='--')
    
for i,axi in enumerate(ax):
    Contours.emission_surface(axi, R, phi, extent=extent, R_lev=np.linspace(0.1, 1.0, 10)*Rout*u.au.to('m'), which=mtags['surf'])
        
patch = Rectangle([-zoomwidth]*2, 2*zoomwidth, 2*zoomwidth, edgecolor=zoomcolor, facecolor='none',
                  lw=2.0, ls=(0, (1,1.5)), capstyle='round')
ax[0].add_artist(patch)

cbar0 = plt.colorbar(im, cax=ax_cbar0, format='%.1f', **kwargs_cbar)
cbar0.ax.tick_params(labelsize=15) 
mod_nticks_cbars([cbar0], nbins=10)
cbar0.set_label(clabel, fontsize=16)
ax[0].set_ylabel('Offset [au]', fontsize=17)
ax[0].set_title(ctitle, pad=40, fontsize=19)
ax[1].set_title('Zoom-in', pad=40, fontsize=19, color=zoomcolor)

plt.savefig('moment+offset_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)
plt.show()
plt.close()
