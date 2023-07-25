from discminer.core import Data
from discminer.rail import Contours
from discminer.plottools import (make_up_ax,
                                 add_cbar_ax,
                                 mod_major_ticks,
                                 mod_minor_ticks,
                                 mod_nticks_cbars,
                                 use_discminer_style)

from utils import (get_2d_plot_decorators,
                   init_data_and_model,
                   get_noise_mask,
                   load_moments,
                   load_disc_grid,
                   add_parser_args)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits

import json
from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='Comparison between moments from different kernels', description='')
parser.add_argument('-t', '--component', default='double', type=str, choices=['single', 'double'], help="Single or double component moment")
args = add_parser_args(parser, moment=True, surface=True, kind=True, Rinner=True, Router=True, smooth=0.5)
     
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

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, fmt_vertical=True)

chot = '0.5'
cmap_mom.set_under(chot)
cmap_mom.set_over(chot)

cmap_res.set_under(chot)
cmap_res.set_over(chot)

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
datacube, model = init_data_and_model(Rmin=0, Rmax=1.1)
noise_mean, mask = get_noise_mask(datacube, thres=3)

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = np.min([xmax, Rmax.value])*1.0
extent= np.array([-xmax, xmax, -xmax, xmax])

#model.make_model()
#*************************
#LOAD DISC GEOMETRY
R, phi, z = load_disc_grid()

#**************************
#MAKE PLOT
nrows=3
if args.component=='double':    
    ncols=4
    kernels = ['quadratic', 'gaussian', 'dgauss', 'dbell']
elif args.component=='single':
    ncols=3
    kernels = ['quadratic', 'gaussian', 'bell']

figx, figy = 2.5*ncols+1, nrows*2+1
    
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figx, figy))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.10, hspace=0.0)

kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=levels_im)
kwargs_cc = dict(colors='k', linestyles='-', extent=extent, levels=levels_cc, linewidths=0.4)

for i,kernel in enumerate(kernels):
    
    try:
        moment_data, moment_model, residuals, mtags = load_moments(
            args,
            kernel=kernel,
            mask=mask,
            clip_Rmin=args.Rinner*datacube.beam_size,
            clip_Rmax=args.Router*Rout*u.au,
            clip_Rgrid=R[args.surface]*u.m
        )
    except FileNotFoundError:
        continue

    im0 = ax[0,i].contourf(moment_data, extend='both', **kwargs_im)
    im1 = ax[1,i].contourf(moment_model, extend='both', **kwargs_im)
    im2 = ax[2,i].contourf(residuals, cmap=cmap_res, origin='lower', extend='both', extent=extent, levels=np.linspace(-1.05*clim, 1.05*clim, 32))

    cc0 = ax[0,i].contour(moment_data, **kwargs_cc)
    cc1 = ax[1,i].contour(moment_model, **kwargs_cc)

    ax[0,i].set_title(kernel.capitalize(), pad=15, color='0.7', fontsize=15)

ax[0,0].set_ylabel(ctitle, fontsize=14)    
ax[1,0].set_ylabel('Model', fontsize=14)
ax[2,0].set_ylabel('Residuals', fontsize=14)

#*************
#COLORBARS
#*************
kwargs_addax = dict(perc=9, pad=0.0, orientation='vertical') 
ax_cbar0 = add_cbar_ax(fig, ax[0,-1], **kwargs_addax)
ax_cbar1 = add_cbar_ax(fig, ax[1,-1], **kwargs_addax)
ax_cbar2 = add_cbar_ax(fig, ax[2,-1], **kwargs_addax)

axp0 = ax_cbar0.get_position()
axp1 = ax_cbar1.get_position()
axp2 = ax_cbar2.get_position()

ax_cbar01 = fig.add_axes([axp1.x0, axp1.y0, axp1.width, axp0.y1-axp1.y0])
ax_cbar0.remove()
ax_cbar1.remove()

kwargs_cbar = dict(orientation='vertical')

cbar1 = plt.colorbar(im1, cax=ax_cbar01, format=cfmt, **kwargs_cbar)
cbar1.ax.tick_params(labelsize=8) 
cbar2 = plt.colorbar(im2, cax=ax_cbar2, format=cfmt, **kwargs_cbar)
cbar2.ax.tick_params(labelsize=8) 

mod_nticks_cbars([cbar1], nbins=10)
mod_nticks_cbars([cbar2], nbins=5)

cbar1.set_label(clabel, fontsize=9)
cbar2.set_label(r'%s'%unit, fontsize=9)

#****************
#AXES DECORATIONS
#****************
kw_tickpars = dict(labelleft=False, labeltop=False, left=False, right=False, bottom=False, top=False)

for i,axrow in enumerate(ax):
    for axi in axrow:
        make_up_ax(axi, xlims=(-xlim, xlim), ylims=(-xlim, xlim))
        mod_major_ticks(axi, axis='both', nbins=8)
        axi.set_aspect(1)
        axi.tick_params(**kw_tickpars)
        #if i==nrows-1: axi.set_xlabel('Offset [au]', fontsize=11)
        
        #model.make_emission_surface(
        Contours.emission_surface(
            axi,
            R, phi, extent=extent,
            R_lev=np.linspace(0.1, 0.97, 10)*Rout*u.au.to('m'),
            which='both',
            kwargs_R={'colors': '0.1', 'linewidths': 0.1, 'linestyles': '-'},
            kwargs_phi={'colors': '0.1', 'linewidths': 0.05, 'linestyles': '-'}
        )
        #model.make_disc_axes(axi)        

for axi in ax[0]:
    datacube.plot_beam(axi, fc='lime')
    
plt.savefig('moment+comparison_%s.png'%args.moment, dpi=200, bbox_inches='tight')
plt.show()
plt.close()
