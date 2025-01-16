from discminer.mining_control import _mining_moment_residuals
from discminer.mining_utils import (get_2d_plot_decorators,
                                    get_noise_mask,
                                    load_moments,
                                    load_disc_grid,
                                    show_output)

from discminer.core import Data
from discminer.rail import Contours
from discminer.plottools import (make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 use_discminer_style)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import copy
import json

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_moment_residuals(None)
    args = parser.parse_args()

#**************************
#JSON AND SOME DEFINITIONS
#**************************    
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']
rings = custom['rings']
vsys = best['velocity']['vsys']
Rout = best['intensity']['Rout']

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment)

if args.moment=='velocity':
    cmap_mom = copy.copy(cmap_mom)
    cmap_mom.set_under('1')
    cmap_mom.set_over('1')
    
    cmap_res = copy.copy(cmap_res)
    cmap_res.set_under('1')
    cmap_res.set_over('1')
    
#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']

dpc = meta['dpc']*u.pc
Rmax = 1.1*args.Router*Rout*u.au #Max model radius, 10% larger than disc Rout

au_to_m = u.au.to('m')

#********************
#LOAD DATA AND GRID
#********************
datacube = Data(file_data, dpc) # Read data and convert to Cube object
#noise_mean, mask = get_noise_mask(datacube)

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.1*np.min([xmax, Rmax.value])
extent= np.array([-xmax, xmax, -xmax, xmax])

#*************************
#LOAD DISC GEOMETRY
R, phi, z = load_disc_grid()
noise_mean, mask = get_noise_mask(datacube, thres=4,
                                  mask_phi={'map2d': np.degrees(phi['upper']),
                                            'lims': args.mask_phi},
                                  mask_R={'map2d': R['upper']/au_to_m,
                                          'lims': args.mask_R}
)

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

for i,axi in enumerate(ax):
    make_up_ax(axi, xlims=(-xlim, xlim), ylims=(-xlim, xlim), labelsize=11)
    mod_major_ticks(axi, axis='both', nbins=8)
    axi.set_aspect(1)

    if i>=1:
        axi.tick_params(labelleft=False)
        
    Contours.emission_surface(axi, R, phi, extent=extent, R_lev=np.arange(25, 0.98*Rout, 50)*u.au.to('m'), which='both')
    r"""
    #Overlay dust rings?
    Contours.emission_surface(
        axi, R, phi, extent=extent,
        R_lev=np.array(rings)*u.au.to('m'), which='upper',
        kwargs_R={'linestyles': '-', 'linewidths': 1.2, 'colors': 'k'},
        kwargs_phi={'colors': 'none'}
    )
    #"""
    
datacube.plot_beam(ax[0], fc='0.8')

plt.savefig('moment+residuals_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)
show_output(args)
