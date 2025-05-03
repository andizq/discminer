from discminer.mining_control import _mining_moment_offset
from discminer.mining_utils import (get_2d_plot_decorators,
                                    get_noise_mask,
                                    load_moments,
                                    load_disc_grid,
                                    init_data_and_model,
                                    mark_planet_location,
                                    overlay_continuum,
                                    show_output)

from discminer.rail import Contours
from discminer.plottools import (make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 get_cmap_from_color,
                                 use_discminer_style)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy import units as u

import json
import warnings

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_moment_offset(None)
    args = parser.parse_args()

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

rings = custom['rings']
gaps = custom['gaps']

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment)
    
file_data = meta['file_data']
tag = meta['tag']

dpc = meta['dpc']*u.pc
Rmax_frac = 1.1 #Max model radius, 10% larger than disc Rout
Rmax = Rmax_frac*Rout*u.au

#********************
#LOAD DATA AND GRID
#********************
#datacube = Data(file_data, dpc) # Read data and convert to Cube object
datacube, model = init_data_and_model(Rmin=0, Rmax=Rmax)
noise_mean, mask = get_noise_mask(datacube, thres=args.sigma)

with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.0*np.min([xmax, Rmax.value])
extent= np.array([-xmax, xmax, -xmax, xmax])

R, phi, z = load_disc_grid()
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)

#**************************
#MAKE PLOT + ZOOM-IN PANEL
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,8))
ax_cbar0 = fig.add_axes([0.13, 0.09, 0.77, 0.05])

kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=levels_im)
kwargs_cc = dict(colors='k', linestyles='-', extent=extent, levels=levels_cc, linewidths=0.4)
kwargs_cbar = dict(orientation='horizontal', pad=0.03, shrink=0.95, aspect=15)
zoomcolor = '0.9'
zoomlabelcolor = '0.3'
zoomwidth = args.zoom_size

def make_plot(ax, xlim=xlim, tickcolor='k', labelcolor='k'):
    if args.moment=='velocity':
        extend='both'
    else:
        extend='max'
    im = ax.contourf(moment_data, extend=extend, **kwargs_im)
    if args.surface!='lower' and args.show_contours:
        cc = ax.contour(moment_data, **kwargs_cc)
    make_up_ax(ax, xlims=(-xlim, xlim), ylims=(-xlim, xlim), labelsize=17, color=tickcolor, labelcolor=labelcolor)
    return im

for i,axi in enumerate(ax):
    if i==1:
        im = make_plot(axi, xlim=zoomwidth, tickcolor=zoomlabelcolor, labelcolor=zoomlabelcolor)
        axi.scatter(best['orientation']['xc'], best['orientation']['yc'],
                    ec='k', fc='w', marker='X', lw=0.5+i, s=60*(i+1), zorder=20)                
    else:
        im = make_plot(axi, xlim=xlim)
        
    mod_major_ticks(axi, axis='both', nbins=8)
    datacube.plot_beam(axi, fc='0.8')
    axi.set_aspect(1)
    #model.disc_axes(axi)
    
for axi in ax[1:]:
    axi.tick_params(labelleft=False)
    for side in ['top','bottom','left','right']:
        axi.spines[side].set_linewidth(4.0)
        axi.spines[side].set_color(zoomlabelcolor)
        axi.spines[side].set_linestyle((0, (1,1.5)))
        axi.spines[side].set_capstyle('round')
    axi.grid(color='k', ls='--')

R_lev = np.arange(25, Rmax_frac*Rout, 50)*u.au.to('m')
surf_color = edge_color = '0.1'

for i,axi in enumerate([ax[0]]):
    Contours.emission_surface(axi, R, phi,
                              extent=extent,
                              R_lev=R_lev,
                              which=mtags['surf'],
                              kwargs_R={'colors': surf_color, 'linewidths': 0.5},
                              kwargs_phi={'colors': surf_color, 'linewidths': 0.4}
    )
    Contours.emission_surface(axi, R, phi,
                              extent=extent,
                              R_lev=[R_lev[-1]],
                              which='upper',
                              kwargs_R={'colors': edge_color, 'linewidths': 1.4, 'linestyles': '-', 'alpha': 0.8},
                              kwargs_phi={'colors': edge_color, 'linewidths': 0., 'linestyles': '-'}
        )

    Contours.emission_surface(axi, R, phi,
                              extent=extent,
                              R_lev=[R_lev[-1]],
                              which='lower',
                              kwargs_R={'colors': edge_color, 'linewidths': 1.1, 'linestyles': '-', 'alpha': 0.3},
                              kwargs_phi={'colors': edge_color, 'linewidths': 0., 'linestyles': '-'}
        )
    

model.make_disc_axes(ax[0], surface=args.surface)
    
patch = Rectangle([-zoomwidth]*2, 2*zoomwidth, 2*zoomwidth, edgecolor=zoomcolor, facecolor='none',
                  lw=2.0, ls=(0, (1,1.5)), capstyle='round')
ax[0].add_artist(patch)

cbar0 = plt.colorbar(im, cax=ax_cbar0, format='%.1f', **kwargs_cbar)
cbar0.ax.tick_params(labelsize=15) 
mod_nticks_cbars([cbar0], nbins=10)
cbar0.set_label(clabel, fontsize=16)
ax[0].set_ylabel('Offset [au]', fontsize=17)
ax[0].set_title(ctitle, pad=40, fontsize=19)
ax[1].set_title('Zoom-in', pad=40, fontsize=19, color=zoomlabelcolor)

#SHOW CONTINUUM?        
if args.show_continuum in ['all', 'scattered']:
    try:
        import cmasher as cmr
        cmap = plt.get_cmap('cmr.watermelon')
        overlay_continuum(ax[1], parfile='parfile_scattered.json', coords='sky', cmap=cmap, vmax=0.8, surface=args.surface, extend='max')
    except FileNotFoundError as e:
        warnings.warn('Unable to load parfile for scattered light image...', Warning)

if args.show_continuum in ['all', 'band7']:
    try:
        cmap = get_cmap_from_color('k', lev=32, vmin=0.0, vmax=0.2)
        overlay_continuum(ax[1], parfile='parfile_band7.json', coords='sky', lev=3, contours=True, cmap=cmap, surface=args.surface, zorder=20)
    except FileNotFoundError as e:
        warnings.warn('Unable to load parfile for band7 image...', Warning)

#*************
#MARK PLANETS
#*************
kwargs_sc = dict(s=700, lw=3.0, edgecolors='tomato')
for axi in ax[1:]:
    mark_planet_location(axi, args, dpc=dpc, coords='sky', zfunc=model.z_upper_func, zpars=best['height_upper'], **best['orientation'], **kwargs_sc)

plt.savefig('moment+offset_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)
show_output(args)
