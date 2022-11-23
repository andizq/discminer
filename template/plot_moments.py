from discminer.core import Data
from discminer.plottools import get_attribute_cmap, make_up_ax, mod_major_ticks, use_discminer_style, mod_nticks_cbars
from discminer.rail import Contours

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import fits
import json

use_discminer_style()
#****************
#SOME DEFINITIONS
#****************
file_data = 'MWC_480_CO_220GHz.robust_0.5.JvMcorr.image.pbcor_clipped_downsamp_2pix_convtb.fits'
dpc = 162*u.pc
Rdisc = 700*u.au
vsys = 5.099463 #from best-fit model

#********************
#LOAD DATA AND GRID
#********************
datacube = Data(file_data, dpc) # Read data and convert to Cube object
noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0) #Noise from line-free channels
mask = np.max(datacube.data, axis=0) < 4*np.mean(noise) 

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.15*xmax
extent= np.array([-xmax, xmax, -xmax, xmax])

#*************************
#LOAD DISC GEOMETRY
R = dict(
    upper=np.load('upper_R.npy'),
    lower=np.load('lower_R.npy')
)

phi = dict(
    upper=np.load('upper_phi.npy'),
    lower=np.load('lower_phi.npy')
)

#*************************
#LOAD MOMENT MAPS
centroid_data = fits.getdata('velocity_data.fits')
centroid_model = fits.getdata('velocity_model.fits') 

#**************************
#MASK AND COMPUTE RESIDUALS
centroid_data = np.where(mask, np.nan, centroid_data)
centroid_model = np.where(mask, np.nan, centroid_model)
centroid_residuals = centroid_data - centroid_model

#**************************
#MAKE PLOT
#fig, ax, ax_cbar0, ax_cbar2 = make_fig_ax_intro()

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15,6))
ax_cbar0 = fig.add_axes([0.15, 0.14, 0.450, 0.04])
ax_cbar2 = fig.add_axes([0.68, 0.14, 0.212, 0.04])
    
cmap_vel = get_attribute_cmap('velocity')
kwargs_im = dict(cmap=cmap_vel, extent=extent, levels=np.linspace(-4.0, 4.0, 48)+vsys)
kwargs_cc = dict(colors='k', linestyles='-', extent=extent, levels=np.linspace(-2, 2, 9)+vsys, linewidths=0.4)
kwargs_cbar = dict(orientation='horizontal', pad=0.03, shrink=0.95, aspect=15)


im0 = ax[0].contourf(centroid_data, extend='both', **kwargs_im)
im1 = ax[1].contourf(centroid_model, extend='both', **kwargs_im)
im2 = ax[2].contourf(centroid_residuals, cmap=cmap_vel, origin='lower', extend='both', extent=extent, levels=np.linspace(-0.2, 0.2, 32))

cc0 = ax[0].contour(centroid_data, **kwargs_cc)
cc1 = ax[1].contour(centroid_model, **kwargs_cc)

cbar0 = plt.colorbar(im0, cax=ax_cbar0, format='%.1f', **kwargs_cbar)
cbar0.ax.tick_params(labelsize=12) 
cbar2 = plt.colorbar(im2, cax=ax_cbar2, format='%.1f', **kwargs_cbar)
cbar2.ax.tick_params(labelsize=12) 

mod_nticks_cbars([cbar0], nbins=8)
mod_nticks_cbars([cbar2], nbins=5)

#for level in kwargs_cc['levels']: cbar0.ax.axvline(level, color=kwargs_cc['colors'], lw=3.0, zorder=1)

ax[0].set_ylabel('Offset [au]', fontsize=15)
ax[0].set_title('MWC 480, $^{12}$CO', pad=40, fontsize=17)
ax[1].set_title('Discminer Model', pad=40, fontsize=17)
ax[2].set_title('Residuals', pad=40, fontsize=17)

cbar0.set_label(r'Centroid Velocity [km s$^{-1}$]', fontsize=14)#, **kwargs_cbar_label)
cbar2.set_label(r'Residuals [km s$^{-1}$]', fontsize=14)#, **kwargs_cbar_label)

#mod_nticks_cbars([cbar0], nbins=8)
#mod_nticks_cbars([cbar2], nbins=5)
#for cbar in [cbar0,cbar2]:
#    cbar.ax.tick_params(which='major', direction='in', width=2.3, size=4.5, pad=7, labelsize=MEDIUM_SIZE-2)
#    cbar.ax.tick_params(which='minor', direction='in', width=2.3, size=3.2)
#    mod_minor_ticks(cbar.ax)

for axi in ax:
    make_up_ax(axi, xlims=(-xlim, xlim), ylims=(-xlim, xlim), labelsize=11)
    mod_major_ticks(axi, axis='both', nbins=8)
    axi.set_aspect(1)
    #plot_beam(axi, Rbeam=Rbeam)
    
for axi in ax[1:]: axi.tick_params(labelleft=False)

for i,axi in enumerate(ax):
    #if i==2: continue
    #if i==2: make_contour_substructures(axi)

    Contours.emission_surface(axi, R, phi, extent=extent, R_lev=np.linspace(0.1, 0.97, 10)*Rdisc.to('m').value,
                              kwargs_R=dict(linestyles=':', linewidths=0.4), kwargs_phi=dict(linestyles=':', linewidths=0.4))
    
plt.savefig('centroid.png', bbox_inches='tight', dpi=200)
#plt.show()
plt.close()
