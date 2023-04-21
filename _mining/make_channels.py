from discminer.core import Data
from discminer.cube import Cube
from discminer.disc2d import General2d
from discminer.plottools import use_discminer_style
import discminer.cart as cart

from utils import get_noise_mask

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import json
import copy
import sys

use_discminer_style()

with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

chan_step = custom['chan_step']
nchans = custom['nchans']

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.3*best['intensity']['Rout']*u.au #Max model radius, 30% larger than disc Rout

#*********
#LOAD DATA
#*********
datacube = Data(file_data, dpc) # Read data and convert to Cube object
vchannels = datacube.vchannels

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = General2d(datacube, Rmax, Rmin=0, prototype=True)

model.z_upper_func = cart.z_upper_exp_tapered
model.z_lower_func = cart.z_lower_exp_tapered
model.velocity_func = model.keplerian_vertical # vrot = sqrt(GM/r**3)*R
model.line_profile = model.line_profile_bell
model.line_uplow = model.line_uplow_mask

if 'I2pwl' in meta['kind']:
    model.intensity_func = cart.intensity_powerlaw_rbreak
elif 'I2pwlnosurf' in meta['kind']:
    model.intensity_func = cart.intensity_powerlaw_rbreak_nosurf    
else:
    model.intensity_func = cart.intensity_powerlaw_rout

#Useful definitions for plots
xmax = model.skygrid['xmax'] 
xlim = 1.15*xmax/au_to_m
extent= np.array([-xmax, xmax, -xmax, xmax])/au_to_m
  
#**************
#PROTOTYPE PARS
#**************
model.params = copy.copy(best)
model.params['intensity']['I0'] /= meta['downsamp_factor']

#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
if meta['mol']=='12co': modelcube = model.make_model(make_convolve=False) #Returns model cube and computes disc coordinates projected on the sky
else: modelcube = model.make_model(make_convolve=True) 
modelcube.filename = 'cube_model_%s.fits'%tag
modelcube.convert_to_tb(writefits=True)

datacube.filename = 'cube_data_%s.fits'%tag
datacube.convert_to_tb(writefits=True)

#**********************
#VISUALISE CHANNEL MAPS
#**********************
modelcube.show(compare_cubes=[datacube], extent=extent, int_unit='Intensity [K]', show_beam=True, surface_from=model)
modelcube.show_side_by_side(datacube, extent=extent, int_unit='Intensity [K]', show_beam=True,  surface_from=model)

#*****************
#PLOT CHANNEL MAPS
#*****************    
idlim = int(0.5*chan_step*(nchans-1))
plot_channels = np.linspace(-idlim,idlim,nchans) + np.argmin(np.abs(vchannels-best['velocity']['vsys']))  #Channel ids to be plotted, in steps of 6 chans, selected around ~vsys channel

#DATA CHANNELS
fig, ax, im, cbar = datacube.make_channel_maps(channels={'indices': plot_channels}, ncols=5, contours_from=model)
plt.savefig('channel_maps_data.png', bbox_inches = 'tight', dpi=200)
plt.close()

#MODEL CHANNELS
fig, ax, im, cbar = modelcube.make_channel_maps(channels={'indices': plot_channels}, ncols=5, levels=im[0].levels, contours_from=model)
plt.savefig('channel_maps_model.png', bbox_inches = 'tight', dpi=200)
plt.close()

#RESIDUAL CHANNELS
noise_mean, mask = get_noise_mask(datacube)

residualscube = Cube(datacube.data-modelcube.data, datacube.header, datacube.vchannels, dpc, beam=datacube.beam)
residualscube.filename = 'cube_residuals_%s.fits'%tag
residualscube.writefits() 

fig, ax, im, cbar = residualscube.make_channel_maps(channels={'indices': plot_channels}, ncols=5,
                                                    kind='residuals',
                                                    contours_from=model,
                                                    projection=None, 
                                                    unit_intensity='Kelvin',
                                                    unit_coordinates='au',
                                                    xlims = (-xlim, xlim),
                                                    ylims = (-xlim, xlim),
                                                    mask_under=3*noise_mean,
                                                    levels=np.linspace(-29, 29, 32))


#SHOW DISC EMISSION SURFACE AND MAIN AXES
ic, jc = (np.asarray(np.shape(ax))/2).astype(int)
model.make_emission_surface(
    ax[ic][jc],
    kwargs_R={'colors': '0.4', 'linewidths': 0.4},
    kwargs_phi={'colors': '0.4', 'linewidths': 0.3}
)

model.make_disc_axes(ax[ic][jc])

"""
for axi in ax[0]:
    model.make_emission_surface(
        axi,
        kwargs_R={'colors': '0.4', 'linewidths': 0.4},
        kwargs_phi={'colors': '0.4', 'linewidths': 0.3}
    )
    
for axi in ax[1]:
    model.make_disc_axes(axi)
"""    

plt.savefig('channel_maps_residuals.png', bbox_inches = 'tight', dpi=200)
plt.close()

