from discminer.mining_control import _mining_channels
from discminer.mining_utils import get_noise_mask, init_data_and_model, show_output

from discminer.cube import Cube
from discminer.plottools import use_discminer_style

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import json
import copy
import sys

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_channels(None)
    args = parser.parse_args()

#**************************
#JSON AND SOME DEFINITIONS
#**************************    
with open('parfile.json') as json_file:
    pars = json.load(json_file)
    
meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

chan_step = custom['chan_step']
nchans = custom['nchans']

file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model(Rmin=0, Rmax=1.2)
vchannels = datacube.vchannels
pix_downsamp = model.grid['step']*meta['downsamp_fit']/au_to_m

#Useful definitions for plots
xmax = model.skygrid['xmax'] 
xlim = 1.0*xmax/au_to_m
extent= np.array([-xmax, xmax, -xmax, xmax])/au_to_m
  
#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
#Return model cube and compute disc coordinates projected on the sky
if args.make_beam==-1:
    if meta['mol'] in ['12co', '13co'] and pix_downsamp>1.01*model.beam_size.value:
        modelcube = model.make_model(make_convolve=False) 
    else:
        modelcube = model.make_model(make_convolve=True)
else:
    modelcube = model.make_model(make_convolve=True*args.make_beam)     

modelcube.filename = 'cube_model_%s.fits'%tag
modelcube.writefits() #Jy/bm
modelcube.convert_to_tb(writefits=True, planck=args.planck) #K

datacube.filename = 'cube_data_%s.fits'%tag
datacube.writefits()
datacube.convert_to_tb(writefits=True, planck=args.planck)

#**********************
#VISUALISE CHANNEL MAPS
#**********************
if args.show_output:
    modelcube.show(compare_cubes=[datacube], extent=extent, int_unit='Intensity [K]', show_beam=True, surface_from=model)
    modelcube.show_side_by_side(datacube, extent=extent, int_unit='Intensity [K]', show_beam=True,  surface_from=model)

#*****************
#PLOT CHANNEL MAPS
#*****************    
idlim = int(0.5*chan_step*(nchans-1))
plot_channels = np.linspace(-idlim,idlim,nchans) + np.argmin(np.abs(vchannels-best['velocity']['vsys']))  #Channel ids to be plotted, selected around ~vsys channel

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
                                                    levels=np.linspace(-custom['Ilim'], custom['Ilim'], 32))


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

