from discminer.core import Data
from discminer.cube import Cube
from discminer.disc2d import Model
from discminer.plottools import use_discminer_style
import discminer.cart as cart

from utils import add_parser_args, get_noise_mask, get_2d_plot_decorators, init_data_and_model

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import json
import copy
from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='plot channel maps', description='Plot channel maps and residuals')
parser.add_argument('-nc', '--nchans', default=7, type=int, help="Number of channels to be plotted")
args = add_parser_args(parser, sigma=3)

#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

chan_step = custom['chan_step']

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc

mol_tex = {
    '12co': r'$^{12}$CO',
    '13co': r'$^{13}$CO',
    'cs': r'CS'
}
#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model(Rmin=0, Rmax=1.2)
vchannels = datacube.vchannels
pix_downsamp = model.grid['step']*meta['downsamp_fit']/au_to_m

#Useful definitions for plots
xmax = model.skygrid['xmax'] 

if meta['mol'] in ['13co', 'cs']:
    xlim = 0.7*xmax/au_to_m
else:
    xlim = xmax/au_to_m

extent= np.array([-xmax, xmax, -xmax, xmax])/au_to_m
  
#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
modelcube = model.make_model(make_convolve=True)
modelcube.convert_to_tb(writefits=False)
datacube.convert_to_tb(writefits=False)

#*****************
#PLOT CHANNEL MAPS
#*****************
# Import CMasher
cmap = None
if 'sum' in meta['kind']:
    import cmasher as cmr
    cmap = plt.get_cmap('cmr.rainforest_r')

nchans = args.nchans
idlim = int(0.5*chan_step*(nchans-1))
plot_channels = np.linspace(-idlim,idlim,nchans) + np.argmin(np.abs(vchannels-best['velocity']['vsys'])) 

kw_channels = dict(channels={'indices': plot_channels}, ncols=nchans, contours_from=model,
                   xlims = (-xlim, xlim), ylims = (-xlim, xlim), 
                   unit_coordinates='au', unit_intensity=None, projection=None)
                   

fig, ax = plt.subplots(nrows=3, ncols=nchans, figsize=(2.5*nchans, 2.5*3)) #Master figure
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)

#DATA CHANNELS
fig, ax0, im0, cbar0 = datacube.make_channel_maps(fig=fig, ax=ax[0], cmap=cmap, **kw_channels)
fig, ax1, im1, cbar1 = modelcube.make_channel_maps(fig=fig, ax=ax[1], levels=im0[0].levels, annotate_channels=False, cmap=cmap, **kw_channels)


#RESIDUAL CHANNELS
noise_mean, mask = get_noise_mask(datacube)
residualscube = Cube(datacube.data-modelcube.data, datacube.header, datacube.vchannels, dpc, beam=datacube.beam)

fig, ax2, im2, cbar2 = residualscube.make_channel_maps(fig=fig, ax=ax[2],
                                                       kind='residuals',
                                                       annotate_channels=False,
                                                       mask_under=args.sigma*noise_mean,
                                                       levels=np.linspace(-custom['Ilim'], custom['Ilim'], 32),
                                                       #cmap=cmap_res,
                                                       **kw_channels)

cbar0.remove()

for axi in ax[:-1,0]:
    axi.set_xlabel('')    
    axi.tick_params(labelleft=False, labelbottom=False)

#ax[0,0].set_title(meta['disc'].upper(), pad=10, loc='left')
ax[0,0].set_ylabel(mol_tex[meta['mol']] + ' Data', color='k', fontsize=13, labelpad=6)
ax[1,0].set_ylabel('Model', color='k', fontsize=13, labelpad=4)


for axi in ax[:,int(round(0.5*(nchans+1)))-1]:
    model.make_emission_surface(
        axi,
        kwargs_R={'colors': '0.4', 'linewidths': 0.4},
        kwargs_phi={'colors': '0.4', 'linewidths': 0.3}
    )
    
    model.make_disc_axes(axi)

plt.savefig('channel_maps_residuals_%s_%s.png'%(meta['disc'], meta['mol']), bbox_inches = 'tight', dpi=200)    
plt.show()

