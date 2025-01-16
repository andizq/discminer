from discminer.mining_control import _mining_channels_peakint
from discminer.mining_utils import (
    get_noise_mask,
    get_2d_plot_decorators,
    init_data_and_model,
    load_moments,
    show_output
)

from discminer.cube import Cube
from discminer.plottools import (
    use_discminer_style,
    make_up_ax,
    get_continuous_cmap
)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import json

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_channels_peakint(None)
    args = parser.parse_args()

#**************************
#JSON AND SOME DEFINITIONS
#**************************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']
Rout = best['intensity']['Rout']

nchans = args.nchans
chan_step = args.step

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment)

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

try:
    mol_tex_meta = mol_tex[meta['mol']]
except KeyError:
    mol_tex_meta = meta['mol']

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model(Rmin=0, Rmax=1.0)
noise_mean, mask = get_noise_mask(datacube, thres=args.sigma)
    
vchannels = datacube.vchannels
pix_downsamp = model.grid['step']*meta['downsamp_fit']/au_to_m

#Useful definitions for plots
xmax = model.skygrid['xmax'] 
"""
if meta['mol'] in ['13co', 'cs']:
    xlim = 0.8*xmax/au_to_m
else:
    xlim = xmax/au_to_m
"""
xlim = args.Router*Rout    
extent= np.array([-xmax, xmax, -xmax, xmax])/au_to_m

#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
modelcube = model.make_model(make_convolve=True)
modelcube.convert_to_tb(writefits=False)
datacube.convert_to_tb(writefits=False)

#*************************
#LOAD MOMENT MAPS
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)

#*****************
#PLOT CHANNEL MAPS
#*****************
noise_mean, mask = get_noise_mask(datacube, thres=args.sigma) #again, but in Tb units
print ('Mean RMS: %.2e K'%noise_mean)

cmap = None
"""
if 'sum' in meta['kind']:
    import cmasher as cmr
    cmap = cmap_mom = plt.get_cmap('cmr.rainforest_r')
    cmap = plt.get_cmap('cmr.rainforest_r')
"""
idlim = int(0.5*chan_step*(nchans-1))
plot_channels = np.linspace(-idlim,idlim,nchans) + np.argmin(np.abs(vchannels-best['velocity']['vsys'])) 
plot_channels = np.append([0], plot_channels)

kw_channels = dict(channels={'indices': plot_channels}, ncols=nchans+1,
                   xlims = (-xlim, xlim), ylims = (-xlim, xlim), show_beam=False,
                   observable = 'intensity_2',
                   #contours_from=model,
                   #kwargs_contour = dict(colors='w', linewidths=1.0), 
                   unit_coordinates='au', unit_intensity=None, projection=None)
                   
fig, ax = plt.subplots(nrows=3, ncols=nchans+1, figsize=(2.5*(nchans+1), 2.5*3)) #Master figure
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)

#DATA CHANNELS
levels=np.linspace(0*args.sigma*noise_mean, 3.5*custom['Ilim'], 32)

fig, ax0, im0, cbar0 = datacube.make_channel_maps(fig=fig, ax=ax[0,:], levels=levels, extend='max', **kw_channels)
fig, ax1, im1, cbar1 = modelcube.make_channel_maps(fig=fig, ax=ax[1,:], levels=im0[0].levels, annotate_channels=False, cmap=cmap, extend='max', **kw_channels)
cbar0.remove()

#RESIDUAL CHANNELS
kw_channels.update(dict(kwargs_contour = {}, contours_from=model))

residualscube = Cube(datacube.data-modelcube.data, datacube.header, datacube.vchannels, dpc, beam=datacube.beam)

fig, ax2, im2, cbar2 = residualscube.make_channel_maps(fig=fig, ax=ax[2,:],
                                                       kind='residuals',
                                                       annotate_channels=False,
                                                       extend = 'both',
                                                       mask_under=args.sigma*noise_mean,
                                                       levels=np.linspace(-custom['Ilim'], custom['Ilim'], 32),
                                                       #levels=12*np.linspace(-noise_mean, noise_mean, 32),
                                                       **kw_channels)


for axi in ax[:-1,0]:
    axi.set_xlabel('')    
    axi.tick_params(labelleft=False, labelbottom=False)

ax[0,0].set_title('Peak Intensity', color='0.6', fontsize=13, pad=8, loc='center')
ax[0,0].set_ylabel(mol_tex_meta + ' Data', color='k', fontsize=13, labelpad=6)
ax[1,0].set_ylabel('Model', color='k', fontsize=13, labelpad=4)

#PLOT MOMENT MAP
for axi in ax[:,0]:
    axi.set_xlim(-xlim, xlim)
    axi.set_ylim(-xlim, xlim)
    for artist in axi.get_lines() + axi.collections + axi.texts:
        artist.remove()

datacube.plot_beam(ax[0,0], fc='gold')        
        
kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=im0[0].levels[::1])

pm0 = ax[0,0].contourf(moment_data, extend='both', **kwargs_im)
pm1 = ax[1,0].contourf(moment_model, extend='both', **kwargs_im)
pm2 = ax[2,0].contourf(residuals, cmap=cmap_res, origin='lower', extend='both', extent=extent, levels=np.linspace(-clim, clim, 32))

#for axi in ax[:,int(round(0.5*(nchans+1)))-1]:
for axi in ax[:,0]:
    model.make_emission_surface(
        axi,
        kwargs_R={'colors': '0.1', 'linewidths': 0.5},
        kwargs_phi={'colors': '0.1', 'linewidths': 0.4}
    )
    model.make_disc_axes(axi)
    for spine in ['left',  'top', 'right', 'bottom']:
        axi.spines[spine].set_color('0.8')        
    axi.tick_params(which='both', color='0.8')
    #axi.axis('off')

plt.savefig('channel_maps_peakint_%s_%s.png'%(meta['disc'], meta['mol']), bbox_inches = 'tight', dpi=200)    
show_output(args)
