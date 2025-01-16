from discminer.mining_control import _mining_attributes
from discminer.mining_utils import init_data_and_model, show_output
from discminer.plottools import use_discminer_style, make_up_ax, make_1d_legend

import json
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_attributes(None)
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

file_data = meta['file_data']
tag = meta['tag']

Rmax = 1.1*Rout*u.au #Max model radius

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model()
model.make_model()

#**************
#MAKE PLOT
#**************
fig = plt.figure(figsize=(16,6.5))
ax0 = fig.add_axes([0.08,0.1,0.4,0.4])
ax1 = fig.add_axes([0.08,0.5,0.4,0.4])
ax2 = fig.add_axes([0.52,0.5,0.4,0.4])
ax3 = fig.add_axes([0.52,0.1,0.4,0.4])

#UPPER SURFACE
R_profile = np.linspace(0*u.au, Rmax, 100)
coords = {'R': R_profile}

linewidth_upper = model.get_attribute_map(coords, 'linewidth', surface='upper') #Computes coords{'z'} if not found.
intensity_upper = model.get_attribute_map(coords, 'intensity', surface='upper')
velocity_upper = model.get_attribute_map(coords, 'velocity', surface='upper') * model.params['velocity']['vel_sign']
z_upper = coords['z']*u.m.to('au')

kwargs_plot = dict(lw=3.5, c='tomato', label='Upper surface')
ax0.plot(R_profile, z_upper, **kwargs_plot)
ax1.plot(R_profile, velocity_upper, **kwargs_plot)
ax2.plot(R_profile, 1e3*intensity_upper, **kwargs_plot)
ax3.plot(R_profile, linewidth_upper, **kwargs_plot)

ax0.set_xlabel('Radius [au]')
ax0.set_ylabel('Elevation [au]', fontsize=13)

ax1.set_ylabel('Rot. Velocity [km/s]', fontsize=13)
ax2.set_ylabel('Peak Intensity [mJy/pix]', fontsize=13, labelpad=25, rotation=-90)
ax2.yaxis.set_label_position('right')

ax3.set_xlabel('Radius [au]')
ax3.set_ylabel('Linewidth [km/s]', fontsize=13, labelpad=25, rotation=-90)
ax3.yaxis.set_label_position('right')

for axi in [ax0, ax1, ax2, ax3]:
    axi.yaxis.set_major_formatter(FormatStrFormatter('%4.1f'))
    
#LOWER SURFACE
coords = {'R': R_profile}    
linewidth_lower = model.get_attribute_map(coords, 'linewidth', surface='lower')
intensity_lower = model.get_attribute_map(coords, 'intensity', surface='lower')
velocity_lower = model.get_attribute_map(coords, 'velocity', surface='lower') * model.params['velocity']['vel_sign']
z_lower = -coords['z']*u.m.to('au')

kwargs_plot = dict(lw=3.5, ls=':', c='dodgerblue', label='Lower surface')
ax0.plot(R_profile, z_lower, **kwargs_plot)
ax1.plot(R_profile, velocity_lower, **kwargs_plot)
ax2.plot(R_profile, 1e3*intensity_lower, **kwargs_plot)
ax3.plot(R_profile, linewidth_lower, **kwargs_plot)

#PLOT DECORATIONS
xlims = ax0.get_xlim()
make_up_ax(ax0, xlims=xlims, ylims=(0.0, 1.1*np.nanmax(z_upper)), labelbottom=True, labeltop=False, labelsize=12)
make_up_ax(ax1, xlims=xlims, labelbottom=False, labeltop=False, labelsize=12)
make_up_ax(ax2, xlims=xlims, labelbottom=False, labeltop=False, labelleft=False, labelright=True, labelsize=12)
make_up_ax(ax3, xlims=xlims, ylims=(0.1, 1.3), labelbottom=True, labeltop=False, labelleft=False, labelright=True, labelsize=12)

make_1d_legend(ax1, fontsize=13, loc='lower left', bbox_to_anchor=(0.0, 1.0))    
#ax0.legend(frameon=False)
fig.suptitle('Model Attributes', fontsize=18)

plt.savefig('model_attributes.png', bbox_inches='tight', dpi=200)
show_output(args)
