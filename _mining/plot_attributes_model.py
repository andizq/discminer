from discminer.plottools import use_discminer_style, make_up_ax
from utils import init_data_and_model

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']

Rmax = 1.1*best['intensity']['Rout']*u.au #Max model radius, 10% larger than disc Rout

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model()
model.make_model()

#**************
#MAKE PLOT
#**************
fig = plt.figure(figsize=(14,5))
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

kwargs_plot = dict(lw=3.5, c='tomato')
ax0.plot(R_profile, z_upper, label='Upper surface', **kwargs_plot)
ax1.plot(R_profile, velocity_upper, **kwargs_plot)
ax2.plot(R_profile, intensity_upper, **kwargs_plot)
ax3.plot(R_profile, linewidth_upper, **kwargs_plot)

make_up_ax(ax0, ylims=(0.0, 1.1*np.nanmax(z_upper)), labelbottom=True, labeltop=False, labelsize=11)
make_up_ax(ax1, labelbottom=False, labeltop=False, labelsize=11)
make_up_ax(ax2, labelbottom=False, labeltop=False, labelleft=False, labelright=True, labelsize=11)
make_up_ax(ax3, ylims=(0.1, 1.3), labelbottom=True, labeltop=False, labelleft=False, labelright=True, labelsize=11)

ax0.set_xlabel('Radius [au]')
ax0.set_ylabel('Elevation [au]', fontsize=12)

ax1.set_ylabel('Rot. Velocity [km/s]', fontsize=12)
ax2.set_ylabel('Peak Intensity []', fontsize=12, labelpad=25, rotation=-90)
ax2.yaxis.set_label_position('right')

ax3.set_xlabel('Radius [au]')
ax3.set_ylabel('Linewidth [km/s]', fontsize=12, labelpad=25, rotation=-90)
ax3.yaxis.set_label_position('right')

for axi in [ax0, ax1, ax2, ax3]:
    axi.yaxis.set_major_formatter(FormatStrFormatter('%4.1f'))

#LOWER SURFACE
coords = {'R': R_profile}    
linewidth_lower = model.get_attribute_map(coords, 'linewidth', surface='lower')
intensity_lower = model.get_attribute_map(coords, 'intensity', surface='lower')
velocity_lower = model.get_attribute_map(coords, 'velocity', surface='lower') * model.params['velocity']['vel_sign']
z_lower = -coords['z']*u.m.to('au')

kwargs_plot = dict(lw=3.5, ls=':', c='dodgerblue')
ax0.plot(R_profile, z_lower, label='Lower surface', **kwargs_plot)
ax1.plot(R_profile, velocity_lower, **kwargs_plot)
ax2.plot(R_profile, intensity_lower, **kwargs_plot)
ax3.plot(R_profile, linewidth_lower, **kwargs_plot)

ax0.legend(frameon=False)
fig.suptitle('Model Attributes', fontsize=18)

plt.savefig('model_attributes.png', bbox_inches='tight', dpi=200)
plt.show()
