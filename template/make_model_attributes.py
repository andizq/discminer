from discminer.core import Data
from discminer.cube import Cube
from discminer.disc2d import General2d
from discminer.rail import Contours
from discminer.plottools import use_discminer_style, make_up_ax

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from astropy import units as u
import sys

use_discminer_style()
#****************
#SOME DEFINITIONS
#****************
file_data = 'MWC_480_CO_220GHz.robust_0.5.JvMcorr.image.pbcor_clipped_downsamp_2pix_convtb.fits'
tag = 'mwc480_12co'
nwalkers = 256
nsteps = 10000
au_to_m = u.au.to('m')

dpc = 162*u.pc
Rmax = 700*u.au

#********
#GRIDDING
#********
downsamp_pro = 2 # Downsampling used for prototype
downsamp_fit = 10 # Downsampling used for MCMC fit
downsamp_factor = (downsamp_fit/downsamp_pro)**2 # Required to correct intensity normalisation for prototype

datacube = Data(file_data, dpc) # Read data and convert to Cube object
vchannels = datacube.vchannels

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = General2d(datacube, Rmax, prototype = True)
# Prototype? If False discminer assumes you'll run an MCMC fit

def intensity_powerlaw_rout(coord, I0=30.0, R0=100, p=-0.4, z0=100, q=0.3, Rout=500):
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    z = coord['z']
    R0*=au_to_m
    z0*=au_to_m
    Rout*=au_to_m
    A = I0*R0**-p*z0**-q
    Ieff = np.where(R<=Rout, A*R**p*np.abs(z)**q, 0.0)
    return Ieff

def z_upper(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

def z_lower(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return -au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

model.z_upper_func = z_upper
model.z_lower_func = z_lower
model.velocity_func = model.keplerian_vertical # vrot = sqrt(GM/r**3)*R
model.line_profile = model.line_profile_bell
model.intensity_func = intensity_powerlaw_rout

#**************
#PROTOTYPE PARS
#**************
best_fit_pars = np.loadtxt('./log_pars_%s_cube_%dwalkers_%dsteps.txt'%(tag, nwalkers, nsteps))[1]
Mstar, vsys, incl, PA, xc, yc, I0, p, q, L0, pL, qL, Ls, pLs, z0_upper, p_upper, Rb_upper, q_upper, z0_lower, p_lower, Rb_lower, q_lower = best_fit_pars

model.params['velocity']['Mstar'] = Mstar
model.params['velocity']['vel_sign'] = -1
model.params['velocity']['vsys'] = vsys
model.params['orientation']['incl'] = incl
model.params['orientation']['PA'] = PA
model.params['orientation']['xc'] = xc
model.params['orientation']['yc'] = yc
model.params['intensity']['I0'] = I0/downsamp_factor #Jy/pix
model.params['intensity']['p'] = p
model.params['intensity']['q'] = q
model.params['intensity']['Rout'] = Rmax.to('au').value
model.params['linewidth']['L0'] = L0 
model.params['linewidth']['p'] = pL
model.params['linewidth']['q'] = qL
model.params['lineslope']['Ls'] = Ls
model.params['lineslope']['p'] = pLs
model.params['height_upper']['z0'] = z0_upper
model.params['height_upper']['p'] = p_upper
model.params['height_upper']['Rb'] = Rb_upper
model.params['height_upper']['q'] = q_upper
model.params['height_lower']['z0'] = z0_lower
model.params['height_lower']['p'] = p_lower
model.params['height_lower']['Rb'] = Rb_lower
model.params['height_lower']['q'] = q_lower


#**************
#MAKE PLOT
#**************
fig = plt.figure(figsize=(14,5))
ax0 = fig.add_axes([0.08,0.1,0.4,0.4])
ax1 = fig.add_axes([0.08,0.5,0.4,0.4])
ax2 = fig.add_axes([0.52,0.5,0.4,0.4])
ax3 = fig.add_axes([0.52,0.1,0.4,0.4])

#UPPER SURFACE
R_profile = np.linspace(datacube.beam_size, Rmax, 100)
coords = {'R': R_profile}

linewidth_upper = model.get_attribute_map(coords, 'linewidth', surface='upper') #Fills in coords{'z'} if not found.
intensity_upper = model.get_attribute_map(coords, 'intensity', surface='upper')
velocity_upper = model.get_attribute_map(coords, 'velocity', surface='upper') * model.params['velocity']['vel_sign']
z_upper = coords['z']*u.m.to('au')

kwargs_plot = dict(lw=3.5, c='tomato')
ax0.plot(R_profile, z_upper, label='Upper surface', **kwargs_plot)
ax1.plot(R_profile, velocity_upper, **kwargs_plot)
ax2.plot(R_profile, intensity_upper, **kwargs_plot)
ax3.plot(R_profile, linewidth_upper, **kwargs_plot)

make_up_ax(ax0, labelbottom=True, labeltop=False, labelsize=11)
make_up_ax(ax1, labelbottom=False, labeltop=False, labelsize=11)
make_up_ax(ax2, labelbottom=False, labeltop=False, labelleft=False, labelright=True, labelsize=11)
make_up_ax(ax3, ylims=(0.1, 1.3), labelbottom=True, labeltop=False, labelleft=False, labelright=True, labelsize=11)

ax0.set_xlabel('Radius [au]')
ax0.set_ylabel('Elevation [au]', fontsize=12)

ax1.set_ylabel('Velocity [km/s]', fontsize=12)
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
