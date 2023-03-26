from discminer.core import Data
from discminer.plottools import get_discminer_cmap, make_up_ax, mod_major_ticks, use_discminer_style, mod_nticks_cbars
from discminer.rail import Rail, Contours
from discminer.disc2d import General2d

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from astropy import units as u
from astropy.io import fits
import json

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

noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0)
mask = np.max(datacube.data, axis=0) < 4*np.mean(noise)

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

model.make_model()
#*************************
#LOAD MOMENT MAPS
centroid_data = fits.getdata('velocity_gaussian_data_%s.fits'%tag)
centroid_model = fits.getdata('velocity_gaussian_model_%s.fits'%tag) 

#**************************
#MASK AND COMPUTE RESIDUALS
centroid_data = np.abs(np.where(mask, np.nan, centroid_data) - vsys)
centroid_model = np.abs(np.where(mask, np.nan, centroid_model) - vsys)

#**************************
#MAKE PLOT

beam_au = datacube.beam_size.to('au').value
R_prof = np.arange(1*beam_au, 0.8*Rmax.to('au').value, beam_au/4) #

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,4))

#DATA CURVE
rail_phi = Rail(model, centroid_data, R_prof)
vel_phi, vel_phi_error = rail_phi.get_average()
div_factor = (2*np.sin(np.abs(incl)))/np.pi

vel_phi /= div_factor
vel_phi_error /= div_factor 

ysav_phi = savgol_filter(vel_phi, 5, 3)
ax.plot(R_prof, ysav_phi, c='dodgerblue', lw=3.5, label=r'Data curve', zorder=12)
ax.fill_between(R_prof, vel_phi+vel_phi_error, vel_phi-vel_phi_error, color='dodgerblue', alpha=0.15, zorder=9)

#MODEL CURVE
rail_phi = Rail(model, centroid_model, R_prof)
vel_phi, vel_phi_error = rail_phi.get_average()
div_factor = (2*np.sin(np.abs(incl)))/np.pi

vel_phi /= div_factor
vel_phi_error /= div_factor 

ysav_phi = savgol_filter(vel_phi, 5, 3)
ax.plot(R_prof, ysav_phi, c='tomato', lw=3, label=r'Model curve', zorder=11)
#ax.fill_between(R_prof, vel_phi+vel_phi_error, vel_phi-vel_phi_error, color='tomato', alpha=0.15, zorder=9)

#PERFECT KEPLERIAN
coords = {'R': R_prof*au_to_m}
velocity_upper = model.get_attribute_map(coords, 'velocity', surface='upper') * model.params['velocity']['vel_sign']
ax.plot(R_prof, velocity_upper, c='k', lw=2, ls='--', label=r'Keplerian (%.2f Msun)'%Mstar)

#DECORATIONS
ax.axhline(0, lw=2, ls='--', color='0.7')
Contours.make_substructures(ax, gaps=[76, 149], rings=[98, 165], kinks=[245])

ax.set_xlabel('Radius [au]')
ax.set_ylabel(r'Rotation velocity [km/s]')
ax.set_ylim(1.0, 8.0)
mod_major_ticks(ax, axis='x', nbins=10)
ax.legend(frameon=False, fontsize=12)

plt.savefig('rotation_curve.png', bbox_inches='tight', dpi=200)
plt.show()


