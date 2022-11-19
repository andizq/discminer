from discminer.core import Data, Model
from discminer.cube import Cube
from discminer.disc2d import General2d, Tools

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u

#****************
#SOME DEFINITIONS
#****************
file_data = 'MWC_480_CO_220GHz.robust_0.5.JvMcorr.image.pbcor_clipped_downsamp_2pix_convtb.fits'
tag = 'mwc480_12co'
nwalkers = 256
nsteps = 10000
au_to_m = u.au.to('m')

dpc = 162*u.pc
Rdisc = 700*u.au

#********
#GRIDDING
#********
downsamp_pro = 2 # Downsampling used for prototype
downsamp_fit = 10 # Downsampling used for MCMC fit
downsamp_factor = (downsamp_fit/downsamp_pro)**2 # Required to correct intensity normalisation for prototype

datacube = Data(file_data) # Read data and convert to Cube object
noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0)

mgrid = Model(datacube, dpc, Rdisc) # Make grid from datacube info
vchannels = datacube.vchannels

#Useful definitions for plots
xmax = mgrid.skygrid['xmax'] 
xlim = 1.15*xmax/au_to_m
extent= np.array([-xmax, xmax, -xmax, xmax])/au_to_m

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = General2d(mgrid.discgrid, # Disc grid
                  skygrid=mgrid.skygrid, # Sky grid
                  prototype = True, # Prototype? If False it assumes you'll run MCMC fit
                  beam=datacube.beam, # Data beam
                  kwargs_beam={'distance': dpc.value} 
)

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

#If not redefined, intensity and linewidth are powerlaws 
 #of R and z by default, whereas lineslope is constant.
  #See Table 1 of discminer paper 1.

#**************
#PROTOTYPE PARS
#**************
best_fit_pars = np.loadtxt('./log_pars_%s_cube_%dwalkers_%dsteps.txt'%(tag, nwalkers, nsteps))[1]
Mstar, vsys, incl, PA, xc, yc, I0, p, q, L0, pL, qL, Ls, pLs, z0_upper, p_upper, Rb_upper, q_upper, z0_lower, p_lower, Rb_lower, q_lower = best_fit_pars

r'''
Mstar = 1.0 
vsys = 4.0 

incl = -0.88
PA = 1.5

I0 = 0.5 
p = -2.5
q = 2.0
Rout = 700

L0 = 0.3
pL = -0.3
qL = -0.4

Ls = 2.0
pLs = 0.3

z0_upper, p_upper, Rb_upper, q_upper = 30, 1.0, 500, 7.0
z0_lower, p_lower, Rb_lower, q_lower = 20, 1.0, 500, 7.0
#'''

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
model.params['intensity']['Rout'] = Rdisc/u.au
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

#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
Rdisc_m = Rdisc.to('m').value
vel2d, int2d, linew2d, lineb2d = model.make_model(R_inner=0, R_disc=Rdisc_m) #Projected attributes
R, phi, z, R_nonan, phi_nonan, z_nonan = model.get_projected_coords(R_inner=0, R_disc=Rdisc_m) #Projected disc coordinates

#**********************
#VISUALISE CHANNEL MAPS
#**********************
cube = model.get_cube(vchannels, vel2d, int2d, linew2d, lineb2d, make_convolve=True) #compute Cube object for model
cube.show(compare_cubes=[datacube], extent=extent, int_unit='Intensity [K]', show_beam=True, surface={'args': (R, phi), 'kwargs': {'extent': extent}})
cube.show_side_by_side(datacube, extent=extent, int_unit='Intensity [K]', show_beam=True, surface={'args': (R, phi), 'kwargs': {'extent': extent}})

#DATA CHANNELS
fig, ax, im, cbar = datacube.make_channel_maps(channels={'interval': [60, 70]}, ncols=5, attribute='intensity')
plt.savefig('testchans_data.png', bbox_inches = 'tight', dpi=100)
plt.close()

#MODEL CHANNELS
modelcube = Cube(cube.data, datacube.header, datacube.vchannels, beam=datacube.beam)
fig, ax, im, cbar = modelcube.make_channel_maps(channels={'interval': [60, 70]}, ncols=5, attribute='intensity', levels=im[0].levels)
plt.savefig('testchans_model.png', bbox_inches = 'tight', dpi=100)
plt.close()

#RESIDUAL CHANNELS
noise_mean = np.nanmean(noise)
residualscube = Cube(datacube.data-modelcube.data, datacube.header, datacube.vchannels, beam=datacube.beam)
fig, ax, im, cbar = residualscube.make_channel_maps(channels={'interval': [60, 70]}, ncols=5, attribute='residuals',
                                                    projection=None, extent=extent, unit_intensity='Kelvin', unit_coordinates='au',
                                                    mask_under=3*noise_mean, levels=np.linspace(-29, 29, 32))
plt.savefig('testchans_residuals.png', bbox_inches = 'tight', dpi=100)
plt.close()

#**********************
#MAKE MOMENT MAPS
#**********************
from astropy.io import fits

def write_moment_maps(moments, which='data', **kw_writeto):
    A,B,C,dA,dB,dC = moments

    fits.writeto('peakintensity_%s.fits'%which, A, **kw_writeto)
    fits.writeto('velocity_%s.fits'%which, B, **kw_writeto)
    fits.writeto('linewidth_%s.fits'%which, C, **kw_writeto)

    fits.writeto('delta_peakintensity_%s.fits'%which, dA, **kw_writeto)
    fits.writeto('delta_velocity_%s.fits'%which, dB, **kw_writeto)
    fits.writeto('delta_linewidth_%s.fits'%which, dC, **kw_writeto)

    return A,B,C,dA,dB,dC

moments_data = datacube.make_moments(method='gaussian')
write_moment_maps(moments_data, which='data', header=datacube.header, overwrite=True)

moments_model = modelcube.make_moments(method='gaussian')
write_moment_maps(moments_model, which='model', header=modelcube.header, overwrite=True)
