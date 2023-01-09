from discminer.core import Data, Model
from discminer.disc2d import General2d

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u

au_to_m = u.au.to('m')

#****************
#SOME DEFINITIONS
#****************
file_data = 'HD_135344B_12CO_robust0.0_width0.100kms_threshold2.0sigma_taper0.1arcsec.clean.image_clipped_downsamp_6pix.fits'
tag_out = 'hd135344_12co_0p1_exoalma'
tag_in = tag_out

nwalkers = 256
nsteps = 5000

dpc = 135*u.pc
vel_sign = 1 # Rotation direction: -1 or 1

#*********
#READ DATA
#*********
datacube = Data(file_data, dpc) # Read data and convert to Cube object
vchannels = datacube.vchannels
Rmax = 300*u.au
Rmin = 0*u.au

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = General2d(datacube, Rmax, Rmin=Rmin, prototype=False) #If Rmin is a dimensionless integer discminer assumes Rmin*beam_size, default Rmin=1. 

def intensity_powerlaw_rout(coord, I0=30.0, R0=100*au_to_m, p=-0.4, z0=100*au_to_m, q=0.3, Rout=200):
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    z = coord['z']
    A = I0*R0**-p*z0**-q
    Ieff = np.where(R<=Rout*au_to_m, A*R**p*np.abs(z)**q, 0.0)
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

#****************************
#SET FREE PARS AND BOUNDARIES
#****************************
# If True, parameter is allowed to vary through mcmc.
#  If float, parameter is fixed to provided value.

model.mc_params['velocity']['vel_sign'] = vel_sign 
model.mc_params['velocity']['Mstar'] = True
model.mc_params['velocity']['vsys'] = True 
model.mc_params['orientation']['incl'] = True
model.mc_params['orientation']['PA'] = True 
model.mc_params['orientation']['xc'] = True
model.mc_params['orientation']['yc'] = True
model.mc_params['intensity'] = {'I0': True, 'p': True, 'q': True, 'Rout': True}
model.mc_params['linewidth']['L0'] = True
model.mc_params['linewidth']['p'] = True
model.mc_params['linewidth']['q'] = True
model.mc_params['lineslope']['Ls'] = True
model.mc_params['lineslope']['p'] = True
model.mc_params['height_upper'] = {'z0': True, 'p': True, 'Rb': True, 'q': True}
model.mc_params['height_lower'] = {'z0': True, 'p': True, 'Rb': True, 'q': True}
                                   
# Boundaries of user-defined attributes must be defined here.
# Boundaries of attributes existing in discminer can be modified here, otherwise default values are taken.

model.mc_boundaries['velocity']['vsys'] = (0, 15)
model.mc_boundaries['intensity']['I0'] = (0, 5)
model.mc_boundaries['intensity']['Rout'] = (100, 300)
model.mc_boundaries['height_upper']['z0'] = (0, 200)
model.mc_boundaries['height_upper']['p'] = (0, 5)
model.mc_boundaries['height_upper']['Rb'] = (0, 1000)
model.mc_boundaries['height_upper']['q'] = (0, 10)
model.mc_boundaries['height_lower']['z0'] = (0, 200)
model.mc_boundaries['height_lower']['p'] = (0, 5)
model.mc_boundaries['height_lower']['Rb'] = (0, 1000)
model.mc_boundaries['height_lower']['q'] = (0, 10)

#*************************
#PREPARE PARS AND RUN MCMC
#*************************
#best_fit_pars = np.loadtxt('./log_pars_%s_cube_50walkers_1000steps.txt'%(tag_in))[1]
#Mstar, vsys, incl, PA, I0, p, q, Rout, L0, pL, qL, Ls, pLs, z0_upper, p_upper, Rb_upper, q_upper, z0_lower, p_lower, Rb_lower, q_lower = best_fit_pars

#r"""
Mstar = 1.5
vsys = 7.0

incl = 0.2
PA = 0.0
xc = 0.0
yc= 0.0

I0 = 0.5
p = -2.5
q = 2.0
Rout = 200

L0 = 0.3
pL = -0.3
qL = -0.4

Ls = 2.0
pLs = 0.3

z0_upper, p_upper, Rb_upper, q_upper = 30, 1.0, 300, 7.0
z0_lower, p_lower, Rb_lower, q_lower = 20, 1.0, 300, 7.0
#"""

#List of first-guess parameters
p0 = [Mstar, vsys, 
      incl, PA, xc, yc,
      I0, p, q, Rout,
      L0, pL, qL,
      Ls, pLs, 
      z0_upper, p_upper, Rb_upper, q_upper,
      z0_lower, p_lower, Rb_lower, q_lower
]


# Set up the backend
import emcee

filename = "backend_%s.h5"%tag_out
backend = None

#First run: Initialise backend
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, len(p0))

#Succesive runs
#backend = emcee.backends.HDFBackend(filename)

#Run MCMC
print("Backend Initial size: {0} steps".format(backend.iteration))

# Noise in each pixel is stddev of intensity from first and last 5 channels 
noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0) 

model.run_mcmc(datacube.data, vchannels,
               p0_mean=p0, nwalkers=nwalkers, nsteps=nsteps,
               backend=backend,
               tag=tag_out,
               frac_stats=0.1,
               frac_stddev=1e-2,
               noise_stddev=noise) 

print("Backend Final size: {0} steps".format(backend.iteration))

#***************************************
#SAVE SEEDING, BEST FIT PARS, AND ERRORS
np.savetxt('log_pars_%s_cube_%dwalkers_%dsteps.txt'%(tag_out, nwalkers, backend.iteration), 
           np.array([p0, model.best_params, model.best_params_errneg, model.best_params_errpos]), 
           fmt='%.6f', header=str(model.mc_header))
