from discminer.core import Data
from discminer.disc2d import Model

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
import emcee

from argparse import ArgumentParser

parser = ArgumentParser(prog='Handle emcee backend', description='Handle emcee backend')
parser.add_argument('-b', '--backend', default=1, type=int, choices=[0, 1], help="If 0, create new backend. If 1, reuse existing backend")
args = parser.parse_args()

#*********************
#REQUIREd DEFINITIONS
#*********************
file_data = 'MWC_480_CO_220GHz.robust_0.5.JvMcorr.image.pbcor_clipped_downsamp_10pix.fits'
tag_out = 'mwc480_12co_0p2_maps' #PREFERRED FORMAT: disc_mol_chan_program_extratags
tag_in = tag_out

nwalkers = 150
nsteps = 15000

dpc = 162.0*u.pc
vel_sign = -1 #Rotation direction: -1 or 1

Rmax = 1000*u.au #Model maximum radius

#*********
#READ DATA
#*********
datacube = Data(file_data, dpc) #Read data and convert to Cube object
vchannels = datacube.vchannels

au_to_m = u.au.to('m')

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = Model(datacube, Rmax, Rmin=0, prototype = False)                  

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
model.velocity_func = model.keplerian_vertical # i.e. vrot = sqrt(GM/r**3)*R
model.line_profile = model.line_profile_bell
model.intensity_func = intensity_powerlaw_rout

#If not redefined, intensity and linewidth are powerlaws 
 #of R and z by default, whereas lineslope is constant.
  #See Table 1 of discminer paper 1 (izquierdo+2021).

#****************************
#SET FREE PARS AND BOUNDARIES
#****************************
# If True, parameter is allowed to vary freely.
#  If float, parameter is fixed to the value provided.

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

model.mc_boundaries['velocity']['vsys'] = (-10, 10)
model.mc_boundaries['orientation']['incl'] = (-1.2, 1.2)
model.mc_boundaries['intensity']['I0'] = (0, 5)
model.mc_boundaries['intensity']['Rout'] = (100, 1000)
model.mc_boundaries['height_upper']['z0'] = (0, 200)
model.mc_boundaries['height_upper']['p'] = (0, 5)
model.mc_boundaries['height_upper']['Rb'] = (0, 1000)
model.mc_boundaries['height_upper']['q'] = (0, 10)
model.mc_boundaries['height_lower']['z0'] = (0, 200)
model.mc_boundaries['height_lower']['p'] = (0, 5)
model.mc_boundaries['height_lower']['Rb'] = (0, 1000)
model.mc_boundaries['height_lower']['q'] = (0, 10)

#***************************
#INITIAL GUESS OF PARAMATERS
#***************************
Mstar = 2.0
vsys = 6.0

incl = -0.8
PA = 0.7
xc = 0
yc = 0

I0 = 1.0
p = -2.5
q = 2.0
Rout = 800

L0 = 0.3
pL = -0.3
qL = -0.4

Ls = 2.0
pLs = 0.3

z0_upper, p_upper, Rb_upper, q_upper = 30, 1.2, 600, 3.0
z0_lower, p_lower, Rb_lower, q_lower = 20, 1.2, 600, 3.0

p0 = [Mstar, vsys,                              #Velocity
      incl, PA, xc, yc,                         #Orientation
      I0, p, q, Rout,                           #Intensity
      L0, pL, qL,                               #Line width
      Ls, pLs,                                  #Line slope
      z0_upper, p_upper, Rb_upper, q_upper,     #Upper surface height
      z0_lower, p_lower, Rb_lower, q_lower      #Lower surface height
]

#********
#RUN MCMC
#********
# Set up the emcee backend
filename = "backend_%s.h5"%tag_out
backend = None

#try and except statement failing with FileNotFoundError/OSError
if args.backend:
    #Succesive runs
    backend = emcee.backends.HDFBackend(filename)
else:
    #First run: Initialise backend
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, len(p0))

print("Backend Initial size: {0} steps".format(backend.iteration))

# Noise in each pixel is stddev of intensity from first and last 5 channels 
noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0) 

#Run Emcee
model.run_mcmc(datacube.data, vchannels,
               p0_mean=p0, nwalkers=nwalkers, nsteps=nsteps,
               backend=backend,
               tag=tag_out,
               #nthreads=96, # If not specified considers maximum possible number of cores
               frac_stats=0.1,
               frac_stddev=1e-2,
               noise_stddev=noise) 

print("Backend Final size: {0} steps".format(backend.iteration))

#***************************************
#SAVE SEEDING, BEST FIT PARS, AND ERRORS
model.mc_header.append('vel_sign')
np.savetxt('log_pars_%s_cube_%dwalkers_%dsteps.txt'%(tag_out, nwalkers, backend.iteration), 
           np.array([np.append(p0, vel_sign),
                     np.append(model.best_params, vel_sign),
                     np.append(model.best_params_errneg, 0.0),
                     np.append(model.best_params_errpos, 0.0)
           ]), 
           fmt='%.6f', header=str(model.mc_header))
