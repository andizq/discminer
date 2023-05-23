from discminer.core import Data
from discminer.disc2d import General2d
import discminer.cart as cart

import numpy as np
from astropy import units as u

import json
import copy

from argparse import ArgumentParser

parser = ArgumentParser(prog='make moments', description='Make double gauss or double bell moments and save outputs into .fits files')
parser.add_argument('-m', '--method', default='doublebell', type=str, choices=['dgauss', 'dbell', 'doublegaussian', 'doublebell'], help="Type of two-component kernel to fit")
parser.add_argument('-k', '--kind', default='mask', type=str, choices=['mask', 'sum'], help="How the two kernels must be merged")
parser.add_argument('-s', '--sigma', default=5, type=float, help='Mask out pixels with peak intensities below sigma threshold')
args = parser.parse_args()

if args.method == 'dbell': args.method='doublebell'
if args.method == 'dgauss': args.method='doublegaussian'

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
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.3*best['intensity']['Rout']*u.au #Max model radius, 30% larger than disc Rout

#*********
#LOAD DATA
#*********
datacube = Data(file_data, dpc) # Read data and convert to Cube object
vchannels = datacube.vchannels

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = General2d(datacube, Rmax, Rmin=0, prototype=True)

model.velocity_func = model.keplerian_vertical # vrot = sqrt(GM/r**3)*R
model.line_profile = model.line_profile_bell

if 'sum' in meta['kind']:
    model.line_uplow = model.line_uplow_sum
else:
    model.line_uplow = model.line_uplow_mask

if 'I2pwl' in meta['kind']:
    model.intensity_func = cart.intensity_powerlaw_rbreak
elif 'I2pwlnosurf' in meta['kind']:
    model.intensity_func = cart.intensity_powerlaw_rbreak_nosurf    
else:
    model.intensity_func = cart.intensity_powerlaw_rout

if 'surf2pwl' in meta['kind']:
    model.z_upper_func = cart.z_upper_powerlaw
    model.z_lower_func = cart.z_lower_powerlaw
else:
    model.z_upper_func = cart.z_upper_exp_tapered
    model.z_lower_func = cart.z_lower_exp_tapered
  
#**************
#PROTOTYPE PARS
#**************
model.params = copy.copy(best)
model.params['intensity']['I0'] /= meta['downsamp_factor']
model.make_model()

#**********
#LOAD MODEL
#**********
datacube.convert_to_tb(writefits=False)
file_model = 'cube_model_%s_convtb.fits'%tag
modelcube = Data(file_model, dpc) # Read model and convert to Cube object

#**********************
#MAKE MOMENT MAPS
#**********************
#Use model priors + kernel
moments_data = datacube.make_moments(model=model, method=args.method, kind=args.kind, sigma_thres=args.sigma,
                                     writecomp=True, parcube=True, tag='_data') 
moments_model = modelcube.make_moments(model=model, method=args.method, kind=args.kind, sigma_thres=args.sigma,
                                       writecomp=True, parcube=True, tag='_model') 

