from discminer.core import Data
from discminer.disc2d import Model
import discminer.cart as cart

import numpy as np
from astropy import units as u

import json
import copy

from argparse import ArgumentParser
from utils import add_parser_args

parser = ArgumentParser(prog='make moments', description='Make double gauss or double bell moments and save outputs into .fits files')
parser.add_argument('-ni', '--niter', default=10, type=int,
                    help="Number of iterations to re-do fit on hot pixels. DEFAULS to 10")
parser.add_argument('-ne', '--neighs', default=5, type=int,
                    help="Number of neighbour pixels on each side of hot pixel considered for the iterative fit. DEFAULS to 5")
args = add_parser_args(parser, kernel='doublebell', kind=True, sigma=True)

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
model = Model(datacube, Rmax, Rmin=0, prototype=True)

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
moments_data = datacube.make_moments(model=model, method=args.kernel, kind=args.kind, sigma_thres=args.sigma, niter=args.niter, neighs=args.neighs,
                                     writecomp=True, parcube=True, tag='_data') 
moments_model = modelcube.make_moments(model=model, method=args.kernel, kind=args.kind, sigma_thres=0, niter=5, #set sigma 0 to prevent masking of model
                                       writecomp=True, parcube=True, tag='_model') 

