from discminer.core import Data
from utils import add_parser_args

import numpy as np
from astropy import units as u

import os
import json
from argparse import ArgumentParser


parser = ArgumentParser(prog='make moments', description='Make gauss or bell moments and save outputs into .fits files')
args = add_parser_args(parser, kernel=True, sigma=4)

with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']

#****************
#SOME DEFINITIONS
#****************
dpc = meta['dpc']*u.pc

#*********
#LOAD DATA
#*********
tag = meta['tag']
file_data = 'cube_data_%s_convtb.fits'%tag
file_model = 'cube_model_%s_convtb.fits'%tag

datacube = Data(file_data, dpc) # Read datacube and convert to Cube object
modelcube = Data(file_model, dpc) # Read modelcube and convert to Cube object

#**********************
#MAKE MOMENT MAPS
#**********************

if args.kernel=='quadratic':
    os.system('bettermoments '+file_data)
    os.system('bettermoments '+file_model)
    for old, new in zip(['_v0', '_dv0', '_Fnu', '_dFnu'], ['velocity_', 'delta_velocity_', 'peakintensity_', 'delta_peakintensity_']):
        os.rename(file_data.split('.fits')[0]+old+'.fits', new+'quadratic_data.fits')
        os.rename(file_model.split('.fits')[0]+old+'.fits', new+'quadratic_model.fits')
    
else:
    moments_data = datacube.make_moments(method=args.kernel, tag='data', sigma_thres=args.sigma)
    moments_model = modelcube.make_moments(method=args.kernel, tag='model', sigma_thres=0)

