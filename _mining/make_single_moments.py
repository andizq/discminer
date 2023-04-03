from discminer.core import Data

import json
from astropy import units as u
from argparse import ArgumentParser

parser = ArgumentParser(prog='make moments', description='Make gauss or bell moments and save outputs into .fits files')
parser.add_argument('-k', '--kind', default='gaussian', type=str, choices=['gauss', 'gaussian', 'bell'], help="1d moment kernel")
args = parser.parse_args()

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

datacube = Data(file_data, dpc) # Read data and convert to Cube object
modelcube = Data(file_model, dpc) # Read data and convert to Cube object

#**********************
#MAKE MOMENT MAPS
#**********************
moments_data = datacube.make_moments(method=args.kind, tag='data')
moments_model = modelcube.make_moments(method=args.kind, tag='model')

