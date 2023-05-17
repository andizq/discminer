from discminer.core import Data

import json
from astropy import units as u
from argparse import ArgumentParser

parser = ArgumentParser(prog='make moments', description='Make gauss or bell moments and save outputs into .fits files')
parser.add_argument('-m', '--method', default='gaussian', type=str, choices=['gauss', 'gaussian', 'bell'], help="1d moment kernel")
parser.add_argument('-s', '--sigma', default=4, type=float, help='Mask out pixels with peak intensities below sigma threshold')
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

datacube = Data(file_data, dpc) # Read datacube and convert to Cube object
modelcube = Data(file_model, dpc) # Read modelcube and convert to Cube object

#**********************
#MAKE MOMENT MAPS
#**********************
moments_data = datacube.make_moments(method=args.method, tag='data', sigma_thres=args.sigma)
moments_model = modelcube.make_moments(method=args.method, tag='model', sigma_thres=args.sigma)

