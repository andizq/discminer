from discminer.core import Data
from discminer.cube import Cube
from discminer.disc2d import General2d
from discminer.rail import Contours
from discminer.plottools import use_discminer_style
from discminer.tools import fit_kernel

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import fits

import json
import sys

from argparse import ArgumentParser
from utils import add_parser_args

use_discminer_style()

parser = ArgumentParser(prog='Plot reconstructed cubes', description='Plot upper+lower reconstructed cubes')
args = add_parser_args(parser, kernel='doublebell', kind=True)

#**********
#READ JSON
#**********
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc

datacube = Data(file_data, dpc)
datacube.convert_to_tb(writefits=False)

vchannels = datacube.vchannels

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
extent= np.array([-xmax, xmax, -xmax, xmax])

#****************
#READ PARCUBE AND MAKE CUBE FROM IT
#****************
parcube_up = fits.getdata('parcube_up_%s_%s_data.fits'%(args.kernel, args.kind))
parcube_low = fits.getdata('parcube_low_%s_%s_data.fits'%(args.kernel, args.kind))

fitdata = fit_kernel.get_channels_from_parcube(parcube_up, parcube_low, vchannels, method=args.kernel, kind=args.kind, n_fit=None)
fitcube = Cube(fitdata, datacube.header, vchannels, dpc, beam=datacube.beam)

#SHOW 
fitcube.show(compare_cubes=[datacube], extent=extent, int_unit='Intensity [K]', vmin=0.0, vmax=40, show_beam=True)
fitcube.show_side_by_side(datacube, extent=extent, int_unit='Intensity [K]', vmin=0.0, vmax=40, show_beam=True)
