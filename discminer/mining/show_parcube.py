from discminer.mining_control import _mining_parcube
from discminer.plottools import use_discminer_style
from discminer.tools import fit_kernel
from discminer.core import Data
from discminer.cube import Cube

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import fits

import json

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_parcube(None)
    args = parser.parse_args()

#**************************
#JSON AND SOME DEFINITIONS
#**************************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

custom = pars['custom']
meta = pars['metadata']
file_data = meta['file_data']
tag = meta['tag']
Ilim = custom['Ilim']

au_to_m = u.au.to('m')
dpc = meta['dpc']*u.pc

datacube = Data(file_data, dpc)
datacube.convert_to_tb(writefits=False)

vchannels = datacube.vchannels

with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
extent= np.array([-xmax, xmax, -xmax, xmax])

#**********************************
#READ PARCUBE AND RECONSTRUCT CUBE
#**********************************
if args.kernel in ['gaussian', 'bell']:
    parcube_up = fits.getdata('parcube_%s_data.fits'%args.kernel)
    parcube_low = None
else:
    parcube_up, parcube_low = None, None
    if args.surface in ['upper', 'both']:
        parcube_up = fits.getdata('parcube_up_%s_%s_data.fits'%(args.kernel, args.kind))
    if args.surface in ['lower', 'both']:        
        parcube_low = fits.getdata('parcube_low_%s_%s_data.fits'%(args.kernel, args.kind))

fitdata = fit_kernel.get_channels_from_parcube(parcube_up, parcube_low, vchannels, method=args.kernel, kind=args.kind, n_fit=None)
fitcube = Cube(fitdata, datacube.header, vchannels, dpc, beam=datacube.beam)

#SHOW 
fitcube.show(compare_cubes=[datacube], extent=extent, int_unit='Intensity [K]', vmin=0.0, vmax=3*Ilim, show_beam=True)
fitcube.show_side_by_side(datacube, extent=extent, int_unit='Intensity [K]', vmin=0.0, vmax=3*Ilim, show_beam=True)
