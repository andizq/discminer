from discminer.mining_control import _mining_skewkurt
from discminer.mining_utils import get_noise_mask
from discminer.core import Data

from astropy import units as u

import json

if __name__ == '__main__':
    parser = _mining_skewkurt(None)
    args = parser.parse_args()

#**************************
#JSON AND SOME DEFINITIONS
#**************************    
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']

dpc = meta['dpc']*u.pc
tag = meta['tag']

#**********
#LOAD DATA
#**********
file_data = 'cube_data_%s_convtb.fits'%tag
file_model = 'cube_model_%s_convtb.fits'%tag

#*****************
#MAKE MOMENT MAPS
#*****************
if args.fit_data:
    datacube = Data(file_data, dpc)    
    moments_data = datacube.make_skewkurt(mtype=args.mtype, tag='data')
if args.fit_model:
    modelcube = Data(file_model, dpc)
    moments_model = modelcube.make_skewkurt(mtype=args.mtype, tag='model')
