from discminer.mining_control import _mining_skewkurt
from discminer.mining_utils import get_noise_mask
from discminer.core import Data

from astropy import units as u
from astropy.io import fits

import os
import json
import subprocess

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
def bm_percentiles(filepath, which='data', bluered=True):
    """
    Try computing line percentile maps with bettermoments if installed
    """
    try:
        betterm = subprocess.run(['bettermoments', filepath, '-method', 'percentiles'], capture_output=True, text=True, check=True)
        print("Calculating percentile maps with bettermoments:", betterm.stdout)
        
        for old, new in zip(['_wpdVb', '_dwpdVb', '_wpdVr', '_dwpdVr', '_wp50', '_dwp50', '_wp1684', '_dwp1684'], ['bluewidth_', 'delta_bluewidth_', 'redwidth_', 'delta_redwidth_', 'medianvelocity_', 'delta_medianvelocity_', 'meanvelocity_', 'delta_meanvelocity_']):
            os.rename(filepath.split('.fits')[0]+old+'.fits', new+'gaussian_%s.fits'%which)

        if bluered:
            bluewidth, bluehdr = fits.getdata('bluewidth_gaussian_%s.fits'%which, header=True)
            redwidth, redhdr = fits.getdata('redwidth_gaussian_%s.fits'%which, header=True)
            fits.writeto('bluered_gaussian_%s.fits'%which, bluewidth-redwidth, header=bluehdr, overwrite=True)
            
    except FileNotFoundError as e:
        print(f"Skipping calculation of percentile maps as Bettermoments may not be installed on your system: {e}")
        print("Try installing it via 'pip install bettermoments' or upgrade it to the latest stable version using 'pip install -U bettermoments'")
    
if args.fit_data:
    datacube = Data(file_data, dpc)    
    moments_data = datacube.make_skewkurt(mtype=args.mtype, tag='data')
    bm_percentiles(file_data, which='data')
    
if args.fit_model:
    modelcube = Data(file_model, dpc)
    moments_model = modelcube.make_skewkurt(mtype=args.mtype, tag='model')
    bm_percentiles(file_model, which='model')
