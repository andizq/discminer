from discminer.mining_control import _mining_skewkurt
from discminer.mining_utils import get_noise_mask, load_disc_grid
from discminer.core import Data

from astropy import units as u
from astropy.io import fits
import numpy as np

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
best = pars['metadata']

dpc = meta['dpc']*u.pc
tag = meta['tag']

#**********
#LOAD DATA
#**********
file_data = 'cube_data_%s_convtb.fits'%tag
file_model = 'cube_model_%s_convtb.fits'%tag

#LOAD DISC GEOMETRY
R, phi, z = load_disc_grid()

phi_up = phi['upper']
phi_low = phi['lower']

def swap_halfplanes(a, b, mask):
    """Swap values between arrays a and b wherever mask is True."""
    a2 = a.copy()
    b2 = b.copy()
    a2[mask] = b[mask]
    b2[mask] = a[mask]
    return a2, b2

#*****************
#MAKE MOMENT MAPS
#*****************
def bm_percentiles(filepath, which='data', bluered=True, mirror=args.mirror):
    """
    Compute line percentile maps using `bettermoments`, if available.

    Parameters
    ----------
    filepath : str
        Path to the input data or model cube.

    which : {'data', 'model'}, optional
        Select whether percentiles are computed from the observed data cube
        ('data') or from a model cube ('model'). Default is 'data'.

    bluered : bool, optional
        If True, compute blue-red percentile metrics (e.g., bluewidth minus
        redwidth) instead of returning the individual components. Default is True.

    mirror : bool, optional
        If True, mirror the blue- and red-width metrics across the minor axis
        of the disc. Because emission from the backside of the disc is
        antisymmetric in velocity, the red- and blue-width maps trace opposite
        disc surfaces on either side of the minor axis and are therefore not
        directly comparable. This option enforces a consistent front/back
        assignment across the disc. Default is True.

    Notes
    -----
    This function requires the external `bettermoments` package to be installed
    and accessible from the command line.
    """

    try:
        betterm = subprocess.run(['bettermoments', filepath, '-method', 'percentiles'], capture_output=True, text=True, check=True)
        print("Calculating percentile maps with bettermoments:", betterm.stdout)
        
        for old, new in zip(['_wpdVb', '_dwpdVb', '_wpdVr', '_dwpdVr', '_wp50', '_dwp50', '_wp1684', '_dwp1684'], ['bluewidth_', 'delta_bluewidth_', 'redwidth_', 'delta_redwidth_', 'medianvelocity_', 'delta_medianvelocity_', 'meanvelocity_', 'delta_meanvelocity_']):
            os.rename(filepath.split('.fits')[0]+old+'.fits', new+'gaussian_%s.fits'%which)

        if mirror: 
            # mask for azimuths: phi > +pi/2 or phi < -pi/2 using both disc surfaces
            mirror_mask = (phi_up > np.pi/2) | (phi_up < -np.pi/2) | (phi_low > np.pi/2) | (phi_low < -np.pi/2)

            bluewidth, bluehdr = fits.getdata(f'bluewidth_gaussian_{which}.fits', header=True)
            redwidth, redhdr  = fits.getdata(f'redwidth_gaussian_{which}.fits',  header=True)
            delta_bluewidth, delta_bluehdr = fits.getdata(f'delta_bluewidth_gaussian_{which}.fits', header=True)
            delta_redwidth, delta_redhdr = fits.getdata(f'delta_redwidth_gaussian_{which}.fits', header=True)
            
            bluewidth, redwidth = swap_halfplanes(bluewidth, redwidth, mirror_mask)
            delta_bluewidth, delta_redwidth = swap_halfplanes(delta_bluewidth, delta_redwidth, mirror_mask)

            fits.writeto(f'bluewidth_gaussian_{which}.fits', bluewidth, bluehdr, overwrite=True)
            fits.writeto(f'redwidth_gaussian_{which}.fits',  redwidth,  redhdr,  overwrite=True)
            fits.writeto(f'delta_bluewidth_gaussian_{which}.fits', delta_bluewidth, delta_bluehdr, overwrite=True)
            fits.writeto(f'delta_redwidth_gaussian_{which}.fits', delta_redwidth, delta_redhdr, overwrite=True)
            
        if bluered:
            bluewidth, bluehdr = fits.getdata(f'bluewidth_gaussian_{which}.fits', header=True)
            redwidth, redhdr  = fits.getdata(f'redwidth_gaussian_{which}.fits',  header=True)
            
            fits.writeto(f'bluered_gaussian_{which}.fits', bluewidth-redwidth, header=bluehdr, overwrite=True)
            
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
