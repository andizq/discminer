from discminer.mining_control import _mining_moments1d
from discminer.core import Data

from astropy import units as u

import os
import json

if __name__ == '__main__':
    parser = _mining_moments1d(None)
    args = parser.parse_args()

#**************************
#JSON AND SOME DEFINITIONS
#**************************    
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']

dpc = meta['dpc']*u.pc
tag = meta['tag']

#*********
#LOAD DATA
#*********
if args.planck:
    file_data = 'cube_data_%s.fits'%tag
    file_model = 'cube_model_%s.fits'%tag
else:
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
    moments_data = datacube.make_moments(method=args.kernel, tag='data', sigma_thres=args.sigma, peak_kernel=args.peakkernel, fit_continuum=args.fit_continuum)
    moments_model = modelcube.make_moments(method=args.kernel, tag='model', sigma_thres=0, peak_kernel=args.peakkernel)


#*********************************
#CONVERT TO TB USING PLANCK'S LAW
#*********************************
if args.planck:
    filename = 'peakintensity_%s%s.fits'
    datapeak = Data(filename%(args.kernel, '_data'), dpc, twodim=True)
    datapeak.convert_to_tb(writefits=False, planck=True) #Don't save fits file with custom name
    datapeak.data = datapeak.data.squeeze()
    datapeak.writefits() #Overwrite instead

    modelpeak = Data(filename%(args.kernel, '_model'), dpc, twodim=True)
    modelpeak.convert_to_tb(writefits=False, planck=True)
    modelpeak.data = modelpeak.data.squeeze()
    modelpeak.writefits()
    
