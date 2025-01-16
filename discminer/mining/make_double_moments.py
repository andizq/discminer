from discminer.mining_control import _mining_moments2d
from discminer.mining_utils import init_data_and_model
from discminer.core import Data

from astropy import units as u
from astropy.io import fits

import json

if __name__ == '__main__':
    parser = _mining_moments2d(None)
    args = parser.parse_args()

#**************************
#JSON AND SOME DEFINITIONS
#**************************    
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

Rout = best['intensity']['Rout']

file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.2*Rout*u.au #Max model radius, 30% larger than disc Rout

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model(Rmin=0, Rmax=Rmax)
vchannels = datacube.vchannels
model.make_model()

#**********
#LOAD MODEL
#**********
if args.planck:
    file_model = 'cube_model_%s.fits'%tag #Read Jy/bm version, and convert peakint to K after fit 
else:
    datacube.convert_to_tb(writefits=False) #RJ
    file_model = 'cube_model_%s_convtb.fits'%tag #Must be RJ
    
modelcube = Data(file_model, dpc) # Read model and convert to Cube object

#**********************
#MAKE MOMENT MAPS
#**********************
#Use model priors + kernel
moments_data = datacube.make_moments(model=model, method=args.kernel, kind=args.kind, sigma_thres=args.sigma, niter=args.niter, neighs=args.neighs,
                                     writecomp=True, parcube=True, tag='_data')
moments_model = modelcube.make_moments(model=model, method=args.kernel, kind=args.kind, sigma_thres=0, niter=args.niter, neighs=args.neighs,
                                       writecomp=True, parcube=True, tag='_model')


#*********************************
#CONVERT TO TB USING PLANCK'S LAW
#*********************************
if args.planck:
    for surf in ['up', 'low']:
        filename = 'peakintensity_%s_%s_%s%s.fits'
        datapeak = Data(filename%(surf, args.kernel, args.kind, '_data'), dpc, twodim=True)
        datapeak.convert_to_tb(writefits=False, planck=True) #Don't save fits file with custom name
        datapeak.data = datapeak.data.squeeze()
        datapeak.writefits() #Overwrite instead

        modelpeak = Data(filename%(surf, args.kernel, args.kind, '_model'), dpc, twodim=True)
        modelpeak.convert_to_tb(writefits=False, planck=True)
        modelpeak.data = modelpeak.data.squeeze()
        modelpeak.writefits() 

        #Update parcube peak
        fileparc = 'parcube_%s_%s_%s%s.fits'

        with fits.open(fileparc%(surf, args.kernel, args.kind, '_data'), mode='update') as hdul:
            data = hdul[0].data
            data[0] = datapeak.data
            hdul.flush()

        with fits.open(fileparc%(surf, args.kernel, args.kind, '_model'), mode='update') as hdul:
            data = hdul[0].data
            data[0] = modelpeak.data
            hdul.flush()
                
