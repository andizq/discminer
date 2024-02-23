from .cube import Cube
from .tools.utils import InputError

from .disc2d import Model
from .cart import *

import numpy as np

from astropy.io import fits
from astropy import units as u
from radio_beam import Beam

import copy

class ReferenceModel(Cube):
    def __init__(
            self,
            Rmax = 700*u.au,
            dpc = 100*u.pc,
            nx = 128,
            vchannels = np.linspace(-5.0, 5.0, 101),
            Rmin = 1.0,
            write_extent = True,
            beam = Beam(
                major=0.15*u.arcsec,
                minor=0.15*u.arcsec,
                pa=0*u.deg
            ),
            velocity_func = keplerian_vertical,
            z_upper_func = z_upper_exp_tapered,
            z_lower_func = z_lower_exp_tapered,
            intensity_func = intensity_powerlaw_rout,
            linewidth_func = linewidth_powerlaw,
            lineslope_func = lineslope_powerlaw,
            line_profile = line_profile_bell,
            line_uplow = line_uplow_mask,
            init_params = {},
            init_header = {}
    ): 
        """
        Initialise ReferenceModel. Inherits `~discminer.disc2d.Cube` properties and methods.
        
        Parameters
        ----------
        Rmax : `~astropy.units.Quantity`
            Maximum 1D extent of the sky grid in physical units
        
        dpc : `~astropy.units.Quantity`
            Distance to the disc.
        """
        
        self.params = {
            'velocity': {
                'Mstar': 1.0,
                'vel_sign': 1,
                'vsys': 0
            },
            'orientation': {
                'incl': 0.7,
                'PA': 0.0,
                'xc': 0.0,
                'yc': 0.0
            },
            'intensity': {
                'I0': 0.5,
                'p': -1.5, 
                'q': 2.0,
                'Rout': 500
            },
            'linewidth': {
                'L0': 0.3, 
                'p': -0.5, 
                'q': -0.3
            }, 
            'lineslope': {
                'Ls': 2.0, 
                'p': 0.3, 
                'q': 0.0
            },
            'height_upper': {
                'z0': 40.0,
                'p': 1.0,
                'Rb': 500,
                'q': 2.0
            },
            'height_lower': {
                'z0': 20.0,
                'p': 1.0,
                'Rb': 500,
                'q': 2.0
            }
        }

        for key in init_params:
            self.params[key].update(init_params[key])
            
        nchan = len(vchannels)
        dchan = abs(vchannels[1]-vchannels[0])
        
        #INIT BEAM FOR HEADER
        if isinstance(beam, Beam):
            bmaj = beam.major.to(u.deg).value
            bmin = beam.minor.to(u.deg).value
            bpa = beam.pa.to(u.deg).value                    
        elif beam is None:
            bmaj = 0.0
            bmin = 0.0
            bpa = 0.0
        else:
            raise InputError(beam, "beam must be either None or radio_beam.Beam object")
        
        #HEADER
        dpix = np.arctan((Rmax.to(u.au) / (0.5*(nx-1))) / dpc.to(u.au)).to(u.deg)

        header = dict(
            SIMPLE  = True, 
            BITPIX  = -32, 
            NAXIS   = 3,                                                  
            NAXIS1  = nx,
            NAXIS2  = nx,
            NAXIS3  = nchan,
            EXTEND  = True,                                                  
            BMAJ    = bmaj,
            BMIN    = bmin,
            BPA     = bpa,
            BTYPE   = 'Intensity',                                                           
            OBJECT  = 'ReferenceDiscminer',                                                      
            BUNIT   = 'Jy/beam',      
            CTYPE1  = 'RA---SIN',                                                            
            CRVAL1  = 245.0,
            CDELT1  = -dpix.value,
            CRPIX1  = int(0.5*nx)+1,
            CUNIT1  = 'deg',                                                            
            CTYPE2  = 'DEC--SIN',
            CRVAL2  = -30.0,
            CDELT2  = dpix.value,
            CRPIX2  = int(0.5*nx)+1,
            CUNIT2  = 'deg',
            CTYPE3  = 'VRAD', #/ Radio velocity (linear) 
            CRVAL3  = vchannels[0],
            CDELT3  = dchan,
            CRPIX3  = 1,
            CUNIT3  = 'km/s',                                                            
            RESTFRQ = 3.457959899000E+11, #/Rest Frequency (Hz) - Band 7
            SPECSYS = 'LSRK', #/Spectral reference frame 
            VELREF  = 257 #/1 LSR, 2 HEL, 3 OBS, +256 Radio
        )
        
        header.update(init_header)
        hdu = fits.PrimaryHDU()
        hdu.header.update(header)        
        data = np.zeros((nchan, nx, nx))
        
        super().__init__(data, hdu.header, vchannels, dpc, beam=beam, filename="./referencecube.fits")
        
        #INIT MODEL
        model = Model(self, Rmax=Rmax, Rmin=Rmin, write_extent=write_extent, prototype=True)
        
        model.velocity_func = velocity_func
        model.z_upper_func = z_upper_func
        model.z_lower_func = z_lower_func
        model.intensity_func = intensity_func
        model.linewidth_func = linewidth_func
        model.lineslope_func = lineslope_func
        model.line_profile = line_profile
        model.line_uplow = line_uplow

        model.params = copy.copy(self.params)

        #UPDATE DATA VALUES
        self.data = model.make_model(make_convolve=True, return_data_only=True)                        
        self.model = model
