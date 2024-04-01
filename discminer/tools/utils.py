from .._version import __version__
from .. import constants as sfc
from .. import units as sfu
from astropy import units as u
from astropy import constants as ct

import os
import sys
import copy
import numpy as np

hypot_func = lambda x,y: np.sqrt(x**2 + y**2) #Slightly faster than np.hypot<np.linalg.norm<scipydistance. Checked precision up to au**2 orders.

_molecule = {
    'c+': {
        'name': 'Carbon ion',
        'tex': r'C$^+$',
        'weight': 12.0 * u.u,
    },
    '12co': {
        'name': 'Carbon monoxide',
        'tex': r'$^{\rm 12}$CO',
        'weight': 28.01 * u.u,
    },
    '13co': {
        'name': 'Carbon monoxide',        
        'tex': r'$^{\rm 13}$CO',
        'weight': 29.0 * u.u,
    },
    'c18o': {
        'name': 'Carbon monoxide',        
        'tex': r'C$^{\rm 18}$O',
        'weight': 30.0 * u.u
    },
    'cs': {
        'name': 'Carbon monosulfide',        
        'tex': r'CS',
        'weight': 44.1 * u.u
    },
    'cn': {
        'name': 'Cyano radical',        
        'tex': r'CN',
        'weight': 26.0 * u.u
    },
    'hcn': {
        'name': 'Hydrogen cyanide',        
        'tex': r'HCN',
        'weight': 27.0 * u.u
    },
    'c2h': {
        'name': 'Ethynyl',        
        'tex': r'C$_{\rm 2}$H',
        'weight': 25.0 * u.u
    },
    'hco+': {
        'name': 'Formyl ion',        
        'tex': r'HCO$^+$',
        'weight': 29.0 * u.u
    },    
}


def get_mol_metadata(mol):
    return copy.copy(_molecule[mol])

class _Metadata(object):

    defunits = {
        'Rmin': u.au,
        'Rmax': u.au,
        'dpc': u.pc,        
        'bmaj': u.arcsec,
        'bmin': u.arcsec,
        'bpa': u.deg,
        'bmaj_au': u.au
    }

    @staticmethod
    def _convert_units(json_dict):
        for key in json_dict:
            if key in _Metadata.defunits:
                try:
                    json_dict[key] = json_dict[key].to(_Metadata.defunits[key]).value
                except AttributeError: #bmaj_au can be None
                    pass

    @property
    def json_metadata(self):
        return self._json_metadata

    @json_metadata.setter
    def json_metadata(self, val):
        _Metadata._convert_units(val)
        #print('Updating json_metadata info:', val) 
        self._json_metadata.update(val)
        
    def __init__(
            self,
            disc = None,            
            mol = '12co',
            **kwargs
    ):
        
        self.moldata = get_mol_metadata(mol)        
        
        self._json_metadata = {
            'v_discminer': __version__,
            'disc': disc,
            'mol': mol,
            'mol_weight': self.moldata['weight'].value
        }

        self._json_metadata.update(**kwargs)
        _Metadata._convert_units(self._json_metadata)

        
class InputError(Exception):
    """Exception raised for input errors.

    Parameters
    ----------
    expression : str
        Input expression where error occurred
    
    message : str
        Output description of the error
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
        
    def __str__(self):
        return '%s --> %s'%(self.expression, self.message)

    
class FrontendUtils(object):
    """
    Make outputs prettier
    """    
    path_icons = os.path.dirname(os.path.realpath(__file__))+'/../icons/'

    @staticmethod
    def _progress_bar(percent=0, width=50):
        left = width * percent // 100 
        right = width - left
        """
        print('\r[', '#' * left, ' ' * right, ']',
              f' {percent:.0f}%',
              sep='', end='', flush=True)
        """
        print('\r[', '#' * left, ' ' * right, ']', ' %.0f%%'%percent, sep='', end='') #compatible with python2 rtdocs
        sys.stdout.flush()

    @staticmethod
    def _break_line(init='', border='*', middle='=', end='\n', width=100):
        print('\r', init, border, middle * width, border, sep='', end=end)

    @staticmethod
    def _print_logo(filename=path_icons+'logo.txt'):
        logo = open(filename, 'r')
        print(logo.read())
        logo.close()
        

class FITSUtils(object):
    """
    Useful functions to transform datacube properties.
    """
    @staticmethod
    def _convert_to_tb(data, header, beam, planck=True, writefits=True, tag=""):
        """
        Convert intensity to brightness temperature in units of Kelvin.

        Parameters
        ----------
        data : (nchan, nx, nx) array_like
            Input intensity.
        
        header : FITS header
            Header of the datacube.
        
        beam : `~radio_beam.Beam` instance
            Beam from header, extracted with `~radio_beam`.
        
        planck : bool, optional
            If True, it uses the full Planck law to make the conversion, else it uses the Rayleigh-Jeans approximation. 

        writefits : bool, optional
            If True it creates a FITS file with the new intensity data and header.
        
        tag : str, optional
            String to add at the end of the output filename.

        kwargs : keyword arguments
            Additional keyword arguments to pass to `~astropy.io.fits.writeto` function.
           
        """
        I = data * u.Unit(header["BUNIT"]).to("beam-1 Jy")
        nu = header["RESTFRQ"]  # in Hz
        bmaj = beam.major.to(u.arcsecond).value
        bmin = beam.minor.to(u.arcsecond).value
        # area of gaussian beam
        beam_area = u.au.to("m") ** 2 * np.pi * (bmaj * bmin) / (4 * np.log(2))
        # beam solid angle: beam_area/(dist*pc)**2.
        #  dist**2 cancels out with beamarea's dist**2 from conversion of bmaj, bmin to mks units.
        beam_solid = beam_area / u.pc.to("m") ** 2
        Jy_to_SI = 1e-26
        c_h = ct.h.value
        c_c = ct.c.value
        c_k_B = ct.k_B.value

        if planck:
            Tb = (
                np.sign(I)
                * (
                    np.log(
                        (2 * c_h * nu ** 3)
                        / (c_c ** 2 * np.abs(I) * Jy_to_SI / beam_solid)
                        + 1
                    )
                )
                ** -1
                * c_h
                * nu
                / (c_k_B)
            )
        else:
            wl = c_c / nu
            Tb = 0.5 * wl ** 2 * I * Jy_to_SI / (beam_solid * c_k_B)

        return Tb


def get_tb(I, nu, beam, full=True):
    """
    nu in GHz
    Intensity in mJy/beam
    beam object from radio_beam
    if full: use full Planck law, else use rayleigh-jeans approximation
    """
    bmaj = beam.major.to(u.arcsecond).value
    bmin = beam.minor.to(u.arcsecond).value
    beam_area = sfu.au**2*np.pi*(bmaj*bmin)/(4*np.log(2)) #area of gaussian beam
    #beam solid angle: beam_area/(dist*pc)**2. dist**2 cancels out with beamarea's dist**2 from conversion or bmaj, bmin to mks units. 
    beam_solid = beam_area/sfu.pc**2 
    mJy_to_SI = 1e-3*1e-26
    nu = nu*1e9
    if full:
        Tb = np.sign(I)*(np.log((2*sfc.h*nu**3)/(sfc.c**2*np.abs(I)*mJy_to_SI/beam_solid)+1))**-1*sfc.h*nu/(sfc.kb) 
    else:
        wl = sfc.c/nu 
        Tb = 0.5*wl**2*I*mJy_to_SI/(beam_solid*sfc.kb) 
    #(1222.0*I/(nu**2*(beam.minor/1.0).to(u.arcsecond)*(beam.major/1.0).to(u.arcsecond))).value #nrao RayJeans             
    return Tb


def _get_beam_from(beam, dpix=None, distance=None, frac_pixels=1.0):
    """
    beam must be str pointing to fits file to extract beam from header or radio_beam Beam object.
    If radio_beam Beam instance is provided, pixel size (in SI units) will be extracted from grid obj. Distance (in pc) must be provided.
    #frac_pixels: number of averaged pixels on the data (useful to reduce computing time)
    """
    from astropy.io import fits
    from radio_beam import Beam
    sigma2fwhm = np.sqrt(8*np.log(2))
    if isinstance(beam, str):
        header = fits.getheader(beam)
        beam = Beam.from_fits_header(header)
        pix_scale = header['CDELT2'] * u.Unit(header['CUNIT2']) * frac_pixels
    elif isinstance(beam, Beam):
        if distance is None: raise InputError(distance, 'Wrong input distance. Please provide a value for the distance (in pc) to transform grid pix to arcsec')
        pix_radians = np.arctan(dpix / (distance*sfu.pc)) #dist*ang=projdist
        pix_scale = (pix_radians*u.radian).to(u.arcsec)  
    else: raise InputError(beam, 'beam object must either be str or Beam instance')

    x_stddev = ((beam.major/pix_scale) / sigma2fwhm).decompose().value 
    y_stddev = ((beam.minor/pix_scale) / sigma2fwhm).decompose().value 
    angle = (90*u.deg+beam.pa).to(u.radian).value
    gauss_kern = Gaussian2DKernel(x_stddev, y_stddev, angle) 
    
    #gauss_kern = beam.as_kernel(pix_scale) #as_kernel() slows down the run when passed to astropy.convolve
    return beam, gauss_kern


def weighted_std(prop, weights, weighted_mean=None):
    sum_weights = np.nansum(weights)
    if weighted_mean is None:
        weighted_mean = np.nansum(weights*prop)/sum_weights
    n = np.nansum(weights>0)
    w_std = np.sqrt(np.nansum(weights*(prop-weighted_mean)**2)/((n-1)/n * sum_weights))
    return w_std


def read_if_file_exists(base_str, file_str, n_none=3):
    if base_str is None: return [None]*n_none
    filename = base_str+file_str
    try:
        return np.loadtxt(filename)
    except OSError:
        message = 'File not found: %s'%filename
        warnings.warn(message)
        return [None]*n_none
