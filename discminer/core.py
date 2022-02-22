import numpy as np
from .disc2d import Tools
from .cube import Cube
from .utils import FITSUtils
from astropy import units as u
from astropy import constants as apc
from astropy.io import fits
from astropy.wcs import utils as aputils, WCS
from astropy.convolution import Gaussian2DKernel
from spectral_cube import SpectralCube
from radio_beam import Beam
from radio_beam.beam import NoBeamException
import warnings
import os


#The functions below _writefits()[inclusive] should be in the Cube class instead so that they can be used by the Model class without redundance. The Cube class should inherit a PlotTools class which can be used to make 2D plots (e.g. moment maps). Another advantage of doing that is that the Cube object would not have to be initialised over and over whenever a Cube-linked variable is modified. (DONE, except the PlotTools part)


class Data(Cube):
    def __init__(self, filename):
        """
        Initialise Data object. Inherits `~discminer.disc2d.Cube` properties and methods.
        
        Parameters
        ----------
        filename : path-like 
            Path to input FITS file with datacube information.

        """
        cube_spe = SpectralCube.read(filename)
        cube_vel = cube_spe.with_spectral_unit(
            u.km / u.s,
            velocity_convention="radio",
            rest_value=cube_spe.header["RESTFRQ"] * u.Hz,
        )
        # in km/s, remove .value to keep astropy units
        vchannels = cube_vel.spectral_axis.value
        header = cube_vel.header
        data = cube_vel.hdu.data.squeeze()        

        try:
            beam = Beam.from_fits_header(header)  # radio_beam object
        except NoBeamException:
            beam = None
            warnings.warn('No beam was found in the header of the input FITS file.')
        #self._init_cube()
        super().__init__(data, header, vchannels, beam=beam, filename=filename)

    def _init_cube(self):
        super().__init__(
            self.data,
            self.header,
            self.vchannels,
            beam = self.beam,
            filename = self.filename            
        )


#class Model(Cube, FITSUtils, Mcmc):
"""
beam : None or `~radio_beam.Beam`, optional
            - If None, the beam information is extracted from the header of the input FITS file.
            - If `~radio_beam.Beam` object, it uses this 
"""
