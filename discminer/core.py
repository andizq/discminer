from .cube import Cube
from .grid import grid as dgrid 
from .tools.utils import InputError

import numpy as np

from astropy.io import fits
from astropy import units as u
from radio_beam import Beam, Beams
from radio_beam.beam import NoBeamException
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube

import numbers
import warnings
import json

class Data(Cube):
    def __init__(self, filename, dpc, twodim=False, disc=None, mol='12co', kind=['mask']):
        """
        Initialise Data object. Inherits `~discminer.disc2d.Cube` properties and methods.
        
        Parameters
        ----------
        filename : path-like 
            Path to input FITS file with datacube information.
        
        dpc : `~astropy.units.Quantity`
            Distance to the disc.
        """

        if twodim: #continuum image

            hdu = fits.open(filename)[0]
            header = hdu.header            
            data = hdu.data

            if len(data.shape)>3:
                data = data.squeeze()
            if len(data.shape)==2:
                data = np.expand_dims(data, 0)
                
            try:
                beam = Beam.from_fits_header(header)
            except NoBeamException:
                beam = None
                warnings.warn('No beam was found in the input FITS file.')
            try:
                ncomp = header['NAXIS3'] #e.g. Stokes components
            except KeyError:
                ncomp = 2

            if ncomp<2:
                ncomp = 2
                
            vchannels = np.arange(ncomp)

        else: #line cube
            cube_spe = SpectralCube.read(filename)
            try:
                cube_vel = cube_spe.with_spectral_unit(
                    u.km / u.s,
                    velocity_convention="radio",
                    rest_value=cube_spe.header["RESTFRQ"] * u.Hz,
                )
            except KeyError:
                #Assume velocity axis is already in km/s
                cube_vel = cube_spe
            
            #In km/s, remove .value to keep astropy units
            vchannels = cube_vel.spectral_axis.value
            header = cube_vel.header
        
            #Get data and beam info
            if isinstance(cube_vel, SpectralCube):
                data = cube_vel.hdu.data.squeeze()
                try:
                    beam = Beam.from_fits_header(header)  # radio_beam object
                except NoBeamException:
                    beam = None
                    warnings.warn('No beam was found in the input FITS file.')
            
            elif isinstance(cube_vel, VaryingResolutionSpectralCube):
                data = cube_vel.hdulist[0].data.squeeze()
                """
                beams = cube_vel.hdulist[1].data #One beam per channel
                bmaj = np.median(beams['BMAJ'])
                bmin = np.median(beams['BMIN'])
                bpa = np.median(beams['BPA'])            
                beam = Beam(major=bmaj*u.arcsec, minor=bmin*u.arcsec, pa=bpa*u.deg)
                """
                beams = Beams.from_fits_bintable(cube_vel.hdulist[1])
                beam = beams.common_beam() #Smallest common beam
                header.update(beam.to_header_keywords()) #Add single beam to header
                
            else:
                raise InputError(cube_vel,
                                 'The input datacube is not valid. Only the following spectral_cube instances are supported: SpectralCube, VaryingResolutionSpectralCube.')
            
        super().__init__(data, header, vchannels, dpc, beam=beam, filename=filename, disc=disc, mol=mol, kind=kind)

    def _init_cube(self):
        super().__init__(
            self.data,
            self.header,
            self.vchannels,
            self.dpc,
            beam = self.beam,
            filename = self.filename            
        )

"""
beam : None or `~radio_beam.Beam`, optional
- If None, the beam information is extracted from the header of the input FITS file.
- If `~radio_beam.Beam` object, it uses this 
"""
        
class ModelGrid():
    def __init__(self, datacube, Rmax, Rmin=1.0, write_extent=True):
        """
        Initialise ModelGrid object.

        Parameters
        ----------
        datacube : `~discminer.disc2d.cube.Cube` object
            Datacube to get sky grid from.

        Rmax : `~astropy.units.Quantity`
            Maximum radial extent of the model in physical units. Not to be confused with the disc outer radius.

        Rmin : float or `~astropy.units.Quantity`
            Inner radius to mask out from the model.

            - If float, computes inner radius in number of beams. Default is 1.0.

            - If `~astropy.units.Quantity`, takes the value provided, assumed in physical units.
        
        write_extent : bool
            If True, writes information about grid physical extent into JSON file.

        Attributes
        ----------
        skygrid : dict
            Dictionary with useful information of the sky grid where the disc observable properties are merged for visualisation. This grid matches the spatial grid of the input datacube in physical units.

        grid : dict
            Disc grid where the model disc properties are computed. Why are there two different grids? Sometimes, the maximum (deprojected) extent of the disc is larger than the rectangular size of the sky grid. Therefore, having independent grids is needed to make sure that the deprojected disc properties do not display sharp boundaries at R=skygrid_extent. In other cases, the maximum extent of the disc is smaller than the sky grid, in which case it's useful to employ a (smaller) independent grid to save computing time.
            
        """
        
        self.dpc = datacube.dpc
        self.Rmax = Rmax
        self.write_extent = write_extent
        
        if isinstance(datacube, Cube):
            self.datacube = datacube
            self.header = datacube.header
            self.vchannels = datacube.vchannels
            self.beam = datacube.beam            
            self._make_grid()
        else:
            raise InputError(datacube,
                             'The input cube must be a ~discminer.disc2d.cube.Cube instance.')
        
        if isinstance(Rmin, numbers.Real):
            if datacube.beam is not None:
                beam_au = (self.dpc*np.tan(datacube.beam.major.to(u.radian))).to(u.au)
                self.Rmin = Rmin*beam_au
            else: self.Rmin = 0.0*u.Unit(u.au)
        else:
            self.Rmin = Rmin.to(u.au)
            
        #elif isinstance(datacube, grid):
        #   self.beam = beam
        #   make_header()
        #   make_channels()
        #   make_prototype()
        #
        
        #if prototype:
        #   make_prototype()
        #       vel2d, int2d, linew2d, lineb2d = model.make_model(R_inner=R_inner, R_disc=R_disc)
        #       cube = model.get_cube(vchan_data, vel2d, int2d, linew2d, lineb2d, tb = {'nu': restfreq, 'beam': model.beam_info, 'full': False/True})
        #       cube = model.get_cube(vchan_data, vel2d, int2d, linew2d, lineb2d, make_convolve=False)
        #Cube.__init__(data, self.header, self.vchannels, beam=self.beam, filename=filename)    

    def _make_grid(self):
        dpix_rad = np.abs(self.header['CDELT1'])*u.Unit(self.header['CUNIT1']).to(u.radian)
        dpix_au = (self.dpc*np.tan(dpix_rad)).to(u.au)
        nx = self.header['NAXIS1']
        ny = self.header['NAXIS2']
        if nx!=ny:
            raise InputError((nx, ny), 'Number of spatial rows and columns in the datacube must be equal. '\
                             'Please clip the datacube using the datacube.clip(npix) method to produce a square cube.')
        
        xsky = ysky = ((nx-1) * dpix_au/2.0) # Sky maximum extent
        # dpix_au*nx = np.abs(-xsky - dpix_au/2) + (xsky + dpix_au/2) --> xsky is referred
        #  to the centre of the left/rightmost pixel. To recover the full extent of the sky,
        #   which should be equal to dpix_au*nx, one has to add twice half the pixel size to
        #    account for the total extent of the border pixels.
        grid = dgrid(xsky, nx) #Transforms xsky from au to metres and computes Cartesian grid
        self.skygrid = grid

        # The cell size of the discgrid is the same as that of the skygrid.
        #  The extent of the discgrid is the closest posible to the input Rmax using
        #   the aforementioned cell size.

        # discgrid should always exists, in general npix_disc != npix_sky,
        #  if npix_disc<npix_sky it's useful to use discgrid to save computing time,
        #   if npix_disc>npix_sky discgrid is needed for the emission to appear smooth
        #    instead of having sharp square boundaries.

        nx_disc = nx + int(np.round(2*self.Rmax.to(u.au)/dpix_au - nx))
        nx_disc += nx_disc%2 # making it even
        xdisc = (nx_disc-1)*dpix_au/2.0
        self.discgrid = dgrid(xdisc, nx_disc) #Transforms xdisc from au to metres and computes Cartesian grid
        
        if self.write_extent:
            log_grid = dict(nx=nx, ny=ny, xsky=xsky.value, xdisc=xdisc.value, cellsize=dpix_au.value, unit='au')
            with open("grid_extent.json", "w") as outfile:
                json.dump(log_grid, outfile, indent=4, sort_keys=False)
