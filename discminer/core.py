from .cube import Cube
from .disc2d import Mcmc
from .grid import grid as dgrid 
from astropy import units as u
import numpy as np
from radio_beam import Beam
from radio_beam.beam import NoBeamException
from spectral_cube import SpectralCube
import warnings

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

"""
beam : None or `~radio_beam.Beam`, optional
- If None, the beam information is extracted from the header of the input FITS file.
- If `~radio_beam.Beam` object, it uses this 
"""
        
class Model(Cube, Mcmc):
    def __init__(self, cube, dpc, Rdisc, Rin=1.0, prototype=False, subpixels=False):
        """
        Initialise Model object. Inherits `~discminer.disc2d.cube.Cube` properties and methods.

        Parameters
        ----------
        cube : `~discminer.disc2d.cube.Cube` object
            data (or model) cube to model through MCMC methods (see `~discminer.Model.run_mcmc`). It can also be used as a reference to find a prototype model (see `~discminer.Model.prototype`). The prototype model can then be employed as an initial guess for the MCMC parameter search.

        dpc : `~astropy.units.quantity.Quantity`
            Distance to object in physical units. The unit must be specified using the `~astropy.units` module, e.g. 100*units.pc

        Rdisc : `~astropy.units.quantity.Quantity`
            Disc radius in phisical units.

        Rin : float
            Inner radius of the disc to be masked in the model, in number of beams. Default is 1.0.

        prototype : bool, optional
            Compute a prototype model. This is useful for quick inspection of channels given a set of parameters, which can then be used as seeding parameters for the MCMC fit. Default is False.
        
        subpixels : bool, optional
            Subdivide original grid pixels into finer subpixels to account for large velocity gradients in the disc. This allows for a more precise calculation of line-of-sight velocities in regions where velocity gradients across individual pixels may be large, e.g. near the centre of the disc. Default is False.

        Attributes
        ----------
        skygrid : dict
            Dictionary with information of the sky plane grid where the disc observables are merged.

        discgrid : dict
            In some cases the maximum (deprojected) extent of the disc is larger than the size of the sky grid. Therefore, having independent grids is needed to make sure that the deprojected disc properties do not show sharp boundaries determined by the skyplane grid. In other cases, the maximum extent of the disc is smaller than the skyplane, in which case it's useful to employ a (smaller) independent grid to save computing resources.
            
        """
        self.dpc = dpc
        self.Rdisc = Rdisc
        self.prototype = prototype
        if isinstance(cube, Cube):
            self.datacube = cube
            self.header = cube.header
            self.vchannels = cube.vchannels
            self.beam = cube.beam            
            self.make_grid()
        beam_au = (dpc*np.tan(datacube.beam.major.to(u.radian))).to(u.au)
        self.Rin = Rin*beam_au

        #if prototype:
        #   make_prototype()
        #       vel2d, int2d, linew2d, lineb2d = model.make_model(R_inner=R_inner, R_disc=R_disc)
        #       cube = model.get_cube(vchan_data, vel2d, int2d, linew2d, lineb2d, tb = {'nu': restfreq, 'beam': model.beam_info, 'full': False/True})
        #       cube = model.get_cube(vchan_data, vel2d, int2d, linew2d, lineb2d, make_convolve=False)
        #Cube.__init__(data, self.header, self.vchannels, beam=self.beam, filename=filename)    

    def make_grid(self):
        dpix_rad = np.abs(self.header['CDELT2'])*u.Unit(self.header['CUNIT1']).to(u.radian)
        dpix_au = (self.dpc*np.tan(dpix_rad)).to(u.au)
        nx = self.header['NAXIS1']
        xsky = ysky = ((nx-1) * dpix_au/2.0) # Sky maximum extent
        # dpix_au*nx = np.abs(-xsky - dpix_au/2) + (xsky + dpix_au/2) --> xsky is referred
        #  to the centre of the left/rightmost pixel. To recover the full extent of the sky,
        #   which should be equal to dpix_au*nx, one has to add twice half the pixel size to
        #    account for the total extent of the border pixels.
        grid = dgrid(xsky, nx)        
        self.skygrid = grid

        # The cell size of the discgrid is the same as that of the skygrid.
        #  The extent of the discgrid is the closest posible to the input Rdisc using
        #   the aforementioned cell size.

        # discgrid should always exists, in general npix_disc != npix_sky,
        #  if npix_disc<npix_sky it's useful to use discgrid to save computing time,
        #   if npix_disc>npix_sky discgrid is needed for the emission to appear smooth
        #    instead of having sharp square boundaries.

        nx_disc = nx + int(np.round(2*self.Rdisc.to(u.au)/dpix_au - nx))
        nx_disc += nx_disc%2 # making it even
        xdisc = (nx_disc-1)*dpix_au/2.0
        self.discgrid = dgrid(xdisc, nx_disc)

