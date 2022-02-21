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
            Path to FITS datacube

        """
        self.fileroot = os.path.expanduser(filename).split(".fits")[0]
        cube_spe = SpectralCube.read(filename)
        cube_vel = cube_spe.with_spectral_unit(
            u.km / u.s,
            velocity_convention="radio",
            rest_value=cube_spe.header["RESTFRQ"] * u.Hz,
        )
        # in km/s, remove .value to keep astropy units
        self.vchannels = cube_vel.spectral_axis.value
        self.header = cube_vel.header
        self.data = cube_vel.hdu.data.squeeze()
        self.wcs = WCS(self.header)
        # Assuming (nchan, nx, nx), nchan should be equal to cube_vel.spectral_axis.size
        self.nchan, self.nx, _ = np.shape(self.data)

        try:
            self.beam = Beam.from_fits_header(self.header)  # radio_beam object
        except NoBeamException:
            self.beam = None
            warnings.warn('No beam was found in the header of the input FITS file.')
        self._init_cube()

    def _init_cube(self):
        super().__init__(
            self.data,
            self.header,
            self.vchannels,
            self.wcs,
            self.beam
        )


#class Model(Cube, FITSUtils, Mcmc):
"""
beam : None or `~radio_beam.Beam`, optional
            - If None, the beam information is extracted from the header of the input FITS file.
            - If `~radio_beam.Beam` object, it uses this 
"""


#from .disc2d import Cube
class OldData(Cube, FITSUtils):
    def __init__(self, filename):
        """
        Initialise Data object. Inherits `~discminer.disc2d.Cube` properties and methods.
        
        Parameters
        ----------
        filename : path-like 
            Path to FITS datacube

        """
        self.fileroot = os.path.expanduser(filename).split(".fits")[0]
        cube_spe = SpectralCube.read(filename)
        cube_vel = cube_spe.with_spectral_unit(
            u.km / u.s,
            velocity_convention="radio",
            rest_value=cube_spe.header["RESTFRQ"] * u.Hz,
        )
        # in km/s, remove .value to keep astropy units
        self.vchannels = cube_vel.spectral_axis.value
        self.header = cube_vel.header
        self.data = cube_vel.hdu.data.squeeze()
        self.wcs = WCS(self.header)
        # Assuming (nchan, nx, nx), nchan should be equal to cube_vel.spectral_axis.size
        self.nchan, self.nx, _ = np.shape(self.data)

        self.beam = Beam.from_fits_header(self.header)  # radio_beam object
        self.beam_info = self.beam  # backwards compatibility
        self._init_beam_kernel()  # Get 2D Gaussian kernel from beam
        self._init_cube()

    def _init_cube(self):
        super().__init__(
            self.nchan,
            self.vchannels,
            self.data,
            beam=self.beam,
            beam_kernel=self.beam_kernel,
        )

    def _init_beam_kernel(self):
        """
        Compute 2D Gaussian kernel in pixels from beam info.

        """
        sigma2fwhm = np.sqrt(8 * np.log(2))
        # pixel size in CUNIT2 units
        pix_scale = np.abs(self.header["CDELT2"]) * u.Unit(self.header["CUNIT2"])
        x_stddev = ((self.beam.major / pix_scale) / sigma2fwhm).decompose().value
        y_stddev = ((self.beam.minor / pix_scale) / sigma2fwhm).decompose().value
        beam_angle = (90 * u.deg + self.beam.pa).to(u.radian).value
        self.beam_kernel = Gaussian2DKernel(x_stddev, y_stddev, beam_angle)


    def _writefits(self, logkeys=None, tag="", **kwargs):
        """
        Write fits file
            
        Paramaters
        ----------
        logkeys : list of str, optional
            List of keys to append in the output file name. If multiple keys are pased the order in the output name is maintained.
            Only the keys present in the cube header will be added to the output file name.

        tag : str, optional
            String to add at the end of the output filename.

        kwargs : keyword arguments
            Additional keyword arguments to pass to `~astropy.io.fits.writeto` function.
           
        """
        ktag = ""
        if logkeys is not None:
            for key in logkeys:
                if key in self.header and self.header[key]:
                    ktag += "_" + key.lower()

        self.fileroot += ktag + tag
        fits.writeto(self.fileroot + ".fits", self.data, header=self.header, **kwargs)

    def convert_to_tb(self, planck=True, writefits=True, tag="", **kwargs): #Should be in FitsUtils class, useful for Model too.
        """
        Convert intensity to brightness temperature in units of Kelvin.

        Parameters
        ----------
        planck : bool, optional
            If True, it uses the full Planck law to make the conversion, else it uses the Rayleigh-Jeans approximation. 

        writefits : bool, optional
            If True it creates a FITS file with the new intensity data and header.
        
        tag : str, optional
            String to add at the end of the output filename.

        kwargs : keyword arguments
            Additional keyword arguments for `~astropy.io.fits.writeto` function.
           
        """
        hdrkey = "CONVTB"
        hdrcard = "Converted to Tb by DISCMINER"
        kwargs_io = dict(overwrite=True)  # Default kwargs
        kwargs_io.update(kwargs)

        Tb = self._convert_to_tb(self.data,
                                 self.header,
                                 self.beam,
                                 planck=planck,
                                 writefits=writefits,
                                 tag=tag)                                 
        
        self.data = Tb
        self.header["BUNIT"] = "K"

        self.wcs = WCS(self.header)
        self.header[hdrkey] = (True, hdrcard)
        if writefits:
            self._writefits(logkeys=[hdrkey], tag=tag, **kwargs_io)
        self._init_cube()  # Redo Cube

        
    def downsample(
        self, npix, method=np.median, kwargs_method={}, writefits=True, tag="", **kwargs
    ):
        """
        Downsample data cube to reduce spatial correlations between pixels and/or to save computational costs in the modelling. 

        Parameters
        ----------
        npix : int
            Number of pixels to downsample. For example, if npix=3, the downsampled-pixel will have an area of 3x3 original-pixels

        method : func, optional
            function to compute downsampling

        kwargs_method : keyword arguments
            Additional keyword arguments to pass to the input ``method``.

        writefits : bool, optional
            If True it creates a FITS file with the new intensity data and header.
        
        tag : str, optional
            String to add at the end of the output filename.

        kwargs : keyword arguments
            Additional keyword arguments to pass to `~astropy.io.fits.writeto` function.
           
        """
        hdrkey = "DOWNSAMP"
        hdrcard = "Downsampled by DISCMINER"
        kwargs_io = dict(overwrite=True)  # Default kwargs
        kwargs_io.update(kwargs)

        nchan, nx0 = self.nchan, self.nx
        nx = int(round(nx0 / npix))

        if npix > 1:
            av_data = np.zeros((nchan, nx, nx))  # assuming ny = nx
            progress = Tools._progress_bar
            di = npix
            dj = npix
            print("Averaging %dx%d pixels from data cube..." % (di, dj))
            for k in range(nchan):
                progress(int(100 * k / nchan))
                for i in range(nx):
                    for j in range(nx):
                        av_data[k, j, i] = method(
                            self.data[k, j * dj : j * dj + dj, i * di : i * di + di],
                            **kwargs_method
                        )
            progress(100)

            self.nx = nx
            self.data = av_data

            # nf: number of pix between centre of first pix in the original img and centre of first downsampled pix
            if npix % 2:  # if odd
                nf = (npix - 1) / 2.0
            else:
                nf = 0.5 + (npix / 2 - 1)

            # will be the new CRPIX1 and CRPIX2 (origin is 1,1, not 0,0)                
            refpix = 1.0  
            # coords of reference pixel, using old pixels info
            refpixval = aputils.pixel_to_skycoord(nf, nf, self.wcs)

            CDELT1, CDELT2 = self.header["CDELT1"], self.header["CDELT2"]
            # equivalent to CRVAL1 - CDELT1 * (CRPIX1 - 1 - nf) but using right projection
            self.header["CRVAL1"] = refpixval.ra.value
            self.header["CRVAL2"] = refpixval.dec.value
            self.header["CDELT1"] = CDELT1 * npix
            self.header["CDELT2"] = CDELT2 * npix
            self.header["CRPIX1"] = refpix
            self.header["CRPIX2"] = refpix
            self.header["NAXIS1"] = nx
            self.header["NAXIS2"] = nx

            self.wcs = WCS(self.header)
            self._init_beam_kernel()
            # keeping track of changes to original cube
            self.header[hdrkey] = (True, hdrcard)
            if writefits:
                self._writefits(logkeys=[hdrkey], tag=tag, **kwargs_io)
            self._init_cube()  # Redo Cube

        else:
            print("npix is <= 1, no average was performed...")

    def clip(
        self,
        npix=0,
        icenter=None,
        jcenter=None,
        channels={"interval": None, "indices": None},
        writefits=True,
        tag="",            
        **kwargs
    ):
        """
        Clip spatial and/or velocity axes of the data cube. The extent of the clipped region would be 
        ``[icenter-npix, icenter+npix]`` along the first spatial axis (normally RA), and ``[jcenter-npix, jcenter+npix]`` along the second spatial axis (normally DEC).
        
        See the description of the argument ``channels`` below for details on how the velocity axis is clipped.

        Parameters
        ----------
        npix : int
            Number of pixels to clip above and below (and to the left and right of) the reference centre of the data (icenter, jcenter). 
            The total number of pixels after clipping would be 2*npix on each spatial axis.
        
        icenter, jcenter : int, optional
            Reference centre for the clipped window. Must be integers referred to pixel ids from the input data. 
            If None, the reference centre is determined from the input header as ``icenter=int(header['CRPIX1'])`` and ``jcenter=int(header['CRPIX2'])``
        
        channels : {"interval" : [i0, i1]} or {"indices" : [i0, i1,..., in]}, optional
            Dictionary of indices to clip velocity channels from data. If both entries are None, all velocity channels are considered.         
 
            * If 'interval' is defined, velocity channels between *i0* and *i1* indices are considered, *i1* inclusive.          
            * If 'indices' is defined, only velocity channels corresponding to the input indices will be considered.            
            * If both entries are set, only 'interval' will be taken into account.
        
        writefits : bool, optional
            If True it creates a FITS file with the new intensity data and header.

        tag : str, optional
            String to add at the end of the output filename.

        kwargs : keyword arguments
            Additional keyword arguments to pass to `~astropy.io.fits.writeto` function.
        
        """
        hdrkey = "CLIPPED"
        hdrcard = "Clipped by DISCMINER"
        kwargs_io = dict(overwrite=True)  # Default kwargs
        kwargs_io.update(kwargs)

        if icenter is not None:
            icenter = int(icenter)
        else:  # Assume reference centre at the centre of the image
            icenter = int(0.5 * self.header["NAXIS1"] + 1)
        if jcenter is not None:
            jcenter = int(jcenter)
        else:
            jcenter = int(0.5 * self.header["NAXIS2"] + 1)

        if channels is None:
            channels = {}
        if "interval" not in channels.keys():
            channels.update({"interval": None})
        if "indices" not in channels.keys():
            channels.update({"indices": None})
        if channels["interval"] is not None:
            i0, i1 = channels["interval"]
            idchan = np.arange(i0, i1 + 1).astype(int)
        elif channels["indices"] is not None:
            idchan = np.asarray(channels["indices"]).astype(int)
            warnings.warn(
                "Note that if you select channels that are not regularly spaced the header of the output fits file will not reflect this information and therefore external analysis tools such as CASA or DS9 will not display the velocity information correctly.",
            )
        else:
            idchan = slice(None)

        self.data = self.data[idchan]
        self.vchannels = self.vchannels[idchan]

        if npix > 0:
            self.data = self.data[
                :, jcenter - npix : jcenter + npix, icenter - npix : icenter + npix
            ]  # data shape: (NAXIS3, NAXIS2, NAXIS1)

            # CRVAL1, CRVAL2 = self.header["CRVAL1"], self.header["CRVAL2"]
            # CRPIX1, CRPIX2 = self.header["CRPIX1"], self.header["CRPIX2"]
            # CDELT1, CDELT2 = self.header["CDELT1"], self.header["CDELT2"]
            # the following is wrong because the projection is not Cartesian: self.header["CRVAL1"] = CRVAL1 + (icenter - CRPIX1) * CDELT1,
            # a proper conversion using wcs must be done.
            newcr = aputils.pixel_to_skycoord(icenter, jcenter, self.wcs)
            self.header["CRVAL1"] = newcr.ra.value
            self.header["CRVAL2"] = newcr.dec.value
            self.header["CRPIX1"] = npix + 1.0
            self.header["CRPIX2"] = npix + 1.0

        self.nchan, self.nx, _ = self.data.shape
        self.header["NAXIS1"] = self.nx
        self.header["NAXIS2"] = self.nx
        self.header["NAXIS3"] = self.nchan
        self.header["CRPIX3"] = 1.0
        self.header["CRVAL3"] = self.vchannels[0]
        self.header["CDELT3"] = self.vchannels[1] - self.vchannels[0]

        self.wcs = WCS(self.header)
        self.header[hdrkey] = (True, hdrcard)
        if writefits:
            self._writefits(logkeys=[hdrkey], tag=tag, **kwargs_io)
        self._init_cube()  # Redo Cube

        
