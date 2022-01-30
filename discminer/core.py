import numpy as np
from .disc2d import Cube, Tools
from astropy import units as apu
from astropy.io import fits
from spectral_cube import SpectralCube
import warnings


class Data(Cube):
    def __init__(self, filename, intensity_norm=1e3):
        """
        Initialise Data object. It inherits `~discminer.disc2d.Cube` properties and methods.
        
        Parameters
        ----------
        filename : str
            Name of data FITS file
        intensity_norm : float, optional
            Factor to normalise data intensity. 
            Default is 1e3, to convert Jy to mJy.
        """
        self.fileroot = filename.split(".fits")[0]
        cube_spe = SpectralCube.read(filename)
        cube_vel = cube_spe.with_spectral_unit(
            apu.km / apu.s,
            velocity_convention="radio",
            rest_value=cube_spe.header["RESTFRQ"] * apu.Hz,
        )
        self.vchannels = (
            cube_vel.spectral_axis.value
        )  # in km/s, remove .value to keep astropy units
        self.nchan = cube_vel.spectral_axis.size
        self.header = cube_vel.header
        self.data = cube_vel.hdu.data * intensity_norm
        """
        if beam:
            self.beam = beam
        if beam_kernel:
            self.beam_kernel = beam_kernel
        if isinstance(tb, dict):
            if tb["nu"] and tb["beam"]:
                self.data = Tools.get_tb(
                    self.data, tb["nu"], tb["beam"], full=tb["full"]
                )
        """
        super().__init__(self.nchan, self.vchannels, self.data)

    @staticmethod
    def downsample(data, frac_pixels, av_method=np.median):
        """
        data: datacube with shape (nchan, nx0, ny0)
        frac_pixels: number of pixels to average
        av_method: function to compute average
        """
        nchan, nx0, ny0 = np.shape(data)
        nx = int(round(nx0 / frac_pixels))
        ny = int(round(ny0 / frac_pixels))
        av_data = np.zeros((nchan, nx, ny))
        progress = Tools._progress_bar
        if frac_pixels > 1:
            di = frac_pixels
            dj = frac_pixels
            print("Averaging %dx%d pixels from data cube..." % (di, dj))
            for k in range(nchan):
                progress(int(100 * k / nchan))
                for i in range(nx):
                    for j in range(ny):
                        av_data[k, i, j] = av_method(
                            data[k, i * di : i * di + di, j * dj : j * dj + dj]
                        )
            progress(100)
            return av_data
        else:
            print("frac_pixels is <= 1, no average was performed...")
            return data

    def clip(
        self,
        npix=0,
        icenter=None,
        jcenter=None,
        channels={"interval": None, "indices": None},
        tag="",
        overwrite=True,
        **kwargs
    ):
        """
        Clip spatial and/or velocity axes of the data cube. The extent of the clipped region will be 
        ``[icenter-npix, icenter+npix]`` along the first spatial axis, and ``[jcenter-npix, jcenter+npix]`` along the second spatial axis.

        Parameters
        ----------
        npix : int
            Number of pixels to clip above and below (and to the left and right of) the reference centre of the data (icenter, jcenter).
        
        icenter, jcenter : int, optional
            Reference centre for the clipped window. Must be integers referred to pixel ids from the input data. 
            If None, the reference centre is determined from the input header as ``icenter=int(header['CRPIX1'])`` and ``jcenter=int(header['CRPIX2'])``
        
        channels : {"interval" : [i0, i1]} or {"indices" : [i0, i1,..., in]}, optional
            Dictionary of indices to clip velocity channels from data. If both entries are None, all velocity channels are considered.         
 
            If 'interval' is defined, velocity channels between *i0* and *i1* indices are considered, *i1* inclusive.
          
            If 'indices' is defined, only velocity channels corresponding to the input indices will be considered.
            
            If both entries are set, only 'interval' will be taken into account.
        
        tag : str, optional
            Add string at the end of the output filename.

        overwrite : bool, optional
            If True, overwrite the output FITS file if it exists. Default is True.

        kwargs : keyword arguments
            Additional keyword arguments to pass to `~astropy.io.fits.writeto` function.
        
        """
        if icenter is not None:
            icenter = int(icenter)
        else:
            icenter = int(self.header["CRPIX1"])
        if jcenter is not None:
            jcenter = int(jcenter)
        else:
            jcenter = int(self.header["CRPIX2"])

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
                "Note that if you select channels that are not regularly spaced the header of the output fits file will not reflect this information and thus external analysis tools such as CASA or DS9 will fail at showing details from velocity channels correctly."
            )
        else:
            idchan = slice(None)

        self.data = self.data[
            idchan, icenter - npix : icenter + npix, jcenter - npix : jcenter + npix
        ]
        self.vchannels = self.vchannels[idchan]
        self.nchan = len(self.vchannels)

        self.header["CRPIX1"] = npix + 1
        self.header["CRPIX2"] = npix + 1
        self.header["CRPIX3"] = 1.0
        self.header["CRVAL3"] = self.vchannels[0]

        super().__init__(self.nchan, self.vchannels, self.data)  # Redo Cube object

        fits.writeto(
            self.fileroot + "_clipped%s.fits" % tag,
            self.data,
            header=self.header,
            overwrite=overwrite,
            **kwargs
        )
