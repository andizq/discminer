from .plottools import (get_discminer_cmap,
                        make_up_ax,
                        mod_major_ticks,
                        mod_minor_ticks,
                        mask_cmap_interval)

from .tools.fit_kernel import fit_gaussian, fit_twocomponent, fit_onecomponent
from .tools.utils import FrontendUtils, InputError, _JSON
from .rail import Contours

from astropy.convolution import Gaussian2DKernel
from astropy import units as u
from astropy import constants as apc
from astropy.io import fits
from astropy.wcs import utils as aputils, WCS

from radio_beam import Beam

from scipy import ndimage

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Cursor, Slider, RectangleSelector, TextBox

from collections.abc import Iterable
import numpy as np
import copy
import os
import warnings
from packaging.version import Version

MPL_VERSION = Version(matplotlib.__version__)

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 22

_progress_bar = FrontendUtils._progress_bar
_break_line = FrontendUtils._break_line
path_icons = FrontendUtils.path_icons

class Cube(_JSON):
    def __init__(self, data, header, vchannels, dpc, beam=None, filename="./cube.fits", disc=None, mol='12co', kind=['mask'], parfile=None):
        """
        Initialise Cube object.
        
        Parameters
        ----------
        data : array_like, shape(nchan, nx, nx)
            Input datacube with intensity information.
        
        header : `~astropy.io.fits.header.Header`
            Header of the input datacube.
        
        vchannels : array_like
            Velocity channels associated to the input datacube.

        beam : None or `~radio_beam.beam.Beam`, optional
            If not None, it must be a `~radio_beam` object specifying beam size and beam position angle. 
        
        filename : path-like, optional 
            The name of this file would be used as the root name of new FITS files which would be created if the original datacube is modified via `~discminer.cube.Cube` methods.

        """
        self.data = copy.copy(data)
        self.header = copy.copy(header)
        self.vchannels = copy.copy(vchannels)
        self.beam = beam
        self.wcs = WCS(self.header)
        # Assuming (nchan, nx, nx); nchan should be equal to cube_vel.spectral_axis.size
        self.nchan, self.ny, self.nx = np.shape(data) #ny: nrows, nx: ncols
        self.dpc = dpc
        self.disc = disc
        self.mol = mol
        self.kind = kind
        self._init_sky_extent()
        
        if isinstance(beam, Beam):
            self._init_beam_kernel()  # Get 2D Gaussian kernel from beam
            bmaj = self.beam.major
            bmin = self.beam.minor
            bpa = self.beam.pa

        elif beam is None:
            self.beam_size = None
            self.beam_area = None
            self.beam_kernel = None
            self.beam_area_arcsecs = None
            bmaj = 0.0*u.deg
            bmin = 0.0*u.deg
            bpa = 0.0*u.deg
            
        else:
            raise InputError(beam, "beam must be either None or radio_beam.Beam object")

        self._interactive = self._cursor
        self._interactive_path = self._curve

        #Init JSON metadata
        _JSON.__init__(self, init_metadata=dict(disc=disc, mol=mol, kind=kind), parfile=parfile)

        self.filename = filename #init and run setter; potentially variable for the same obj
        
        self.json_metadata = {
            'dpc': dpc,
            'bmaj': bmaj,
            'bmin': bmin,
            'bpa': bpa,
            'bmaj_au': self.beam_size,
            'downsamp': 1,
            'clipped': False
        }
        self._update_json_metadata()
        
    @property
    def filename(self): 
        return self._filename
          
    @filename.setter 
    def filename(self, name): 
        self.fileroot = os.path.expanduser(name).split(".fits")[0]        
        self.json_metadata = {'file_data': name}
        self._filename = name        
        
    def _update_json_metadata(self):
        v0 = self.vchannels[0]*u.km/u.s
        v1 = self.vchannels[-1]*u.km/u.s
        wchan = (v1-v0)/(self.nchan-1)

        self.json_metadata = {
            'npix': self.nx,
            'nchan': self.nchan,
            'wchan': wchan.value,
            'v0': v0.value,
            'v1': v1.value
        }
        
    def _init_sky_extent(self):
        self.pix_size = np.abs(self.header["CDELT1"]) * u.Unit(self.header["CUNIT1"])
        pix_rad = self.pix_size.to(u.radian)
        pix_au = (self.dpc*np.tan(pix_rad)).to(u.au)
        self.xsky = ((self.nx-1) * pix_au/2.0).value
        self.ysky = ((self.ny-1) * pix_au/2.0).value        
        
    def _init_beam_kernel(self):
        """
        Compute 2D Gaussian kernel in pixels from beam info.

        References
        ----------
           * https://en.wikipedia.org/wiki/Gaussian_function

           * https://science.nrao.edu/facilities/vla/proposing/TBconv  
        """
        sigma2fwhm = np.sqrt(8 * np.log(2))
        x_stddev = ((self.beam.major / self.pix_size) / sigma2fwhm).decompose().value
        y_stddev = ((self.beam.minor / self.pix_size) / sigma2fwhm).decompose().value
        beam_angle = (90 * u.deg + self.beam.pa).to(u.radian).value
        self.beam_kernel = Gaussian2DKernel(x_stddev, y_stddev, beam_angle)
        # area of gaussian beam in pixels**2
        self.beam_area = 2*np.pi*x_stddev*y_stddev
        # area of gaussian beam in arcsecs**2
        bmaj = self.beam.major.to(u.arcsecond).value
        bmin = self.beam.minor.to(u.arcsecond).value        
        self.beam_area_arcsecs = 2*np.pi * (bmaj * bmin) / sigma2fwhm**2        
        #self.beam_area_arcsecs = np.pi * (bmaj * bmin) / (4 * np.log(2))
        self.beam_size = bmaj*self.dpc.to('pc').value * u.au
        
    @staticmethod
    def _channel_picker(channels, warn_hdr=True):
        """
        Returns channel indices based on interval of indices or indices specified

        Parameters
        ----------
        channels : {"interval" : [i0, i1]} or {"indices" : [i0, i1,..., in]}, optional
            Dictionary of indices to clip velocity channels from data. If both entries are None, all velocity channels are considered.         
 
            * If 'interval' is defined, velocity channels between *i0* and *i1* indices are considered, *i1* inclusive.          
            * If 'indices' is defined, only velocity channels corresponding to the input indices will be considered.            
            * If both entries are set, only 'interval' will be taken into account.
        """
        
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
            if warn_hdr: warnings.warn(
                    "Note that if you select channels that are not regularly spaced the header of the output fits file will not reflect this information and therefore external analysis tools such as CASA or DS9 will not display the velocity information correctly.",
            )
        else:
            idchan = slice(None)

        return idchan
    
    def writefits(self, logkeys=None, tag="", **kwargs):
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
        kwargs_io = dict(overwrite=True)  # Default kwargs
        kwargs_io.update(kwargs)

        if logkeys is not None:
            for key in logkeys:
                if key in self.header and self.header[key]:
                    ktag += "_" + key.lower()
        else:
            if len(tag)>0:
                if tag[0]!='_':
                    tag = '_'+tag                    
        
        self.fileroot += ktag + tag
        fits.writeto(self.fileroot + ".fits", self.data, header=self.header, **kwargs_io)

    def convert_to_tb(self, planck=False, writefits=True, tag="", **kwargs):
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

        if planck:
            hdrkey = "PLANCKTB"
        else:
            hdrkey = "CONVTB"
            
        hdrcard = "Converted to Tb by DISCMINER"
        kwargs_io = dict(overwrite=True)  # Default kwargs
        kwargs_io.update(kwargs)

        I = self.data * u.Unit(self.header["BUNIT"]).to("beam-1 Jy")
        nu = self.header["RESTFRQ"]  # in Hz
        # beam_area: C*bmin['']*bmaj[''] * (dist[pc])**2 --> beam area in au**2 units
        # beam solid angle: beam_area/(dist[m])**2.
        #  dist**2 cancels out with beamarea's dist[pc]**2 from conversion of bmaj, bmin to mks units.
        beam_solid = u.au.to("m") ** 2 * self.beam_area_arcsecs / u.pc.to("m") ** 2
        Jy_to_SI = 1e-26
        c_h = apc.h.value
        c_c = apc.c.value
        c_k_B = apc.k_B.value

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

        self.data = Tb
        self.header["BUNIT"] = "K"

        self.wcs = WCS(self.header)
        self.header[hdrkey] = (True, hdrcard)
        if writefits:
            self.writefits(logkeys=[hdrkey], tag=tag, **kwargs_io)

    def downsample(
            self, npix, method=np.median, kwargs_method={}, writefits=True, tag="", crpix_to_center=False, **kwargs
    ):
        """
        Downsample datacube to reduce spatial correlations between pixels and/or to save computational costs in the modelling. 

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
            di = npix
            dj = npix
            print("Averaging %dx%d pixels from datacube..." % (di, dj))
            for k in range(nchan):
                _progress_bar(int(100 * k / nchan))
                for i in range(nx):
                    for j in range(nx):
                        av_data[k, j, i] = method(
                            self.data[k, j * dj : j * dj + dj, i * di : i * di + di],
                            **kwargs_method
                        )
            _progress_bar(100); print('\n')

            self.nx = nx
            self.ny = nx
            self.data = av_data

            # nf: number of pix between centre of first pix in the original img and centre of first downsampled pix
            if npix % 2:  # if odd
                nf = (npix - 1) / 2.0
            else:
                nf = 0.5 + (npix / 2 - 1)

            # will be the new CRPIX1 and CRPIX2 (origin is 1,1, not 0,0)
            refpix = 1
            # coords of reference pixel, using old pixels info
            refpixval = aputils.pixel_to_skycoord(nf, nf, self.wcs) #referred to 0-based coords

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

            if crpix_to_center:
                icenter = int(0.5 * nx)
                refpixval = aputils.pixel_to_skycoord(icenter, icenter, self.wcs) #referred to new, downsampled wcs
                self.header["CRVAL1"] = refpixval.ra.value
                self.header["CRVAL2"] = refpixval.dec.value
                self.header["CRPIX1"] = icenter+1
                self.header["CRPIX2"] = icenter+1
                self.wcs = WCS(self.header)

            self._init_sky_extent()
            
            if isinstance(self.beam, Beam):
                self._init_beam_kernel()  # Get 2D Gaussian kernel from beam
            elif self.beam is None:
                pass
            else:
                raise InputError(self.beam, "beam must be either None or radio_beam.Beam object")

            # keeping track of changes to original cube
            self.header[hdrkey] = (True, hdrcard)
            if writefits:
                self.writefits(logkeys=[hdrkey], tag=tag, **kwargs_io)

        else:
            print("npix is <= 1, no average was performed...")
            
        self._update_json_metadata()
        self.json_metadata = {'downsamp': npix}
        
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
            Reference centre for the clipped window. Must be integers referred to pixel ids from the input data. Must be one-based to keep fits header convention
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

        nsky_min = np.min([self.nx, self.ny])
        if 2*npix>nsky_min:
            raise InputError((2*npix, nsky_min), 'Number of pixels to clip must be less than '\
                             'the number of pixels in the smallest spatial axis of the input data cube.')
                             
        hdrkey = "CLIPPED"
        hdrcard = "Clipped by DISCMINER"
        kwargs_io = dict(overwrite=True)  # Default kwargs
        kwargs_io.update(kwargs)

        print ("Clipping datacube...")
        
        if icenter is not None:
            icenter = int(icenter)
        else:  # Assume reference centre at the centre of the image
            icenter = int(0.5 * self.header["NAXIS1"] + 1.0)
        if jcenter is not None:
            jcenter = int(jcenter)
        else:
            jcenter = int(0.5 * self.header["NAXIS2"] + 1.0)

        idchan = self._channel_picker(channels)
        self.data = self.data[idchan]
        self.vchannels = self.vchannels[idchan]

        # data shape: (NAXIS3, NAXIS2, NAXIS1)
        if npix > 0:
            self.data = self.data[
                :, jcenter - npix : jcenter + npix, icenter - npix : icenter + npix
            ]
            # The following line is wrong because the RA axis is distorted by the DEC:
            #  self.header["CRVAL1"] = CRVAL1 + (icenter - CRPIX1) * CDELT1.
            #   A proper conversion must use wcs:
            newcr = aputils.pixel_to_skycoord(icenter-1, jcenter-1, self.wcs) #-1 to go back to 0-based convention
            self.header["CRVAL1"] = newcr.ra.value
            self.header["CRVAL2"] = newcr.dec.value
            self.header["CRPIX1"] = npix #one-based
            self.header["CRPIX2"] = npix

        self.nchan, self.ny, self.nx = self.data.shape
        self.header["NAXIS1"] = self.nx
        self.header["NAXIS2"] = self.ny
        self.header["NAXIS3"] = self.nchan
        self.header["CRPIX3"] = 1
        self.header["CRVAL3"] = self.vchannels[0]
        self.header["CDELT3"] = self.vchannels[1] - self.vchannels[0]

        self.wcs = WCS(self.header)
        self.header[hdrkey] = (True, hdrcard)
        if writefits:
            self.writefits(logkeys=[hdrkey], tag=tag, **kwargs_io)

        self._update_json_metadata()
        self.json_metadata = {'clipped': True}

    def make_moments(self, method='gaussian', kind='mask', writefits=True, overwrite=True, parcube=True, writecomp=True, tag="", **kwargs_method):
        """
        Make moment maps from line profile observables.

        Parameters
        ----------
        method : str, optional
            Type of kernel to be fitted to each pixel's line profile.
          
            * If 'gaussian' or 'gauss', fit a single Gaussian profile to the line and return [peak, centroid, linewidth], [dpeak, dcent, dlinewidth].
            * If 'bell', fit a single bell profile to the line and return [peak, centroid, linewidth, lineslope], [dpeak, dcent, dlinewidth, dlineslope].
            * If 'doublegaussian', fit double Gaussian profile to the line and return [peak_up, centroid_up, linewidth_up], [dpeak_up, dcent_up, dlinewidth_up], [peak_low, centroid_low, linewidth_low], [dpeak_low, dcent_low, dlinewidth_low].
            * If 'doublebell', fit double Bell profile to the line and return [peak_up, centroid_up, linewidth_up], [dpeak_up, dcent_up, dlinewidth_up], [peak_low, centroid_low, linewidth_low], [dpeak_low, dcent_low, dlinewidth_low].

        kind : str, optional
            'sum' or 'mask' upper and lower surface line profiles for 'doublegaussian' and 'doublebell' methods. 
           
             * If 'sum', the composite line profile will be the simple sum of the two-component profiles.
             * If 'mask', the composite line profile intensity in each pixel and velocity channel will be that of the brighter emission surface (between upper and lower surface).

        """

        hdr_int = copy.copy(self.header)        
        hdr_vel = copy.copy(self.header)
        hdr_ls = copy.copy(self.header)

        hdr_vel["BUNIT"] = "km/s"
        hdr_vel["BTYPE"] = "Velocity"
        hdr_ls["BUNIT"] = ""
        hdr_ls["BTYPE"] = "Line slope"

        hdr_par = copy.copy(self.header) #parcube
        hdr_comp = copy.copy(self.header) # n components matrix        

        hdr_par["BUNIT"] = hdr_int['BUNIT']+" km/s km/s"
        hdr_par["BTYPE"] = "Various"
        hdr_comp["BUNIT"] = ""
        hdr_comp["BTYPE"] = "Number of line components"
        
        kwargs_io_int = dict(overwrite=overwrite, header=hdr_int)
        kwargs_io_vel = dict(overwrite=overwrite, header=hdr_vel)
        kwargs_io_ls = dict(overwrite=overwrite, header=hdr_ls)
        kwargs_io_par = dict(overwrite=overwrite, header=hdr_par)
        kwargs_io_comp = dict(overwrite=overwrite, header=hdr_comp)                 

        if len(tag)>0:
            if tag[0]!='_':
                tag = '_'+tag
                
        if method in ['gaussian', 'gauss', 'bell']:
            _break_line()
            kwargs_m = dict(method=method, lw_chans=1.0, sigma_fit=None)
            kwargs_m.update(kwargs_method)            
            moments = fit_onecomponent(self, **kwargs_m)
            moments, n_fit =  moments[:-1], moments[-1]

            if writefits:
                filenames = [
                    [
                        'peakintensity',
                        'velocity',
                        'linewidth',
                    ],
                    [
                        'delta_peakintensity',
                        'delta_velocity',
                        'delta_linewidth',
                    ]
                ]

                if method == 'bell':
                    for i in range(2):
                        filenames[i] += [filenames[i][-1].replace('width', 'slope')] #Add lineslope to filenames
                    ntypes = 4
                    
                else: 
                    ntypes = 3

                if 'fit_continuum' in kwargs_m:
                    if kwargs_m['fit_continuum']:
                        filenames[0] += ['continuum']
                        filenames[1] += ['delta_continuum']
                        ntypes+=1
                        
                print ('Writing moments into FITS files...')
                for j in range(2):
                    for i in range(ntypes):                    
                        if i==0:
                            kwargs = kwargs_io_int
                        else:
                            kwargs = kwargs_io_vel
                        fits.writeto(filenames[j][i]+'_%s%s.fits'%(method, tag), moments[j][i], **kwargs)
                    
            if parcube:
                print ('Writing parcubes into FITS files...')
                fnameparcubes = ['parcube', 'delta_parcube']                
                kwargs = kwargs_io_par                
                for j in range(2):
                    fits.writeto(fnameparcubes[j]+'_%s%s.fits'%(method, tag), moments[j], **kwargs)
                    
            _break_line()
        
        elif method in ['doublegaussian', 'doublebell']:
            _break_line()
            kwargs_m = dict(lw_chans=1.0, lower2upper=1.0, sigma_fit=None, method=method, kind=kind)
            kwargs_m.update(kwargs_method)
            moments = fit_twocomponent(self,  **kwargs_m)
            moments, n_fit =  moments[:-1], moments[-1]
            
            if writefits:
                filenames = [
                    [
                        'peakintensity_up',
                        'velocity_up',
                        'linewidth_up'
                    ],
                    [
                        'delta_peakintensity_up',
                        'delta_velocity_up',
                        'delta_linewidth_up'
                    ],
                    [    
                        'peakintensity_low',
                        'velocity_low',
                        'linewidth_low'
                    ],
                    [
                        'delta_peakintensity_low',
                        'delta_velocity_low',
                        'delta_linewidth_low'
                    ]
                ]

                if method == 'doublebell':
                    for i in range(4):
                        filenames[i] += [filenames[i][-1].replace('width', 'slope')] #Add lineslope to filenames
                    ntypes = 4
                    
                elif method == 'doublegaussian':
                    ntypes = 3
                
                print ('Writing moments into FITS files...')
                for j in range(4):
                    for i in range(ntypes):
                        if i==0:
                            kwargs = kwargs_io_int
                        elif i==3:
                            kwargs = kwargs_io_ls
                        else:
                            kwargs = kwargs_io_vel
                        fits.writeto(filenames[j][i]+'_%s_%s%s.fits'%(method, kind, tag), moments[j][i], **kwargs)

            if parcube:
                print ('Writing parcubes into FITS files...')
                fnameparcubes = ['parcube_up', 'delta_parcube_up', 'parcube_low', 'delta_parcube_low']                
                kwargs = kwargs_io_par
                for j in range(4):
                    fits.writeto(fnameparcubes[j]+'_%s_%s%s.fits'%(method, kind, tag), moments[j], **kwargs)
            _break_line()
            
        else:
            raise InputError(
                method, "method/kernel requested is not supported by this version of discminer. Available methods: ['gaussian', 'doublegaussian', 'bell', doublebell]"
            )

        self.n_fit = n_fit
        if writecomp:
            kwargs = kwargs_io_comp
            fits.writeto('fit_line_components'+'_%s_%s%s.fits'%(method, kind, tag), n_fit, **kwargs)        

        return moments #A1, c1, lw1, [Ls1], dA1, dc1, dlw1, [dLs1], A2, c2, lw2, [Ls2], dA2, dc2, dlw2, [dLs2]

        
    # *********************************
    # FUNCTIONS FOR INTERACTIVE WINDOWS
    # *********************************
    @property
    def interactive(self):
        return self._interactive

    @interactive.setter
    def interactive(self, func):
        print("Setting interactive function to", func)
        self._interactive = func

    @property
    def interactive_path(self):
        return self._interactive_path

    @interactive_path.setter
    def interactive_path(self, func):
        print("Setting interactive_path function to", func)
        self._interactive_path = func

    def _surface(self, ax, surface_from, **kwargs):
        surface_from.make_emission_surface(ax, **kwargs)

    def plot_beam(self, ax, projection=None, **kwargs_ellipse):
        if self.beam is None:
            return 0
        kwargs=dict(lw=1,fill=True,fc="gray",ec="k")
        kwargs.update(kwargs_ellipse)
            
        if projection is None: #Assume plot in units of au
            dpc = self.dpc.to('pc').value
            bmaj = self.beam.major.to(u.arcsecond).value
            bmin = self.beam.minor.to(u.arcsecond).value
            bmaj*=dpc
            bmin*=dpc
        elif projection=='wcs': #plot in units of pixels (using wcs is merely decorative)
            bmaj = self.beam_kernel.model.x_fwhm
            bmin = self.beam_kernel.model.y_fwhm
            
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        dx = np.abs(xlim[1]-xlim[0])
        dy = np.abs(ylim[1]-ylim[0])
        xbeam, ybeam = xlim[0]+0.07*dx, ylim[0]+0.07*dy

        ellipse = patches.Ellipse(
            xy=(xbeam, ybeam),
            angle=90 + self.beam.pa.value,
            width=bmaj,
            height=bmin,
            transform=ax.transData,
            **kwargs
        )
        ax.add_artist(ellipse)

    def _check_cubes_shape(self, compare_cubes):
        for cube in compare_cubes:
            if cube.data.shape != self.data.shape:
                raise InputError(
                    compare_cubes, "Input cubes for comparison must have the same shape"
                )

    # *************************************
    # SHOW SPECTRUM ON PIXEL and WITHIN BOX
    # *************************************
    def _plot_spectrum_box(
        self,
        x0,
        x1,
        y0,
        y1,
        ax,
        extent=None,
        compare_cubes=[],
        stat_func=np.mean,
        **kwargs
    ):

        kwargs_spec = dict(where="mid", linewidth=2.5, label=r"x0:%d,x1:%d" % (x0, x1))
        kwargs_spec.update(kwargs)
        v0, v1 = self.vchannels[0], self.vchannels[-1]

        if extent is None:
            j0, i0 = int(x0), int(y0)
            j1, i1 = int(x1), int(y1)
        else:
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j0 = int(nx * (x0 - extent[0]) / dx)
            i0 = int(ny * (y0 - extent[2]) / dy)
            j1 = int(nx * (x1 - extent[0]) / dx)
            i1 = int(ny * (y1 - extent[2]) / dy)

        slice_cube = self.data[:, i0:i1, j0:j1]
        spectrum = np.array([stat_func(chan) for chan in slice_cube])
        ncubes = len(compare_cubes)

        if ncubes > 0:
            slice_comp = [compare_cubes[i].data[:, i0:i1, j0:j1] for i in range(ncubes)]
            cubes_spec = [
                np.array([stat_func(chan) for chan in slice_comp[i]])
                for i in range(ncubes)
            ]

        if np.logical_or(np.isinf(spectrum), np.isnan(spectrum)).all():
            return False
        else:
            plot_spec = ax.step(self.vchannels, spectrum, **kwargs_spec)
            if ncubes > 0:
                alpha = 0.2
                dalpha = -alpha / ncubes
                for i in range(ncubes):
                    ax.fill_between(
                        self.vchannels,
                        cubes_spec[i],
                        color=plot_spec[0].get_color(),
                        step="mid",
                        alpha=alpha,
                    )
                    alpha += dalpha
            else:
                ax.fill_between(
                    self.vchannels,
                    spectrum,
                    color=plot_spec[0].get_color(),
                    step="mid",
                    alpha=0.2,
                )
            return plot_spec

    def _box(self, fig, ax, extent=None, compare_cubes=[], stat_func=np.mean, **kwargs):
        def onselect(eclick, erelease):
            if eclick.inaxes is ax[0]:
                plot_spec = self._plot_spectrum_box(
                    eclick.xdata,
                    erelease.xdata,
                    eclick.ydata,
                    erelease.ydata,
                    ax[1],
                    extent=extent,
                    compare_cubes=compare_cubes,
                    stat_func=stat_func,
                    **kwargs
                )

                if plot_spec:
                    print("startposition: (%f, %f)" % (eclick.xdata, eclick.ydata))
                    print("endposition  : (%f, %f)" % (erelease.xdata, erelease.ydata))
                    print("used button  : ", eclick.button)
                    xc, yc = eclick.xdata, eclick.ydata  # Left, bottom corner
                    dx, dy = (
                        erelease.xdata - eclick.xdata,
                        erelease.ydata - eclick.ydata,
                    )
                    rect = patches.Rectangle(
                        (xc, yc),
                        dx,
                        dy,
                        lw=2,
                        edgecolor=plot_spec[0].get_color(),
                        facecolor="none",
                    )
                    ax[0].add_patch(rect)
                    ax[1].legend()
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        def toggle_selector(event):
            print("Key pressed...")
            if event.key in ["C", "c"] and toggle_selector.RS.active:
                print("RectangleSelector deactivated.")
                toggle_selector.RS.set_active(False)
            if event.key in ["A", "a"] and not toggle_selector.RS.active:
                print("RectangleSelector activated.")
                toggle_selector.RS.set_active(True)

        if MPL_VERSION >= Version("3.5"):
            props_kwarg = "props"
        else:
            props_kwarg = "rectprops"

        selector_kwargs = {
            props_kwarg: dict(facecolor="0.7", edgecolor="k", alpha=0.3, fill=True)
        }

        toggle_selector.RS = RectangleSelector(
            ax[0],
            onselect,
            button=[1],
            **selector_kwargs,
        )
        cid = fig.canvas.mpl_connect("key_press_event", toggle_selector)
        return toggle_selector.RS

    def _plot_spectrum_cursor(self, x, y, ax, extent=None, compare_cubes=[], **kwargs):

        kwargs_spec = dict(where="mid", linewidth=2.5, label=r"%d,%d" % (x, y))
        kwargs_spec.update(kwargs)

        if extent is None:
            j, i = int(x), int(y)
        else:
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j = int(nx * (x - extent[0]) / dx)
            i = int(ny * (y - extent[2]) / dy)

        spectrum = self.data[:, i, j]
        v0, v1 = self.vchannels[0], self.vchannels[-1]

        if np.logical_or(np.isinf(spectrum), np.isnan(spectrum)).all():
            return False
        else:
            # plot_fill = ax.fill_between(self.vchannels, spectrum, alpha=0.1)
            plot_spec = ax.step(self.vchannels, spectrum, **kwargs_spec)
            ncubes = len(compare_cubes)
            if ncubes > 0:
                alpha = 0.2
                dalpha = -alpha / ncubes
                for cube in compare_cubes:
                    ax.step(self.vchannels,
                            cube.data[:, i, j],
                            color='k', #plot_spec[0].get_color(),
                            where="mid",
                            lw=0.5
                    )
                    ax.fill_between(
                        self.vchannels,
                        cube.data[:, i, j],
                        color=plot_spec[0].get_color(),
                        step="mid",
                        alpha=alpha,
                    )
                    alpha += dalpha
            else:
                ax.fill_between(
                    self.vchannels,
                    spectrum,
                    color=plot_spec[0].get_color(),
                    step="mid",
                    alpha=0.2,
                )
            return plot_spec

    def _cursor(self, fig, ax, extent=None, compare_cubes=[], **kwargs):
        def onclick(event):
            if event.button == 3:
                print("Right click. Disconnecting click event...")
                fig.canvas.mpl_disconnect(cid)
            elif event.inaxes is ax[0]:
                plot_spec = self._plot_spectrum_cursor(
                    event.xdata,
                    event.ydata,
                    ax[1],
                    extent=extent,
                    compare_cubes=compare_cubes,
                    **kwargs
                )
                if plot_spec is not None:
                    print(
                        "%s click: button=%d, xdata=%f, ydata=%f"
                        % (
                            "double" if event.dblclick else "single",
                            event.button,
                            event.xdata,
                            event.ydata,
                        )
                    )
                    ax[0].scatter(
                        event.xdata,
                        event.ydata,
                        marker="D",
                        s=50,
                        facecolor=plot_spec[0].get_color(),
                        edgecolor="k",
                    )
                    ax[1].legend(
                        frameon=False, handlelength=0.7, fontsize=MEDIUM_SIZE - 1
                    )
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        return cid

    def show(
            self,
            extent=None,
            chan_init=0,
            cube_init=0,
            compare_cubes=[],
            cursor_grid=True,
            cmap="gnuplot2_r",
            int_unit=r"Intensity [mJy beam$^{-1}$]",
            pos_unit="Offset [au]",
            vel_unit=r"km s$^{-1}$",
            show_beam=True,
            surface_from=None,
            kwargs_surface={},
            vmin = None,
            vmax = None,
            **kwargs
    ):

        self._check_cubes_shape(compare_cubes)
        v0, v1 = self.vchannels[0], self.vchannels[-1]
        dv = v1 - v0
        fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
        plt.subplots_adjust(wspace=0.25)
        ncubes = len(compare_cubes)

        y0, y1 = ax[1].get_position().y0, ax[1].get_position().y1
        axcbar = plt.axes([0.47, y0, 0.03, y1 - y0])
        max_data = np.nanmax([self.data] + [comp.data for comp in compare_cubes])
        ax[0].set_xlabel(pos_unit)
        ax[0].set_ylabel(pos_unit)
        ax[1].set_xlabel("l.o.s velocity [%s]" % vel_unit)
        mod_major_ticks(ax[0], axis="both", nbins=5)
        ax[0].tick_params(direction="out")
        ax[1].tick_params(direction="in", right=True, labelright=False, labelleft=False)
        axcbar.tick_params(direction="out")
        ax[1].set_ylabel(int_unit, labelpad=15)
        ax[1].yaxis.set_label_position("right")
        ax[1].set_xlim(v0 - 0.1, v1 + 0.1)

        if extent is None:
            extent = [-self.xsky, self.xsky, -self.ysky, self.ysky]            
        if vmin is None:
            vmin = -1 * max_data / 100
        if vmax is None:
            vmax = 0.7 * max_data
        
        ax[1].set_ylim(vmin, vmax)
        # ax[1].grid(lw=1.5, ls=':')
        cmapc = copy.copy(plt.get_cmap(cmap))
        cmapc.set_bad(color=(0.9, 0.9, 0.9))

        if cube_init == 0:
            img_data = self.data[chan_init]
        else:
            img_data = compare_cubes[cube_init - 1].data[chan_init]

        img = ax[0].imshow(
            img_data, cmap=cmapc, extent=extent, origin="lower", vmin=vmin, vmax=vmax,
        )
        cbar = plt.colorbar(img, cax=axcbar)
        img.cmap.set_under("w")
        current_chan = ax[1].axvline(
            self.vchannels[chan_init], color="black", lw=2, ls="--"
        )
        text_chan = ax[1].text(
            (self.vchannels[chan_init] - v0) / dv,
            1.02,  # Converting xdata coords to Axes coords
            "%4.1f %s" % (self.vchannels[chan_init], vel_unit),
            ha="center",
            color="black",
            transform=ax[1].transAxes,
        )

        if show_beam and self.beam_kernel:
            self.plot_beam(ax[0])
        
        if cursor_grid:
            cg = Cursor(ax[0], useblit=True, color="lime", linewidth=1.5)

        def get_interactive(func):
            return func(fig, ax, extent=extent, compare_cubes=compare_cubes, **kwargs)

        interactive_obj = [get_interactive(self.interactive)]

        # ***************
        # UPDATE FUNCS
        # ***************
        def update_chan(val):
            chan = int(val)
            vchan = self.vchannels[chan]
            img.set_data(self.data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan - v0) / dv)
            text_chan.set_text("%4.1f %s" % (vchan, vel_unit))
            fig.canvas.draw_idle()

        def update_cubes(val):
            i = int(slider_cubes.val)
            chan = int(slider_chan.val)
            vchan = self.vchannels[chan]
            if i == 0:
                img.set_data(self.data[chan])
            else:
                img.set_data(compare_cubes[i - 1].data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan - v0) / dv)
            text_chan.set_text("%4.1f %s" % (vchan, vel_unit))
            fig.canvas.draw_idle()

        def update_clim(expression):
            vmax = float(expression)
            img.norm.autoscale([vmin, vmax])
            ax[1].set_ylim(vmin, vmax)
            fig.canvas.draw_idle()

        def update_cmap(expression):
            if 'cmr.' in expression:
                try:
                    import cmasher as cmr
                except ImportError:
                    print ('cmasher library could not be found...')                    
            cmapc = copy.copy(plt.get_cmap(expression))
            cmapc.set_bad(color=(0.9, 0.9, 0.9))
            img.set_cmap(cmapc)
            img.cmap.set_under("w")                        
            fig.canvas.draw_idle()
            
        if ncubes > 0:
            axcubes = plt.axes([0.2, 0.90, 0.24, 0.025], facecolor="0.7")
            axchan = plt.axes([0.2, 0.95, 0.24, 0.025], facecolor="0.7")
            slider_cubes = Slider(
                axcubes,
                "Cube id",
                0,
                ncubes,
                valstep=1,
                valinit=cube_init,
                valfmt="%1d",
                color="dodgerblue",
            )
            slider_chan = Slider(
                axchan,
                "Channel",
                0,
                self.nchan - 1,
                valstep=1,
                valinit=chan_init,
                valfmt="%2d",
                color="dodgerblue",
            )
            slider_cubes.on_changed(update_cubes)
            slider_chan.on_changed(update_cubes)  # update_cubes works for both
        else:
            axchan = plt.axes([0.2, 0.9, 0.24, 0.05], facecolor="0.7")
            slider_chan = Slider(
                axchan,
                "Channel",
                0,
                self.nchan - 1,
                valstep=1,
                valinit=chan_init,
                valfmt="%2d",
                color="dodgerblue",
            )
            slider_chan.on_changed(update_chan)

        kwargs_def = dict(
            extent=extent,
            chan_init=chan_init,
            cube_init=cube_init,
            compare_cubes=compare_cubes,
            cursor_grid=cursor_grid,
            int_unit=int_unit,
            pos_unit=pos_unit,
            vel_unit=vel_unit,
            show_beam=show_beam,                
            surface_from=surface_from,
            kwargs_surface=kwargs_surface,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
        # *************
        # BUTTONS
        # *************
        def go2cursor(event):
            # If already cursor, return 0 and pass, else, exec cursor func
            if self.interactive == self._cursor:
                return 0
            interactive_obj[0].set_active(False)
            self.interactive = self._cursor
            interactive_obj[0] = get_interactive(self._cursor)

        def go2box(event):
            if self.interactive == self._box:
                return 0
            fig.canvas.mpl_disconnect(interactive_obj[0])
            self.interactive = self._box
            interactive_obj[0] = get_interactive(self._box)

        def go2path(event):
            print("Activating function to extract intensities along path...")
            plt.close()
            chan = int(slider_chan.val)
            if ncubes > 0:
                ci = int(slider_cubes.val)
            else:
                ci = 0
            kwargs_def.update({'chan_init': chan, 'cube_init': ci})                                
            self._show_path( **kwargs_def)

        def go2trash(event):
            print("Cleaning interactive figure...")
            plt.close()
            chan = int(slider_chan.val)
            if ncubes > 0:
                ci = int(slider_cubes.val)
            else:
                ci = 0
            kwargs_def.update({'chan_init': chan, 'cube_init': ci})                
            self.show( **kwargs_def)

        def go2surface(event):
            self._surface(ax[0], surface_from, **kwargs_surface)
            fig.canvas.draw()
            fig.canvas.flush_events()

        box_img = plt.imread(path_icons + "button_box.png")
        cursor_img = plt.imread(path_icons + "button_cursor.jpeg")
        path_img = plt.imread(path_icons + "button_path.png")
        trash_img = plt.imread(path_icons + "button_trash.jpg")
        surface_img = plt.imread(path_icons + "button_surface.png")

        axbcursor = plt.axes([0.05, 0.829, 0.05, 0.05])
        axbbox = plt.axes([0.05, 0.77, 0.05, 0.05])
        axbpath = plt.axes([0.05, 0.711, 0.05, 0.05], frameon=True, aspect="equal")
        axbtrash = plt.axes([0.05, 0.65, 0.05, 0.05], frameon=True, aspect="equal")

        bcursor = Button(axbcursor, "", image=cursor_img)
        bcursor.on_clicked(go2cursor)
        bbox = Button(axbbox, "", image=box_img)
        bbox.on_clicked(go2box)
        bpath = Button(axbpath, "", image=path_img, color="white", hovercolor="lime")
        bpath.on_clicked(go2path)
        btrash = Button(axbtrash, "", image=trash_img, color="white", hovercolor="lime")
        btrash.on_clicked(go2trash)

        if surface_from is not None:
            axbsurf = plt.axes([0.005, 0.809, 0.07, 0.07], frameon=True, aspect="equal")
            bsurf = Button(axbsurf, "", image=surface_img)
            bsurf.on_clicked(go2surface)

        if MPL_VERSION >= Version("3.5"):
            axbclim = plt.axes([0.05, 0.16, 0.04, 0.045], frameon=True)
            clim_box = TextBox(axbclim, 'clim', textalignment='center', color='lightcyan', label_pad=0.1)
            clim_box.on_submit(update_clim)
            clim_box.set_val(np.round(vmax, 2))

            axbcmap = plt.axes([0.05, 0.11, 0.074, 0.045], frameon=True)
            cmap_box = TextBox(axbcmap, 'cmap', textalignment='center', color='lightcyan', label_pad=0.05)
            cmap_box.on_submit(update_cmap)
            cmap_box.set_val(cmap)
        
        plt.show(block=True)

    def show_side_by_side(
            self,
            cube1,
            extent=None,
            chan_init=0,
            cursor_grid=True,
            cmap="gnuplot2_r",
            int_unit=r"Intensity [mJy beam$^{-1}$]",
            pos_unit="Offset [au]",
            vel_unit=r"km s$^{-1}$",
            show_beam=True,
            surface_from=None,
            kwargs_surface={},
            vmin=None,
            vmax=None,
            **kwargs
    ):

        compare_cubes = [cube1]
        self._check_cubes_shape(compare_cubes)

        v0, v1 = self.vchannels[0], self.vchannels[-1]
        dv = v1 - v0
        fig, ax = plt.subplots(ncols=3, figsize=(17, 5))
        plt.subplots_adjust(wspace=0.25)

        y0, y1 = ax[2].get_position().y0, ax[2].get_position().y1
        axcbar = plt.axes([0.63, y0, 0.015, y1 - y0])
        max_data = np.nanmax([self.data] + [comp.data for comp in compare_cubes])
        ax[0].set_xlabel(pos_unit)
        ax[1].set_xlabel(pos_unit)
        ax[0].set_ylabel(pos_unit)
        ax[2].set_xlabel("l.o.s velocity [%s]" % vel_unit)
        mod_major_ticks(ax[0], axis="both", nbins=5)
        mod_major_ticks(ax[1], axis="both", nbins=5)        
        ax[0].tick_params(direction="out")
        ax[1].tick_params(direction="out")        
        ax[2].tick_params(direction="in", right=True, labelright=False, labelleft=False)
        axcbar.tick_params(direction="out")
        ax[2].set_ylabel(int_unit, labelpad=15)
        ax[2].yaxis.set_label_position("right")
        ax[2].set_xlim(v0 - 0.1, v1 + 0.1)

        if extent is None:
            extent = [-self.xsky, self.xsky, -self.ysky, self.ysky]            
        if vmin is None:
            vmin = -1 * max_data / 100
        if vmax is None:
            vmax = 0.7 * max_data

        ax[2].set_ylim(vmin, vmax)
        cmapc = copy.copy(plt.get_cmap(cmap))
        cmapc.set_bad(color=(0.9, 0.9, 0.9))

        img = ax[0].imshow(
            self.data[chan_init],
            cmap=cmapc,
            extent=extent,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        img1 = ax[1].imshow(
            cube1.data[chan_init],
            cmap=cmapc,
            extent=extent,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = plt.colorbar(img, cax=axcbar)
        img.cmap.set_under("w")
        img1.cmap.set_under("w")
        current_chan = ax[2].axvline(
            self.vchannels[chan_init], color="black", lw=2, ls="--"
        )
        text_chan = ax[2].text(
            (self.vchannels[chan_init] - v0) / dv,
            1.02,  # Converting xdata coords to Axes coords
            "%4.1f %s" % (self.vchannels[chan_init], vel_unit),
            ha="center",
            color="black",
            transform=ax[2].transAxes,
        )

        if show_beam and self.beam_kernel:
            self.plot_beam(ax[0])
            self.plot_beam(ax[1])

        if cursor_grid:
            cg = Cursor(ax[0], useblit=True, color="lime", linewidth=1.5)

        def get_interactive(func):
            return func(
                fig,
                [ax[0], ax[2]],
                extent=extent,
                compare_cubes=compare_cubes,
                **kwargs
            )

        interactive_obj = [get_interactive(self.interactive)]

        # ***************
        # UPDATE FUNCS
        # ***************
        def update_chan(val):
            chan = int(val)
            vchan = self.vchannels[chan]
            img.set_data(self.data[chan])
            img1.set_data(cube1.data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan - v0) / dv)
            text_chan.set_text("%4.1f %s" % (vchan, vel_unit))
            fig.canvas.draw_idle()

        def update_clim(expression):
            vmax = float(expression)
            img.norm.autoscale([vmin, vmax])
            img1.norm.autoscale([vmin, vmax])            
            ax[2].set_ylim(vmin, vmax)
            fig.canvas.draw_idle()

        def update_cmap(expression):
            if 'cmr.' in expression:
                try:
                    import cmasher as cmr
                except ImportError:
                    print ('cmasher library could not be found...')                    
            cmapc = copy.copy(plt.get_cmap(expression))
            cmapc.set_bad(color=(0.9, 0.9, 0.9))
            img.set_cmap(cmapc)
            img1.set_cmap(cmapc)            
            img.cmap.set_under("w")
            img1.cmap.set_under("w")                                    
            fig.canvas.draw_idle()
            
        ncubes = len(compare_cubes)
        axchan = plt.axes([0.2, 0.9, 0.24, 0.05], facecolor="0.7")
        slider_chan = Slider(
            axchan,
            "Channel",
            0,
            self.nchan - 1,
            valstep=1,
            valinit=chan_init,
            valfmt="%2d",
            color="dodgerblue",
        )
        slider_chan.on_changed(update_chan)

        # *************
        # BUTTONS
        # *************
        def go2cursor(event):
            if self.interactive == self._cursor:
                return 0
            interactive_obj[0].set_active(False)
            self.interactive = self._cursor
            interactive_obj[0] = get_interactive(self.interactive)

        def go2box(event):
            if self.interactive == self._box:
                return 0
            fig.canvas.mpl_disconnect(interactive_obj[0])
            self.interactive = self._box
            interactive_obj[0] = get_interactive(self.interactive)

        def go2trash(event):
            print("Cleaning interactive figure...")
            plt.close()
            chan = int(slider_chan.val)
            self.show_side_by_side(
                cube1,
                extent=extent,
                chan_init=chan,
                cursor_grid=cursor_grid,
                int_unit=int_unit,
                pos_unit=pos_unit,
                vel_unit=vel_unit,
                show_beam=show_beam,                
                surface_from=surface_from,
                kwargs_surface=kwargs_surface,
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )

        def go2surface(event):
            self._surface(ax[0], surface_from, **kwargs_surface)
            self._surface(ax[1], surface_from, **kwargs_surface)            
            fig.canvas.draw()
            fig.canvas.flush_events()

        box_img = plt.imread(path_icons + "button_box.png")
        cursor_img = plt.imread(path_icons + "button_cursor.jpeg")
        trash_img = plt.imread(path_icons + "button_trash.jpg")
        surface_img = plt.imread(path_icons + "button_surface.png")
        axbcursor = plt.axes([0.05, 0.779, 0.05, 0.05])
        axbbox = plt.axes([0.05, 0.72, 0.05, 0.05])
        axbtrash = plt.axes([0.05, 0.661, 0.05, 0.05], frameon=True, aspect="equal")
        bcursor = Button(axbcursor, "", image=cursor_img)
        bcursor.on_clicked(go2cursor)
        bbox = Button(axbbox, "", image=box_img)
        bbox.on_clicked(go2box)
        btrash = Button(axbtrash, "", image=trash_img, color="white", hovercolor="lime")
        btrash.on_clicked(go2trash)

        if surface_from is not None:
            axbsurf = plt.axes([0.005, 0.759, 0.07, 0.07], frameon=True, aspect="equal")
            bsurf = Button(axbsurf, "", image=surface_img)
            bsurf.on_clicked(go2surface)

        if MPL_VERSION >= Version("3.5"):
            axbclim = plt.axes([0.05, 0.16, 0.04, 0.045], frameon=True)
            clim_box = TextBox(axbclim, 'clim', textalignment='center', color='lightcyan', label_pad=0.1)
            clim_box.on_submit(update_clim)
            clim_box.set_val(np.round(vmax, 2))

            axbcmap = plt.axes([0.05, 0.11, 0.074, 0.045], frameon=True)
            cmap_box = TextBox(axbcmap, 'cmap', textalignment='center', color='lightcyan', label_pad=0.05)
            cmap_box.on_submit(update_cmap)
            cmap_box.set_val(cmap)
            
        plt.show(block=True)

    # ************************
    # SHOW SPECTRUM ALONG PATH
    # ************************
    def _plot_spectrum_path(
        self,
        fig,
        ax,
        xa,
        ya,
        chan,
        color_list=[],
        extent=None,
        plot_color=None,
        compare_cubes=[],
        **kwargs_curve
    ):

        if xa is None:
            return 0

        if extent is None:
            j = xa.astype(int)
            i = ya.astype(int)
        else:
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j = (nx * (xa - extent[0]) / dx).astype(int)
            i = (ny * (ya - extent[2]) / dy).astype(int)

        pix_ids = np.arange(len(i))
        path_val = self.data[chan, i, j]

        if plot_color is None:
            plot_path = ax[1].step(pix_ids, path_val, where="mid", lw=2, **kwargs_curve)
            plot_color = plot_path[0].get_color()
            color_list.append(plot_color)
        else:
            plot_path = ax[1].step(
                pix_ids, path_val, where="mid", lw=2, color=plot_color, **kwargs_curve
            )
        path_on_cube = ax[0].plot(xa, ya, color=plot_color, lw=2, **kwargs_curve)

        ncubes = len(compare_cubes)
        if ncubes > 0:
            alpha = 0.2
            dalpha = -alpha / ncubes
            for cube in compare_cubes:
                ax[1].fill_between(
                    pix_ids,
                    cube.data[chan, i, j],
                    color=plot_color,
                    step="mid",
                    alpha=alpha,
                )
                alpha += dalpha
        else:
            ax[1].fill_between(
                pix_ids, path_val, color=plot_color, step="mid", alpha=0.1
            )

        #fig.canvas.draw()
        #fig.canvas.flush_events()

    def _curve(
        self,
        fig,
        ax,
        xa_list=[],
        ya_list=[],
        color_list=[],
        extent=None,
        click=True,
        compare_cubes=[],
        **kwargs
    ):
        kwargs_curve = dict(linewidth=2.5)
        kwargs_curve.update(kwargs)

        xa, ya = None, None
        xm = [None]
        ym = [None]

        def mouse_move(event):
            xm[0] = event.xdata
            ym[0] = event.ydata

        def toggle_selector(event):
            toggle_selector.RS.set_active(True)

        def onselect(eclick, erelease):
            if eclick.inaxes is ax[0]:
                # Must correct if click and realease are not right by comparing with current pos of mouse.
                if xm[0] < erelease.xdata:
                    eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
                if ym[0] < erelease.ydata:
                    eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
                x0, y0 = eclick.xdata, eclick.ydata
                x1, y1 = erelease.xdata, erelease.ydata
                xa = np.linspace(x0, x1, 50)
                ya = np.linspace(y0, y1, 50)
                print("startposition: (%.1f, %.1f)" % (x0, y0))
                print("endposition  : (%.1f, %.1f)" % (x1, y1))
                print("used button  : ", eclick.button)
                xa_list.append(xa)
                ya_list.append(ya)
                self._plot_spectrum_path(
                    fig,
                    ax,
                    xa,
                    ya,
                    self._chan_path,
                    color_list=color_list,
                    extent=extent,
                    compare_cubes=compare_cubes,
                    **kwargs
                )
                #color_cycle = next(ax[0]._get_lines.prop_cycler)['color']
                ax[0].scatter(x0, y0, s=30, lw=1.5, edgecolor=color_list[-1], facecolor='none')
                
        if MPL_VERSION >= Version("3.5"):
            props_kwarg = "props"
        else:
            props_kwarg = "rectprops"

        selector_kwargs = {
            props_kwarg: dict(facecolor="0.7", edgecolor="k", alpha=0.3, fill=True)
        }

        if click:
            toggle_selector.RS = RectangleSelector(
                ax[0],
                onselect,                
                **selector_kwargs,
            )
                
        cid = fig.canvas.mpl_connect("key_press_event", toggle_selector)
        fig.canvas.mpl_connect("motion_notify_event", mouse_move)
        return cid

    def _show_path(
            self,
            extent=None,
            chan_init=0,
            cube_init=0,
            compare_cubes=[],
            cursor_grid=True,
            cmap="gnuplot2_r",
            int_unit=r"Intensity [mJy beam$^{-1}$]",
            pos_unit="au",
            vel_unit=r"km s$^{-1}$",
            show_beam=True,
            surface_from=None,
            kwargs_surface={},
            vmin = None,
            vmax = None,
            **kwargs
    ):

        self._check_cubes_shape(compare_cubes)
        v0, v1 = self.vchannels[0], self.vchannels[-1]
        dv = v1 - v0
        fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
        plt.subplots_adjust(wspace=0.25)
        ncubes = len(compare_cubes)
        self._chan_path = chan_init

        y0, y1 = ax[1].get_position().y0, ax[1].get_position().y1
        axcbar = plt.axes([0.47, y0, 0.03, y1 - y0])
        max_data = np.nanmax([self.data] + [comp.data for comp in compare_cubes])
        ax[0].set_xlabel(pos_unit)
        ax[0].set_ylabel(pos_unit)
        mod_major_ticks(ax[0], axis="both", nbins=5)
        ax[0].tick_params(direction="out")

        ax[1].tick_params(direction="in", right=True, labelright=False, labelleft=False)
        axcbar.tick_params(direction="out")
        ax[1].set_xlabel("Cell id along path")
        ax[1].set_ylabel(int_unit, labelpad=15)
        ax[1].yaxis.set_label_position("right")

        if extent is None:
            extent = [-self.xsky, self.xsky, -self.ysky, self.ysky]            
        if vmin is None:
            vmin = -1 * max_data / 100
        if vmax is None:
            vmax = 0.7 * max_data

        ax[1].set_ylim(vmin, vmax)
        # ax[1].grid(lw=1.5, ls=':')
        cmapc = copy.copy(plt.get_cmap(cmap))
        cmapc.set_bad(color=(0.9, 0.9, 0.9))

        if cube_init == 0:
            img_data = self.data[chan_init]
        else:
            img_data = compare_cubes[cube_init - 1].data[chan_init]

        img = ax[0].imshow(
            img_data, cmap=cmapc, extent=extent, origin="lower", vmin=vmin, vmax=vmax,
        )
        cbar = plt.colorbar(img, cax=axcbar)
        text_chan = ax[1].text(
            0.15,
            1.04,  # Converting xdata coords to Axes coords
            r"v$_{\rm chan}$=%4.1f %s" % (self.vchannels[chan_init], vel_unit),
            ha="center",
            color="black",
            transform=ax[1].transAxes,
        )

        if show_beam and self.beam_kernel:
            self.plot_beam(ax[0])
        
        if cursor_grid:
            cg = Cursor(ax[0], useblit=True, color="lime", linewidth=1.5)
        box_img = plt.imread(path_icons + "button_box.png")
        cursor_img = plt.imread(path_icons + "button_cursor.jpeg")

        xa_list, ya_list, color_list = [], [], []

        def get_interactive(func, click=True):
            cid = func(
                fig,
                ax,
                xa_list=xa_list,
                ya_list=ya_list,
                color_list=color_list,
                extent=extent,
                click=click,
                compare_cubes=compare_cubes,
                **kwargs
            )
            return cid

        interactive_obj = [get_interactive(self.interactive_path)]

        # ***************
        # UPDATE FUNCS
        # ***************
        def update_chan(val):
            chan = int(val)
            vchan = self.vchannels[chan]
            self._chan_path = chan

            if ncubes > 0:
                ci = int(slider_cubes.val)
                if ci == 0:
                    img.set_data(self.data[chan])
                else:
                    img.set_data(compare_cubes[ci - 1].data[chan])
            else:
                img.set_data(self.data[chan])

            for line in ax[1].get_lines():
                line.remove()
            for i in range(
                len(xa_list)
            ):  # Needs to be done more than once for some (memory) reason
                for mcoll in ax[1].collections:
                    mcoll.remove()

            text_chan.set_text(r"v$_{\rm chan}$=%4.1f %s" % (vchan, vel_unit))
            for i in range(len(xa_list)):
                if xa_list[i] is not None:
                    self._plot_spectrum_path(
                        fig,
                        ax,
                        xa_list[i],
                        ya_list[i],
                        chan,
                        extent=extent,
                        plot_color=color_list[i],
                        compare_cubes=compare_cubes,
                        **kwargs
                    )
            fig.canvas.draw_idle()

        def update_cubes(val):
            ci = int(val)
            chan = int(slider_chan.val)
            vchan = self.vchannels[chan]
            if ci == 0:
                img.set_data(self.data[chan])
            else:
                img.set_data(compare_cubes[ci - 1].data[chan])
            fig.canvas.draw_idle()

        def update_clim(expression):
            vmax = float(expression)
            img.norm.autoscale([vmin, vmax])
            ax[1].set_ylim(vmin, vmax)
            fig.canvas.draw_idle()

        def update_cmap(expression):
            if 'cmr.' in expression:
                try:
                    import cmasher as cmr
                except ImportError:
                    print ('cmasher library could not be found...')                    
            cmapc = copy.copy(plt.get_cmap(expression))
            cmapc.set_bad(color=(0.9, 0.9, 0.9))
            img.set_cmap(cmapc)
            img.cmap.set_under("w")                        
            fig.canvas.draw_idle()
            
        if ncubes > 0:
            axcubes = plt.axes([0.2, 0.90, 0.24, 0.025], facecolor="0.7")
            axchan = plt.axes([0.2, 0.95, 0.24, 0.025], facecolor="0.7")
            slider_cubes = Slider(
                axcubes,
                "Cube id",
                0,
                ncubes,
                valstep=1,
                valinit=cube_init,
                valfmt="%1d",
                color="dodgerblue",
            )
            slider_chan = Slider(
                axchan,
                "Channel",
                0,
                self.nchan - 1,
                valstep=1,
                valinit=chan_init,
                valfmt="%2d",
                color="dodgerblue",
            )
            slider_cubes.on_changed(update_cubes)
            slider_chan.on_changed(update_chan)

        else:
            axchan = plt.axes([0.2, 0.9, 0.24, 0.05], facecolor="0.7")
            slider_chan = Slider(
                axchan,
                "Channel",
                0,
                self.nchan - 1,
                valstep=1,
                valinit=chan_init,
                valfmt="%2d",
                color="dodgerblue",
            )
            slider_chan.on_changed(update_chan)

        # *************
        # BUTTONS
        # *************
        def go2show(event):
            print("Returning to intensity vs velocity plot...")
            plt.close()
            chan = int(slider_chan.val)
            if ncubes > 0:
                ci = int(slider_cubes.val)
            else:
                ci = 0
            self.show(
                extent=extent,
                chan_init=chan,
                cube_init=ci,
                compare_cubes=compare_cubes,
                cursor_grid=cursor_grid,
                int_unit=int_unit,
                pos_unit=pos_unit,
                vel_unit=vel_unit,
                show_beam=show_beam,
                surface_from=surface_from,
                kwargs_surface=kwargs_surface,
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )

        def go2trash(event):
            print("Cleaning interactive figure...")
            plt.close()
            chan = int(slider_chan.val)
            if ncubes > 0:
                ci = int(slider_cubes.val)
            else:
                ci = 0
            self._show_path(
                extent=extent,
                chan_init=chan,
                cube_init=ci,
                compare_cubes=compare_cubes,
                cursor_grid=cursor_grid,
                int_unit=int_unit,
                pos_unit=pos_unit,
                vel_unit=vel_unit,
                show_beam=show_beam,
                surface_from=surface_from,
                kwargs_surface=kwargs_surface,
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )

        def go2surface(event):
            self._surface(ax[0], surface_from, **kwargs_surface)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        return_img = plt.imread(path_icons + "button_return.png")
        axbretu = plt.axes([0.043, 0.711, 0.0635, 0.0635], frameon=True, aspect="equal")
        bretu = Button(axbretu, "", image=return_img, color="white", hovercolor="lime")
        bretu.on_clicked(go2show)

        trash_img = plt.imread(path_icons + "button_trash.jpg")
        axbtrash = plt.axes([0.05, 0.65, 0.05, 0.05], frameon=True, aspect="equal")
        btrash = Button(axbtrash, "", image=trash_img, color="white", hovercolor="lime")
        btrash.on_clicked(go2trash)

        surface_img = plt.imread(path_icons + "button_surface.png")
        if surface_from is not None:
            axbsurf = plt.axes([0.005, 0.809, 0.07, 0.07], frameon=True, aspect="equal")
            bsurf = Button(axbsurf, "", image=surface_img)
            bsurf.on_clicked(go2surface)

        if MPL_VERSION >= Version("3.5"):
            axbclim = plt.axes([0.05, 0.16, 0.04, 0.045], frameon=True)
            clim_box = TextBox(axbclim, 'clim', textalignment='center', color='lightcyan', label_pad=0.1)
            clim_box.on_submit(update_clim)
            clim_box.set_val(np.round(vmax, 2))

            axbcmap = plt.axes([0.05, 0.11, 0.074, 0.045], frameon=True)
            cmap_box = TextBox(axbcmap, 'cmap', textalignment='center', color='lightcyan', label_pad=0.05)
            cmap_box.on_submit(update_cmap)
            cmap_box.set_val(cmap)
            
        plt.show(block=True)

    # *************
    # MAKE GIF
    # *************
    def make_gif(
        self,
        folder="./gif/",
        extent=None,
        velocity2d=None,
        unit=r"Brightness Temperature [K]",
        gif_command="convert -delay 10 *int2d* cube_channels.gif",
    ):
        cwd = os.getcwd()
        if folder[-1] != "/":
            folder += "/"
        os.system("mkdir %s" % folder)
        max_data = np.max(self.data)

        clear_list, coll_list = [], []
        fig, ax = plt.subplots()
        contour_color = "red"
        cmap = plt.get_cmap("binary")
        cmap.set_bad(color=(0.9, 0.9, 0.9))
        ax.plot(
            [None],
            [None],
            color=contour_color,
            linestyle="--",
            linewidth=2,
            label="Upper surface",
        )
        ax.plot(
            [None],
            [None],
            color=contour_color,
            linestyle=":",
            linewidth=2,
            label="Lower surface",
        )
        ax.set_xlabel("au")
        ax.set_ylabel("au")

        for i in range(self.nchan):
            vchan = self.vchannels[i]
            int2d = ax.imshow(
                self.data[i], cmap=cmap, extent=extent, origin="lower", vmax=max_data
            )
            cbar = plt.colorbar(int2d)
            cbar.set_label(unit)
            if velocity2d is not None:
                vel_near = ax.contour(
                    velocity2d["upper"],
                    levels=[vchan],
                    colors=contour_color,
                    linestyles="--",
                    linewidths=1.3,
                    extent=extent,
                )
                vel_far = ax.contour(
                    velocity2d["lower"],
                    levels=[vchan],
                    colors=contour_color,
                    linestyles=":",
                    linewidths=1.3,
                    extent=extent,
                )
                coll_list = [vel_near, vel_far]
            text_chan = ax.text(
                0.7, 1.02, "%4.1f km/s" % vchan, color="black", transform=ax.transAxes
            )
            ax.legend(loc="upper left")
            plt.savefig(folder + "int2d_chan%04d" % i)

            clear_list = [cbar, int2d, text_chan]
            for obj in clear_list:
                obj.remove()
            for obj in coll_list:
                for coll in obj.collections:
                    coll.remove()
        plt.close()
        os.chdir(folder)
        print("Making movie...")
        os.system(gif_command)
        os.chdir(cwd)

        
    def make_channel_maps(self,
                          fig=None, ax=None,
                          channels={'interval': None, 'indices': None}, ncols=5,
                          unit_intensity=None, unit_coordinates=None,
                          fmt_channels='%.2f', annotate_channels=True,
                          observable='intensity', kind='attribute', cmap=None,
                          xlims=None, ylims=None, max_frac=0.8,
                          moving_center=False, center_from=None, zoom_factor=0.7,
                          fmt_cbar='%3d', mask_under=None,
                          contours_from=None, kwargs_contour={},
                          projection='wcs',
                          show_beam='all', kwargs_beam={},
                          **kwargs_contourf):
        
        try:
            vmin = kwargs_contourf['levels'][0]
            vmax = kwargs_contourf['levels'][-1]            
        #If levels are not provided, min and max data values are used as min and max boundaries for contourf levels
        except KeyError:
            vmin = np.nanmin(self.data)
            vmax = np.nanmax(self.data)
            if kind=='attribute':
                vmin, vmax = 0.0, max_frac*vmax
            elif kind=='residuals':
                max_val = np.max(np.abs([vmin, vmax]))
                vmin, vmax = -max_frac*max_val, max_frac*max_val

        if unit_intensity is None:
            try:
                unit_intensity = ' [%s]'%self.header['BUNIT']
            except KeyErorr:
                unit_intensity = ''
        else:
            unit_intensity = ' [%s]'%unit_intensity

        if unit_coordinates is None:
            unit_coordinates = ''
        else:
            unit_coordinates = ' [%s]'%unit_coordinates

        if cmap is None:
            cmap_chan = get_discminer_cmap(observable, kind=kind)
        else:
            cmap_chan = cmap

        if mask_under is not None:
            if kind=='attribute':
                mask_up = mask_under
                mask_down = 0
            elif kind=='residuals':
                mask_up = mask_under
                mask_down = -mask_under
            cmap_chan = mask_cmap_interval(cmap_chan, [vmin, vmax], [mask_down, mask_up], mask_color=(1,1,1,0), append=True)
            
        kwargs_cf = dict(cmap=cmap_chan, levels=np.linspace(vmin, vmax, 32))
        kwargs_cf.update(kwargs_contourf)

        kwargs_cc = dict(colors='k', linewidths=1.0)
        kwargs_cc.update(kwargs_contour)

        #*******************************
        #PREPARE PROJECTION AND CHANNELS
        #*******************************        
        if projection=='wcs':
            plot_projection=self.wcs.celestial

        else: #Assume plot in au
            plot_projection=None
            extent = [-self.xsky, self.xsky, -self.ysky, self.ysky]
            kwargs_cf.update(dict(extent=extent))
            kwargs_cc.update(dict(extent=extent))            
            
        idchan = self._channel_picker(channels, warn_hdr=False) #Warning on hdr not relevant here
        plot_data = self.data[idchan]
        plot_channels = self.vchannels[idchan]
        plot_nchan = len(plot_channels)

        if center_from is not None:
            plot_movcr = center_from[idchan]
            
        #*****************
        #FIGURE PROPERTIES
        #*****************        
        nrows = int(plot_nchan/ncols)
        if plot_nchan >= ncols:
            lastrow_ncols = plot_nchan%ncols
            if lastrow_ncols==0:
                lastrow_ncols=ncols
            else:
                nrows += 1
        else:
            lastrow_ncols = ncols = plot_nchan
            nrows += 1

        if fig is None:
            figx = 2*ncols
            figy = 2*nrows
            fig = plt.figure(figsize=(figx, figy))
        else:
            figx, figy = fig.get_size_inches()

        dw = 0.9/ncols
        dh = dw*figx/figy
        
        if ax is None:
            ax = [[fig.add_axes([0.05+i*dw, 0.97-(j+1)*dh-0.02*j, dw, dh], projection=plot_projection)
                   for i in range(ncols)] for j in range(nrows)]
        else:
            ax = np.atleast_2d(ax)
            
        for axi in ax[-1][lastrow_ncols:]:
            axi.set_visible(False)

        #*****************
        #BEAM
        #*****************        
        kwargs_bb = dict(lw=0.5, fill=True, fc='cyan', ec='k')
        kwargs_bb.update(kwargs_beam)
        
        #*****************
        #PLOT
        #*****************        
        ichan = 0
        im = []
        fakecolor = '0.6'

        for j in range(nrows):
            for i in range(ncols):

                axji = ax[j][i]
                data_i = plot_data[ichan]
                
                if moving_center:
                    if center_from is None:
                        data_nonoise = np.where(data_i>0.01*data_i.max(), data_i, 0)
                    else:
                        movcr_i = plot_movcr[ichan]
                        data_nonoise = np.where(movcr_i>0.01*movcr_i.max(), movcr_i, 0)
                    crow, ccol = tuple(np.int32(ndimage.center_of_mass(data_nonoise)))
                    border_row = np.min([np.abs(crow-nx), crow])
                    border_col = np.min([np.abs(ccol-nx), ccol])
                    border = int(zoom_factor*np.min([border_row, border_col]))
                    
                    x_i = xlist[ccol-border:ccol+border]
                    y_i = xlist[crow-border:crow+border]

                im.append(axji.contourf(data_i, **kwargs_cf))

                if contours_from is not None:
                    try:
                        vel2d, int2d, linew2d, lineb2d = contours_from.props #i.e. if Model instance provided
                        axji.contour(vel2d['upper'], levels=[plot_channels[ichan]], linestyles='-', **kwargs_cc)                        
                        axji.contour(vel2d['lower'], levels=[plot_channels[ichan]], linestyles='--', **kwargs_cc)
                    except AttributeError:
                        cca = np.moveaxis(np.atleast_3d(contours_from), -1, 0)
                        for cci in cca:
                            axji.contour(cci, levels=[plot_channels[ichan]], linestyles='-', **kwargs_cc)
            
                if annotate_channels:
                    axji.text(0.05,0.95, fmt_channels%plot_channels[ichan]+r'$^{\rm km/s}$', va='top', fontsize=SMALL_SIZE+2, transform=axji.transAxes)
                                                    
                if j==nrows-1 and i==0:
                    labelbottom, labelleft = True, True
                    if projection=='wcs': xlabel, ylabel = 'Right Ascension', 'Declination'
                    else: xlabel, ylabel = 'Offset%s'%unit_coordinates, 'Offset%s'%unit_coordinates
                    axji.set_xlabel(xlabel, labelpad=4, fontsize=MEDIUM_SIZE-3, color=fakecolor)                       
                    axji.set_ylabel(ylabel, labelpad=4, fontsize=MEDIUM_SIZE-3, color=fakecolor)

                else:
                    labelbottom, labelleft = False, False

                                        
                for axi in ['x', 'y']:
                    #WCS projection does not allow some props to be modified simultaneously in both axes
                    make_up_ax(axji, axis=axi, 
                               direction='in', 
                               pad=8, 
                               left=True, right=True,
                               bottom=True, top=True,
                               color=fakecolor,
                               labelcolor=fakecolor,
                               labelsize=SMALL_SIZE-1)
                axji.tick_params(axis='x', labelbottom=labelbottom, labeltop=False, labelcolor=fakecolor)
                axji.tick_params(axis='y', labelleft=labelleft)
                axji.tick_params(which='major', width=1.7, size=6.5)
                #axji.tick_params(which='minor', width=3.0, size=5.3)
                mod_major_ticks(axji, axis='x', nbins=3)
                mod_major_ticks(axji, axis='y', nbins=3)
                
                if self.beam is not None:
                    if show_beam=='all':
                        self.plot_beam(axji, projection=projection, **kwargs_bb)
                    elif isinstance(show_beam, Iterable):
                        bj, bi = show_beam
                        if j==bj and i==bi:
                            self.plot_beam(axji, projection=projection, **kwargs_bb)
                            
                if moving_center:
                    axji.set_xlim(np.min(x_i), np.max(x_i))
                    axji.set_ylim(np.min(y_i), np.max(y_i))                    
                else:
                    axji.set_xlim(xlims)
                    axji.set_ylim(ylims)
                            
                if ichan==plot_nchan-1:
                    break
                ichan+=1                

        #*****************
        #COLORBAR
        #*****************                    
        axc_pos = ax[0][-1].axes.get_position()
        axc_cbar = fig.add_axes([axc_pos.x1+0.005, axc_pos.y0, 0.08*dw, 0.5*dh])

        im_cbar = im[ncols-1]        
        cbar = plt.colorbar(im_cbar, cax=axc_cbar, format=fmt_cbar, orientation='vertical', 
                            ticks=np.linspace(im_cbar.levels[0], im_cbar.levels[-1], 5))

        if kind=='attribute':
            clabel = observable.split('_')[0].capitalize()+'%s'%unit_intensity
        elif kind=='residuals':
            clabel = kind.capitalize()+'%s'%unit_intensity
        cbar.set_label(clabel, fontsize=SMALL_SIZE+0, rotation=-90, labelpad=20)
        cbar.ax.tick_params(which='major', direction='in', width=1.7, size=3.8, pad=2, labelsize=SMALL_SIZE-1)
        cbar.ax.tick_params(which='minor', direction='in', width=1.7, size=2.3)
        mod_minor_ticks(cbar.ax)
        cbar.outline.set_linewidth(2) #Polygon patch; modifying cbar.ax spines does not work
        
        return fig, ax, im, cbar

    
