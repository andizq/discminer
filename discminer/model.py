import numpy as np
from .disc2d import Cube, Tools
from .grid import grid as dgrid
from astropy import units as u
from astropy import constants as apc
from astropy.io import fits
from astropy.wcs import utils as aputils, WCS
from astropy.convolution import Gaussian2DKernel
from spectral_cube import SpectralCube
from radio_beam import Beam
import warnings
import os


"""
intensity_norm : float, optional
Factor to normalise intensity. 
Default is 1e3, i.e. to convert to mJy/beam assuming input in Jy/beam
Will be deprecated when astropy units come into play.
"""
#Leave prototype as a separate method, not as a bool kwarg.
# the prototype method would then recieve grid parameters.
#  the run_mcmc method would receive a Data object from which the model grid will be created.

class Model(Cube):
    def __init__(self, dpc, prototype=True, subpixels=False, discgrid=None):
        """
        Initialise Model object. Inherits `~discminer.disc2d.Cube` properties and methods.

        Parameters
        ----------
        dpc : float
            Distance to the disc, in pc.

        prototype : bool, optional
            Computes a prototype model. Useful for quick inspection of channels given a set of parameters, which can then be used as seeds for the MCMC fit. Default is False.
        
        subpixels : bool, optional
            If velocity gradients within single pixels are large it is useful to generate subpixels for better sampling and computation of line-of-sight velocities. Default is False.
        
        discgrid : dict, optional
            In some cases the deprojected extent of the disc is larger than the grid of the skyplane, therefore having independent grids is sometimes needed to make sure that the deprojected disc properties do not show sharp boundaries.
            The provided dictionary should contain kwargs needed to compute a grid.  

        """
        Tools._print_logo()

        dpix = np.abs(self.header['CDELT2'])*3600
        dau = dpix * dpc
        nx = self.header['NAXIS1']
        xmax = ymax = (nx-1) * dau/2. * u.au.to("m")
        grid = dgrid(xmax, nx)
        self.skygrid = grid
        self.prototype = prototype
        if discgrid is None: discgrid = grid
        #discgrid should always exists, in general npix_disc != npix_sky, if npix_disc<npix_sky it's useful to use discgrid to save computing time, if npix_disc>npix_sky discgrid is needed for the emission to appear smooth instead of having sharp square boundaries.
        
        self._beam_info = False #Should be None; on if statements use isinstance(beam, Beam) instead
        self._beam_from = False #Should be deprecated
        self._beam_kernel = False #Should be None
        self._beam_area = False 
        if beam is not None: 
            self.beam_info, self.beam_kernel = Tools._get_beam_from(beam, dpix=discgrid['step'], **kwargs_beam)

        self._z_upper_func = General2d.z_cone
        self._z_lower_func = General2d.z_cone_neg
        self._velocity_func = General2d.keplerian
        self._intensity_func = General2d.intensity_powerlaw
        self._linewidth_func = General2d.linewidth_powerlaw
        self._lineslope_func = General2d.lineslope_powerlaw
        self._line_profile = General2d.line_profile_bell
        self._use_temperature = False
        self._use_full_channel = False
 
        x_true, y_true = discgrid['x'], discgrid['y']
        self.x_true, self.y_true = x_true, y_true
        self.phi_true = discgrid['phi']
        self.R_true = discgrid['R']         
        self.mesh = skygrid['meshgrid'] #disc grid will be interpolated onto this sky grid in make_model(). Must match data dims for mcmc. 
        
        self.R_1d = None #will be modified if selfgravity is considered

        if subpixels and isinstance(subpixels, int):
            if subpixels%2 == 0: subpixels+=1 #If input even becomes odd to contain pxl centre
            pix_size = discgrid['step']
            dx = dy = pix_size / subpixels
            centre = int(round((subpixels-1)/2.))
            centre_sq = int(round((subpixels**2-1)/2.))
            x_shift = np.arange(0, subpixels*dx, dx) - dx*centre
            y_shift = np.arange(0, subpixels*dy, dy) - dy*centre
            sub_x_true = [x_true + x0 for x0 in x_shift]
            sub_y_true = [y_true + y0 for y0 in y_shift]
            self.sub_R_true = [[hypot_func(sub_x_true[j], sub_y_true[i]) for j in range(subpixels)] for i in range(subpixels)]
            self.sub_phi_true = [[np.arctan2(sub_y_true[i], sub_x_true[j]) for j in range(subpixels)] for i in range(subpixels)]
            self.sub_x_true = sub_x_true
            self.sub_y_true = sub_x_true
            self.sub_dA = dx*dy
            self.pix_dA = pix_size**2
            self.sub_centre_id = centre_sq
            self.subpixels = subpixels
            self.subpixels_sq = subpixels**2
        else: self.subpixels=False

        #Get and print default parameters for default functions
        self.categories = ['velocity', 'orientation', 'intensity', 'linewidth', 'lineslope', 'height_upper', 'height_lower']

        self.mc_params = {'velocity': {'Mstar': True, 
                                       'vel_sign': 1,
                                       'vsys': 0},
                          'orientation': {'incl': True, 
                                          'PA': True,
                                          'xc': False,
                                          'yc': False},
                          'intensity': {'I0': True, 
                                        'p': True, 
                                        'q': False},
                          'linewidth': {'L0': True, 
                                        'p': True, 
                                        'q': 0.1}, 
                          'lineslope': {'Ls': False, 
                                        'p': False, 
                                        'q': False},
                          'height_upper': {'psi': True},
                          'height_lower': {'psi': True},
                          }
        
        self.mc_boundaries = {'velocity': {'Mstar': [0.05, 5.0],
                                           'vsys': [-10, 10],
                                           'Ec': [0, 300],
                                           'Rc': [50, 300],
                                           'gamma': [0.5, 2.0],
                                           'beta': [0, 1.0],
                                           'H0': [0.1, 20]},
                              'orientation': {'incl': [-np.pi/3, np.pi/3], 
                                              'PA': [-np.pi, np.pi],
                                              'xc': [-50, 50],
                                              'yc': [-50, 50]},
                              'intensity': {'I0': [0, 1000], 
                                            'p': [-10.0, 10.0], 
                                            'q': [0, 5.0]},
                              'linewidth': {'L0': [0.005, 5.0], 
                                            'p': [-5.0, 5.0], 
                                            'q': [-5.0, 5.0]},
                              'lineslope': {'Ls': [0.005, 20], 
                                            'p': [-5.0, 5.0], 
                                            'q': [-5.0, 5.0]},
                              'height_upper': {'psi': [0, np.pi/2]},
                              'height_lower': {'psi': [0, np.pi/2]}
                              }
         
        if prototype:
            self.params = {}
            for key in self.categories: self.params[key] = {}
            print ('Available categories for prototyping:', self.params)            
        else: 
            self.mc_header, self.mc_kind, self.mc_nparams, self.mc_boundaries_list, self.mc_params_indices = General2d._get_params2fit(self.mc_params, self.mc_boundaries)
            #print ('Default parameter header for mcmc fitting:', self.mc_header)
            #print ('Default parameters to fit and fixed parameters:', self.mc_params)
        
