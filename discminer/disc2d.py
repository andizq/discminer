"""
2D Disc models
==============
Classes: Rosenfeld2d, Model, Velocity, Intensity, Cube, Tools
"""

#TODO in v1.0: migrate to astropy units
#TODO in Model: Implement irregular grids (see e.g.  meshio from nschloe on github) for the disc grid. It could be useful for better refinement of e.g. disc centre w/o wasting too much in the outer region.
#TODO in Model: Compute props in the interpolated grid (not in the original grid) to avoid interpolation of props and save time.
#TODO in Model: Allow the lower surface to have independent intensity and line width parametrisations.
#TODO in make_model(): Allow for warped emitting surfaces, check notes for ideas as to how to solve for multiple intersections between l.o.s and emission surface.
#TODO in __main__ file: show intro message when python -m disc2d
#TODO in run_mcmc(): use get() methods instead of allowing the user to use self obj attributes.
#TODO in make_model(): Enable 3D velocities too when subpixel algorithm is used
#TODO in make_model(): Find a smart way (e.g. having a dict per attribute) to pass only the coords needed by a prop attribute, i.e. not all coordinates need to be passed to compute e.g. keplerian velocities.
#TODO in show(): use text labels on line profiles to distinguish profiles when more than 2 cubes are shown.
#TODO in make_model(): Save/load bestfit/input parameters in json files. These should store relevant info in separate dicts (e.g. nwalkers, attribute functions). 
#TODO in run_mcmc(): Implement other minimisation kernels (i.e. Delta_v). Only one kernel atm: difference of intensities per pixel per channel.
#TODO in core.py: Allow setting up a mock grid to make a prototype model without requiring an actual datacube. 
    
from __future__ import print_function

import copy
import itertools
import numbers
import os
import pprint
import sys
import time
import warnings
from multiprocessing import Pool

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy import units as u
from matplotlib import ticker
from scipy.integrate import quad
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit
from scipy.special import ellipe, ellipk
from .tools.utils import FrontendUtils, InputError
from . import constants as sfc
from . import units as sfu
from .core import ModelGrid
from .cube import Cube
from .rail import Contours

#from .cart import Intensity

os.environ["OMP_NUM_THREADS"] = "1"

try: 
    import termtables
    found_termtables = True
except ImportError:
    print ("\n*** For nicer outputs we recommend installing 'termtables' by typing in terminal: pip install termtables ***")
    found_termtables = False

#warnings.filterwarnings("error")
__all__ = ['Cube', 'Tools', 'Intensity', 'Velocity', 'Model', 'Rosenfeld2d']
path_icons = os.path.dirname(os.path.realpath(__file__))+'/icons/'
_break_line = FrontendUtils._break_line

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 22


hypot_func = lambda x,y: np.sqrt(x**2 + y**2) #Slightly faster than np.hypot<np.linalg.norm<scipydistance. Checked precision up to au**2 orders and seemed ok.

class Tools:
    @staticmethod
    def _rotate_sky_plane(x, y, ang):
        xy = np.array([x,y])
        cos_ang = np.cos(ang)
        sin_ang = np.sin(ang)
        rot = np.array([[cos_ang, -sin_ang],
                        [sin_ang, cos_ang]])
        return np.dot(rot, xy)

    @staticmethod
    def _rotate_sky_plane3d(x, y, z, ang, axis='z'):
        xyz = np.array([x,y,z])
        cos_ang = np.cos(ang)
        sin_ang = np.sin(ang)
        if axis == 'x':
            rot = np.array([[1, 0, 0],
                            [0, cos_ang, -sin_ang],
                            [0, sin_ang, cos_ang]])
        if axis == 'y':
            rot = np.array([[cos_ang, 0, -sin_ang],
                            [0, 1, 0],
                            [sin_ang, 0, cos_ang]])
            
        if axis == 'z':
            rot = np.array([[cos_ang, -sin_ang , 0],
                            [sin_ang, cos_ang, 0], 
                            [0, 0, 1]])
        return np.dot(rot, xyz)

    @staticmethod
    def _project_on_skyplane(x, y, z, cos_incl, sin_incl):
        x_pro = x
        y_pro = y * cos_incl - z * sin_incl
        z_pro = y * sin_incl + z * cos_incl
        return x_pro, y_pro, z_pro

    @staticmethod
    def get_sky_from_disc_coords(R, az, z, incl, PA, xc=0, yc=0):
        xp = R*np.cos(az)
        yp = R*np.sin(az)
        zp = z
        xp, yp, zp = Tools._project_on_skyplane(xp, yp, zp, np.cos(incl), np.sin(incl))
        xp, yp = Tools._rotate_sky_plane(xp, yp, PA)
        return xp+xc, yp+yc, zp

    @staticmethod #should be a bound method, self.grid is constant except for z_upper, z_lower
    def _compute_prop(grid, prop_funcs, prop_kwargs):
        n_funcs = len(prop_funcs)
        props = [{} for i in range(n_funcs)]
        for side in ['upper', 'lower']:
            x, y, z, R, phi, R_1d, z_1d = grid[side]
            coord = {'x': x, 'y': y, 'z': z, 'phi': phi, 'R': R, 'R_1d': R_1d, 'z_1d': z_1d}
            for i in range(n_funcs): props[i][side] = prop_funcs[i](coord, **prop_kwargs[i])
        return props
    
    @staticmethod
    def _progress_bar(percent=0, width=50):
        left = width * percent // 100 
        right = width - left
        """
        print('\r[', '#' * left, ' ' * right, ']',
              f' {percent:.0f}%',
              sep='', end='', flush=True)
        """
        print('\r[', '#' * left, ' ' * right, ']', ' %.0f%%'%percent, sep='', end='') #compatible with python2 docs
        sys.stdout.flush()

    @staticmethod
    def _break_line(init='', border='*', middle='=', end='\n', width=100):
        print('\r', init, border, middle * width, border, sep='', end=end)

    @staticmethod
    def _print_logo(filename=path_icons+'logo.txt'):
        logo = open(filename, 'r')
        print(logo.read())
        logo.close()

    @staticmethod
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
            #print (pix_scale, pix_radians)
        else: raise InputError(beam, 'beam object must either be str or Beam instance')

        x_stddev = ((beam.major/pix_scale) / sigma2fwhm).decompose().value 
        y_stddev = ((beam.minor/pix_scale) / sigma2fwhm).decompose().value 
        #print (x_stddev, beam.major, pix_scale)
        angle = (90*u.deg+beam.pa).to(u.radian).value
        gauss_kern = Gaussian2DKernel(x_stddev, y_stddev, angle) 

        #gauss_kern = beam.as_kernel(pix_scale) #as_kernel() is slowing down the run when used in astropy.convolve
        return beam, gauss_kern

    @staticmethod
    def average_pixels_cube(data, frac_pixels, av_method=np.median):
        """
        data: datacube with shape (nchan, nx0, ny0)
        frac_pixels: number of pixels to average
        av_method: function to compute average
        """
        nchan, nx0, ny0 = np.shape(data)
        nx = int(round(nx0/frac_pixels))
        ny = int(round(ny0/frac_pixels))
        av_data = np.zeros((nchan, nx, ny))
        progress = Tools._progress_bar
        if frac_pixels>1:
            di = frac_pixels
            dj = frac_pixels
            print ('Averaging %dx%d pixels from data cube...'%(di, dj))
            for k in range(nchan):
                progress(int(100*k/nchan))
                for i in range(nx):
                    for j in range(ny):
                        av_data[k,i,j] = av_method(data[k,i*di:i*di+di,j*dj:j*dj+dj])
            progress(100)
            return av_data
        else:
            print('frac_pixels is <= 1, no average was performed...')
            return data

    @staticmethod
    def weighted_std(prop, weights, weighted_mean=None):
        sum_weights = np.sum(weights)
        if weighted_mean is None:
            weighted_mean = np.sum(weights*prop)/sum_weights
        n = np.sum(weights>0)
        w_std = np.sqrt(np.sum(weights*(prop-weighted_mean)**2)/((n-1)/n * sum_weights))
        return w_std
            
    #define a fit_double_bell func, with a model input as an optional arg to constrain initial guesses better
    @staticmethod
    def fit_one_gauss_cube(data, vchannels, lw_chan=1.0, sigma_fit=None):
        """
        Fit Gaussian profile along velocity axis to input data
        lw_chan: initial guess for line width is lw_chan*np.mean(dvi).  
        sigma_fit: cube w/ channel weights for each pixel, passed to curve_fit
        """
        gauss = lambda x, *p: p[0]*np.exp(-(x-p[1])**2/(2.*p[2]**2))
        nchan, nx, ny = np.shape(data)
        peak, dpeak = np.zeros((nx, ny)), np.zeros((nx, ny))
        centroid, dcent = np.zeros((nx, ny)), np.zeros((nx, ny))
        linewidth, dlinew = np.zeros((nx, ny)), np.zeros((nx, ny))
        nbad = 0
        ind_max = np.nanargmax(data, axis=0)
        I_max = np.nanmax(data, axis=0)
        vel_peak = vchannels[ind_max]
        dv = lw_chan*np.mean(vchannels[1:]-vchannels[:-1])
        progress = Tools._progress_bar   
        if sigma_fit is None: sigma_func = lambda i,j: None
        else: sigma_func = lambda i,j: sigma_fit[:,i,j]
        print ('Fitting Gaussian profile to pixels (along velocity axis)...')
        for i in range(nx):
            for j in range(ny):
                isfin = np.isfinite(data[:,i,j])
                try: coeff, var_matrix = curve_fit(gauss, vchannels[isfin], data[:,i,j][isfin],
                                                   p0=[I_max[i,j], vel_peak[i,j], dv],
                                                   sigma=sigma_func(i,j))
                except RuntimeError: 
                    nbad+=1
                    continue
                peak[i,j] = coeff[0]
                centroid[i,j] = coeff[1]
                linewidth[i,j] = coeff[2]
                dpeak[i,j], dcent[i,j], dlinew[i,j] = np.sqrt(np.diag(var_matrix))
            progress(int(100*i/nx))
        progress(100)
        print ('\nGaussian fit did not converge for %.2f%s of the pixels'%(100.0*nbad/(nx*ny),'%'))
        return peak, centroid, linewidth, dpeak, dcent, dlinew
        
    @staticmethod
    def get_tb(I, nu, beam, full=True):
        """
        nu in GHz
        Intensity in mJy/beam
        beam object from radio_beam
        if full: use full Planck law, else use rayleigh-jeans approximation
        """
        from astropy import units as u
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

    @staticmethod
    def _get_tb(*args, **kwargs): return Tools.get_tb(*args, **kwargs)
        

class Residuals:
    pass


class PlotTools:
    @staticmethod
    def append_stddev_panel(ax, prop, weights=None, hist=False, fit_gauss_hist=False): #attach significance panel to ax, based on dist. of points prop 
        gauss = lambda x, A, mu, sigma: A*np.exp(-(x-mu)**2/(2.*sigma**2))
        ax1_ylims = ax[-2].get_ylim()
        for axi in ax[:-1]: axi.tick_params(which='both', right=False, labelright=False)
        ax[-1].tick_params(which='both', top=False, bottom=False, labelbottom=False, 
                           left=False, labelleft=False, right=True, labelright=True)
        ax[-1].yaxis.set_label_position('right')
        ax[-1].spines['left'].set_color('0.6')
        ax[-1].spines['left'].set_linewidth(3.5)

        if weights is not None:
            prop_mean = np.sum(weights*prop)/np.sum(weights)
            prop_std = Tools.weighted_std(prop, weights, weighted_mean=prop_mean)
        else:
            prop_mean = np.mean(prop)
            prop_std = np.std(prop)
        max_y = 1.0
        if hist:
            n, bins, patches = ax[-1].hist(prop, bins=2*int(round(len(prop)**(1/3.)))-1, orientation='horizontal', 
                                           density=True, linewidth=1.5, facecolor='0.95', edgecolor='k', alpha=1.0)
            max_y = np.max(n)
            if fit_gauss_hist: #Fit Gaussian to histogram to compare against data distribution
                coeff, var_matrix = curve_fit(gauss, 0.5*(bins[1:]+bins[:-1]), n, p0=[max_y, prop_mean, prop_std])
                prop_x = np.linspace(prop_mean-4*prop_std, prop_mean+4*prop_std, 100)
                prop_y = gauss(prop_x, *coeff)
                ax[-1].plot(prop_y, prop_x, color='tomato', ls='--', lw=2.0)
            
        prop_x = np.linspace(prop_mean-4*prop_std, prop_mean+4*prop_std, 100)
        prop_pars =  [max_y, prop_mean, prop_std]
        prop_y = gauss(prop_x, *prop_pars)
        ax[-1].plot(prop_y, prop_x, color='limegreen', lw=3.5)
        ax[-1].set_xlim(-0.2*max_y, 1.2*max_y)

        #ax[-1].plot([-0.2, 1.0], [prop_mean]*2, color='0.6', lw=2.5)
        #for axi in ax[:-1]: axi.axhline(prop_mean, color='0.6', lw=2.5)    
        for i in range(0,4): 
            prop_stdi = prop_mean+i*prop_std
            gauss_prop_stdi = gauss(prop_stdi, *prop_pars)
            ax[-1].plot([-0.2*max_y, gauss_prop_stdi], [prop_stdi]*2, color='0.6', ls=':', lw=2.)
            for axi in ax[:-1]: axi.axhline(prop_stdi, color='0.6', ls=':', lw=2.)
            if prop_stdi < ax1_ylims[-1] and i>0:
                ax[-1].text(gauss_prop_stdi+0.2*max_y, prop_stdi, r'%d$\sigma$'%i, 
                            fontsize=14, ha='center', va='center', rotation=-90)
        for axi in ax: axi.set_ylim(*ax1_ylims)

       
class Canvas3d:
    pass

class Height:
    @property
    def z_upper_func(self): 
        return self._z_upper_func
          
    @z_upper_func.setter 
    def z_upper_func(self, upper): 
        print('Setting upper surface height function to', upper) 
        self._z_upper_func = upper

    @z_upper_func.deleter 
    def z_upper_func(self): 
        print('Deleting upper surface height function') 
        del self._z_upper_func

    @property
    def z_lower_func(self): 
        return self._z_lower_func
          
    @z_lower_func.setter 
    def z_lower_func(self, lower): 
        print('Setting lower surface height function to', lower) 
        self._z_lower_func = lower

    @z_lower_func.deleter 
    def z_lower_func(self): 
        print('Deleting lower surface height function') 
        del self._z_lower_func

    psi0 = 15*np.pi/180
    @staticmethod
    def z_cone(coord, psi=psi0):
        R = coord['R'] 
        z = np.tan(psi) * R
        return z

    @staticmethod
    def z_cone_neg(coord, psi=psi0):
        return -Height.z_cone(coord, psi)

    @staticmethod
    def z_upper_irregular(coord, file='0.txt', **kwargs):
        R = coord['R']/sfu.au
        Rf, zf = np.loadtxt(file)
        z_interp = interp1d(Rf, zf, **kwargs)
        return sfu.au*z_interp(R)


class Linewidth:
    @property
    def linewidth_func(self): 
        return self._linewidth_func
          
    @linewidth_func.setter 
    def linewidth_func(self, linewidth): 
        print('Setting linewidth function to', linewidth) 
        self._linewidth_func = linewidth

    @linewidth_func.deleter 
    def linewidth_func(self): 
        print('Deleting linewidth function') 
        del self._linewidth_func

    @staticmethod
    def linewidth_powerlaw(coord, L0=0.2, p=-0.4, q=0.3, R0=100*sfu.au, z0=100*sfu.au):
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        z = coord['z']        
        A = L0*R0**-p*z0**-q
        return A*R**p*np.abs(z)**q


class Lineslope:
    @property
    def lineslope_func(self): 
        return self._lineslope_func
          
    @lineslope_func.setter 
    def lineslope_func(self, lineslope): 
        print('Setting lineslope function to', lineslope) 
        self._lineslope_func = lineslope

    @lineslope_func.deleter 
    def lineslope_func(self): 
        print('Deleting lineslope function') 
        del self._lineslope_func

    @staticmethod
    def lineslope_powerlaw(coord, Ls=5.0, p=0.0, q=0.0, R0=100*sfu.au, z0=100*sfu.au):
        if p==0.0 and q==0.0:
            return Ls
        else:
            if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
            else: R = coord['R'] 
            z = coord['z']        
            A = Ls*R0**-p*z0**-q
            return A*R**p*np.abs(z)**q


class ScaleHeight:
    @property
    def scaleheight_func(self): 
        return self._scaleheight_func
          
    @scaleheight_func.setter 
    def scaleheight_func(self, surf): 
        print('Setting scaleheight function to', surf) 
        self._scaleheight_func = surf

    @scaleheight_func.deleter 
    def scaleheight_func(self): 
        print('Deleting scaleheight function') 
        del self._scaleheight_func

    @staticmethod
    def powerlaw(coord, H0=6.5, psi=1.25, R0=100.0):
        #Simple powerlaw in R. 
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        R = R/sfu.au
        return sfu.au*H0*(R/R0)**psi

        
class SurfaceDensity:
    @property
    def surfacedensity_func(self): 
        return self._surfacedensity_func
          
    @surfacedensity_func.setter 
    def surfacedensity_func(self, surf): 
        print('Setting surfacedensity function to', surf) 
        self._surfacedensity_func = surf

    @surfacedensity_func.deleter 
    def surfacedensity_func(self): 
        print('Deleting surfacedensity function') 
        del self._surfacedensity_func

    @staticmethod
    def powerlaw(coord, Ec=30.0, gamma=1.0, Rc=100.0):
        #Simple powerlaw in R.
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        R = R/sfu.au
        return Ec*(R/Rc)**-gamma

    @staticmethod
    def powerlaw_tapered(coord, Ec=30.0, Rc=100.0, gamma=1.0): #30.0 kg/m2 or 3.0 g/cm2
        #Self-similar model of thin, viscous accretion disc, see Rosenfeld+2013.
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        R = R/sfu.au
        return Ec*(R/Rc)**-gamma * np.exp(-(R/Rc)**(2-gamma))
    

class Temperature:
    @property
    def temperature_func(self): 
        return self._temperature_func
          
    @temperature_func.setter 
    def temperature_func(self, temp): 
        print('Setting temperature function to', temp) 
        self._temperature_func = temp

    @temperature_func.deleter 
    def temperature_func(self): 
        print('Deleting temperature function') 
        del self._temperature_func

    @staticmethod
    def temperature_powerlaw(coord, T0=100.0, R0=100*sfu.au, p=-0.4, z0=100*sfu.au, q=0.3):
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        z = coord['z']        
        A = T0*R0**-p*z0**-q
        return A*R**p*np.abs(z)**q        


class Velocity:
    @property
    def velocity_func(self): 
        return self._velocity_func
          
    @velocity_func.setter 
    def velocity_func(self, vel): 
        print('Setting velocity function to', vel) 
        self._velocity_func = vel
        if (vel is Velocity.keplerian_vertical_selfgravity or
            vel is Velocity.keplerian_vertical_selfgravity_pressure):
            """
            R_true_au = self.R_true/sfu.au
            tmp = np.unique(np.array(R_true_au).astype(np.int32))
            #Adding missing upper bound for interp1d purposes:
            self.R_1d = np.append(tmp, tmp[-1]+1) #short 1D list of R in au
            """
            R_true_au = self.R_true/sfu.au
            tmp = np.max(R_true_au)
            self.R_1d = np.arange(1, 4*tmp, 5) #short 1D list of R in au

    @velocity_func.deleter 
    def velocity_func(self): 
        print('Deleting velocity function') 
        del self._velocity_func

    @staticmethod
    def keplerian(coord, Mstar=1.0, vel_sign=1, vsys=0):
        Mstar *= sfu.MSun
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        return vel_sign*np.sqrt(sfc.G*Mstar/R) * 1e-3
    
    @staticmethod
    def keplerian_vertical(coord, Mstar=1.0, vel_sign=1, vsys=0):
        Mstar *= sfu.MSun
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        if 'r' not in coord.keys(): r = hypot_func(R, coord['z'])
        else: r = coord['r']
        return vel_sign*np.sqrt(sfc.G*Mstar/r**3)*R * 1e-3 

    @staticmethod
    def keplerian_pressure(coord, Mstar=1.0, vel_sign=1, vsys=0,
                           gamma=1.0, beta=0.5, H0=6.5, R0=100.0):
        #pressure support taken from Lodato's 2021 notes and Viscardi+2021 thesis
        #--> pressure term assumes vertically isothermal disc, T propto R**-beta, and surfdens propto R**-gamma (using Rosenfeld+2013 notation).
        #--> R0 is the ref radius for scaleheight powerlaw, no need to be set as free par during mcmc.
        Mstar *= sfu.MSun
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R']
        alpha = 1.5 + gamma + 0.5*beta
        dlogH_dlogR = 1.5 - 0.5*beta
        psi = -0.5*beta + 1.5
        H = ScaleHeight.powerlaw({'R': R}, H0=H0, R0=R0, psi=psi)
        vk2 = sfc.G*Mstar/R
        vp2 = vk2*(-alpha*(H/R)**2) #pressure term
        return vel_sign*np.sqrt(vk2 + vp2) * 1e-3 

    @staticmethod
    def keplerian_vertical_pressure(coord, Mstar=1.0, vel_sign=1, vsys=0,
                                    gamma=1.0, beta=0.5, H0=6.5, R0=100.0):
        #pressure support taken from Lodato's 2021 notes and Viscardi+2021 thesis
        #--> pressure term assumes vertically isothermal disc, T propto R**-beta, and surfdens propto R**-gamma (using Rosenfeld+2013 notation).
        #--> R0 is the ref radius for scaleheight powerlaw, no need to be set as free par during mcmc.
        Mstar *= sfu.MSun
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R']
        if 'r' not in coord.keys(): r = hypot_func(R, coord['z'])
        else: r = coord['r']
        z = coord['z']
        
        z_R2 = (z/R)**2
        z_R32 = (1+z_R2)**1.5
        alpha = 1.5 + gamma + 0.5*beta
        dlogH_dlogR = 1.5 - 0.5*beta
        psi = -0.5*beta + 1.5
        H = ScaleHeight.powerlaw({'R': R}, H0=H0, R0=R0, psi=psi)
        vk2 = sfc.G*Mstar/R
        vp2 = vk2*( -alpha*(H/R)**2 + (2/z_R32)*( 1+1.5*z_R2-z_R32-dlogH_dlogR*(1+z_R2-z_R32) ) ) #pressure term        
        return vel_sign*np.sqrt(R**2*sfc.G*Mstar/r**3 + vp2) * 1e-3 

    
    @staticmethod
    def keplerian_vertical_selfgravity(coord, Mstar=1.0, vel_sign=1, vsys=0,
                                       Ec=30.0, gamma=1.0, Rc=100.0,
                                       surfacedensity_func=SurfaceDensity.powerlaw):
        #disc self-gravity contribution taken from Veronesi+2021
        #Surface density function defaults to powerlaw, surfdens propto R**-gamma
        #--> Rc is the critical radius, no need to be set as free par during mcmc if input function is powerlaw.
        Mstar *= sfu.MSun
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        if 'r' not in coord.keys(): r = hypot_func(R, coord['z'])
        else: r = coord['r']
        R_1d = coord['R_1d'] #in au to ease computing below
        z_1d = coord['z_1d'] #in au
        
        def SG_integral(Rp, R, z):
            dR = np.append(Rp[0], Rp[1:]-Rp[:-1]) ##
            Rp_R = Rp/R
            RpxR = Rp*R
            k2 = 4*RpxR/((R+Rp)**2 + z**2)
            k = np.sqrt(k2)
            K1 = ellipk(k2) #It's k2 (not k) here. The def in the Gradshteyn+1980 book differs from that of scipy.
            E2 = ellipe(k2)
            #surf_dens = SurfaceDensity.powerlaw_tapered({'R': Rp*sfu.au}, Ec=Ec, Rc=Rc, gamma=gamma)
            surf_dens = surfacedensity_func({'R': Rp*sfu.au}, Ec=Ec, Rc=Rc, gamma=gamma)
            val = (K1 - 0.25*(k2/(1-k2))*(Rp_R - R/Rp + z**2/RpxR)*E2) * np.sqrt(Rp_R)*k*surf_dens
            #return sfc.G*val*sfu.au 
            return sfc.G*np.sum(val*dR)*sfu.au ##

        R_len = len(R_1d)
        SG_1d = []    
        for i in range(R_len):
            #SG_1d.append(quad(SG_integral, 0, np.inf, args=(R_1d[i], z_1d[i]), limit=100)[0])
            SG_1d.append(SG_integral(R_1d, R_1d[i], z_1d[i])) ##
        SG_2d = interp1d(R_1d, SG_1d)

        return vel_sign*np.sqrt(R**2*sfc.G*Mstar/r**3 + SG_2d(R/sfu.au)) * 1e-3 
    
    @staticmethod
    def keplerian_vertical_selfgravity_pressure(coord, Mstar=1.0, vel_sign=1, vsys=0,
                                                Ec=30.0, gamma=1.0,
                                                beta=0.5,
                                                H0=6.5,
                                                Rc=100.0, R0=100.0):
        #--> Rc and R0 are reference radii for surfdens and scaleheight powerlaws, no need to be set as free pars during mcmc.
        Mstar *= sfu.MSun
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R']
        if 'r' not in coord.keys(): r = hypot_func(R, coord['z'])
        else: r = coord['r']
        R_1d = coord['R_1d'] 
        z_1d = coord['z_1d'] 
        
        def SG_integral(Rp, R, z):
            dR = np.append(Rp[0], Rp[1:]-Rp[:-1]) ##
            Rp_R = Rp/R
            RpxR = Rp*R
            k2 = 4*RpxR/((R+Rp)**2 + z**2)
            k = np.sqrt(k2)
            K1 = ellipk(k2) #It's k2 (not k) here. The def in the Gradshteyn+1980 book differs from that of scipy.
            E2 = ellipe(k2)
            surf_dens = SurfaceDensity.powerlaw({'R': Rp*sfu.au}, Ec=Ec, Rc=Rc, gamma=gamma)
            val = (K1 - 0.25*(k2/(1-k2))*(Rp_R - R/Rp + z**2/RpxR)*E2) * np.sqrt(Rp_R)*k*surf_dens
            #return sfc.G*val*sfu.au 
            return sfc.G*np.sum(val*dR)*sfu.au ##

        R_len = len(R_1d)
        SG_1d = []    
        for i in range(R_len):
            #SG_1d.append(quad(SG_integral, 0, np.inf, args=(R_1d[i], z_1d[i]), limit=100)[0])
            SG_1d.append(SG_integral(R_1d, R_1d[i], z_1d[i])) ##
        SG_2d = interp1d(R_1d, SG_1d)

        #calculate pressure support
        z = coord['z']
        z_R2 = (z/R)**2
        z_R32 = (1+z_R2)**1.5
        alpha = 1.5 + gamma + 0.5*beta
        dlogH_dlogR = 1.5 - 0.5*beta
        psi = -0.5*beta + 1.5
        H = ScaleHeight.powerlaw({'R': R}, H0=H0, R0=R0, psi=psi)
        vk2 = sfc.G*Mstar/R
        vp2 = vk2*( -alpha*(H/R)**2 + (2/z_R32)*( 1+1.5*z_R2-z_R32-dlogH_dlogR*(1+z_R2-z_R32) ) )
        
        return vel_sign*np.sqrt(R**2*sfc.G*Mstar/r**3 + SG_2d(R/sfu.au) + vp2) * 1e-3 
    

class Intensity:   
    @property
    def beam_info(self):
        return self._beam_info

    @beam_info.setter 
    def beam_info(self, beam_info): 
        print('Setting beam_info var to', beam_info)
        self._beam_info = beam_info

    @beam_info.deleter 
    def beam_info(self): 
        print('Deleting beam_info var') 
        del self._beam_info     

    @property
    def beam_kernel(self):
        return self._beam_kernel

    @beam_kernel.setter 
    def beam_kernel(self, beam_kernel): 
        print('Setting beam_kernel var to', beam_kernel)
        self._beam_kernel = beam_kernel
        
    @beam_kernel.deleter 
    def beam_kernel(self): 
        print('Deleting beam_kernel var') 
        del self._beam_kernel     

    @property
    def beam_from(self):
        return self._beam_from

    @beam_from.setter 
    def beam_from(self, file): 
        #Rework this, missing beam kwargs info
        print('Setting beam_from var to', file)
        if file: self.beam_info, self.beam_kernel = Tools._get_beam_from(file) #Calls beam_kernel setter
        self._beam_from = file

    @beam_from.deleter 
    def beam_from(self): 
        print('Deleting beam_from var') 
        del self._beam_from     

    @property
    def use_temperature(self):
        return self._use_temperature

    @use_temperature.setter 
    def use_temperature(self, use): 
        use = bool(use)
        print('Setting use_temperature var to', use)
        if use: self.line_profile = self.line_profile_temp
        #else: self.line_profile = self.line_profile_v_sigma
        self._use_temperature = use

    @use_temperature.deleter 
    def use_temperature(self): 
        print('Deleting use_temperature var') 
        del self._use_temperature

    @property
    def use_full_channel(self):
        return self._use_full_channel

    @use_full_channel.setter 
    def use_full_channel(self, use): #Needs remake, there is now a new kernel (line_profile_bell) 
        use = bool(use)
        print('Setting use_full_channel var to', use)
        if use: 
            if self.use_temperature: self.line_profile = self.line_profile_temp_full
            else: self.line_profile = self.line_profile_v_sigma_full
        else: 
            if self.use_temperature: self.line_profile = self.line_profile_temp
            else: self.line_profile = self.line_profile_v_sigma
        self._use_full_channel = use

    @use_full_channel.deleter 
    def use_full_channel(self): 
        print('Deleting use_full_channel var') 
        del self._use_full_channel

    @property
    def line_profile(self): 
        return self._line_profile
          
    @line_profile.setter 
    def line_profile(self, profile): 
        print('Setting line profile function to', profile) 
        self._line_profile = profile

    @line_profile.deleter 
    def line_profile(self): 
        print('Deleting intensity function') 
        del self._line_profile
    
    @property
    def intensity_func(self): 
        return self._intensity_func
          
    @intensity_func.setter 
    def intensity_func(self, intensity): 
        print('Setting intensity function to', intensity) 
        self._intensity_func = intensity

    @intensity_func.deleter 
    def intensity_func(self): 
        print('Deleting intensity function') 
        del self._intensity_func

    @staticmethod
    def intensity_powerlaw(coord, I0=30.0, R0=100*sfu.au, p=-0.4, z0=100*sfu.au, q=0.3):
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        z = coord['z']        
        A = I0*R0**-p*z0**-q
        return A*R**p*np.abs(z)**q
        
    @staticmethod
    def nuker(coord, I0=30.0, Rt=100*sfu.au, alpha=-0.5, gamma=0.1, beta=0.2):
        if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
        else: R = coord['R'] 
        A = I0*Rt**gamma
        return A*(R**-gamma) * (1+(R/Rt)**alpha)**((gamma-beta)/alpha)

    @staticmethod
    def line_profile_subchannel(line_profile_func, v_chan, v, v_sigma, b_slope, channel_width=0.1, **kwargs): #Currently not used
        half_chan = 0.5*channel_width
        v0 = v_chan - half_chan
        v1 = v_chan + half_chan
        nsub = 10
        vsub = np.linspace(v0, v1, nsub)
        dvsub = vsub[1]-vsub[0]
        J = 0
        for vs in vsub:
            J += line_profile_func(vs, v, v_sigma, b_slope, **kwargs) 
        J = J * dvsub/channel_width
        return J
        
    @staticmethod
    def line_profile_temp(v_chan, v, T, dum, v_turb=0.0, mmol=2*sfu.amu):
        v_sigma = np.sqrt(sfc.kb*T/mmol + v_turb**2) * 1e-3 #in km/s
        #return 1/(np.sqrt(np.pi)*v_sigma) * np.exp(-((v-v_chan)/v_sigma)**2)
        return np.exp(-0.5*((v-v_chan)/v_sigma)**2)

    @staticmethod
    def line_profile_temp_full(v_chan, v, T, dum, v_turb=0, mmol=2*sfu.amu, channel_width=0.1):
        v_sigma = np.sqrt(sfc.kb*T/mmol + v_turb**2) * 1e-3 #in km/s
        half_chan = 0.5*channel_width
        v0 = v_chan - half_chan
        v1 = v_chan + half_chan
        nsub = 10
        vsub = np.linspace(v0, v1, nsub)
        dvsub = vsub[1]-vsub[0]
        J = 0
        for vs in vsub:
            J += np.exp(-0.5*((v-vs)/v_sigma)**2)
        J = J * dvsub/channel_width
        return J

    @staticmethod
    def line_profile_v_sigma(v_chan, v, v_sigma, dum):
        #return 1/(np.sqrt(2*np.pi)*v_sigma) * np.exp(-0.5*((v-v_chan)/v_sigma)**2)
        #return np.where(np.abs(v-v_chan) < 0.5*v_sigma, 1, 0)
        return np.exp(-0.5*((v-v_chan)/v_sigma)**2)
    
    @staticmethod
    def line_profile_v_sigma_full(v_chan, v, v_sigma, dum, channel_width=0.1):
        half_chan = 0.5*channel_width
        v0 = v_chan - half_chan
        v1 = v_chan + half_chan
        nsub = 10
        vsub = np.linspace(v0, v1, nsub)
        dvsub = vsub[1]-vsub[0]
        J = 0
        for vs in vsub:
            J += np.exp(-0.5*((v-vs)/v_sigma)**2)
        J = J * dvsub/channel_width
        return J

    @staticmethod
    def line_profile_bell(v_chan, v, v_sigma, b_slope):
        return 1/(1+np.abs((v-v_chan)/v_sigma)**(2*b_slope))        

    @staticmethod
    def line_profile_bell_full(v_chan, v, v_sigma, b_slope, channel_width=0.1):
        half_chan = 0.5*channel_width
        v0 = v_chan - half_chan
        v1 = v_chan + half_chan
        nsub = 10
        vsub = np.linspace(v0, v1, nsub)
        dvsub = vsub[1]-vsub[0]
        J = 0
        for vs in vsub:
            J += 1/(1+np.abs((v-vs)/v_sigma)**(2*b_slope))
        J = J * dvsub/channel_width
        return J

    def get_line_profile(self, v_chan, vel2d, linew2d, lineb2d, **kwargs):
        if self.subpixels:
            v_near, v_far = [], []
            for i in range(self.subpixels_sq):
                v_near.append(self.line_profile(v_chan, vel2d[i]['upper'], linew2d['upper'], lineb2d['upper'], **kwargs))
                v_far.append(self.line_profile(v_chan, vel2d[i]['lower'], linew2d['lower'], lineb2d['lower'], **kwargs))

            integ_v_near = np.sum(np.array(v_near), axis=0) * self.sub_dA / self.pix_dA
            integ_v_far = np.sum(np.array(v_far), axis=0) * self.sub_dA / self.pix_dA
            return integ_v_near, integ_v_far
        
        else: 
            v_near = self.line_profile(v_chan, vel2d['upper'], linew2d['upper'], lineb2d['upper'], **kwargs)
            v_far = self.line_profile(v_chan, vel2d['lower'], linew2d['lower'], lineb2d['lower'], **kwargs)
            return v_near, v_far 

    def get_channel(self, velocity2d, intensity2d, linewidth2d, lineslope2d, v_chan, **kwargs):                    
        vel2d, int2d, linew2d, lineb2d = velocity2d, {}, {}, {}

        if isinstance(intensity2d, numbers.Number): int2d['upper'] = int2d['lower'] = intensity2d
        else: int2d = intensity2d
        if isinstance(linewidth2d, numbers.Number): linew2d['upper'] = linew2d['lower'] = linewidth2d
        else: linew2d = linewidth2d
        if isinstance(lineslope2d, numbers.Number): lineb2d['upper'] = lineb2d['lower'] = lineslope2d
        else: lineb2d = lineslope2d
    
        v_near, v_far = self.get_line_profile(v_chan, vel2d, linew2d, lineb2d, **kwargs)
        """
        v_near_clean = np.where(np.isnan(v_near), -np.inf, v_near)
        v_far_clean = np.where(np.isnan(v_far), -np.inf, v_far)
        #vmap_full = np.array([v_near_clean, v_far_clean]).max(axis=0)        
        int2d_near = np.where(np.isnan(int2d['upper']), -np.inf, int2d['upper'] * v_near_clean)# / v_near_clean.max())
        int2d_far = np.where(np.isnan(int2d['lower']), -np.inf, int2d['lower'] * v_far_clean)# / v_far_clean.max())
        """
        int2d_full = np.nanmax([int2d_near, int2d_far], axis=0)
        
        if self.beam_kernel:
            """
            inf_mask = np.isinf(int2d_full)
            """
            inf_mask = np.isnan(int2d_full)
            int2d_full = np.where(inf_mask, 0.0, int2d_full) # Use np.nan_to_num instead
            int2d_full = self.beam_area*convolve(int2d_full, self.beam_kernel, preserve_nan=False)

        return int2d_full

    def get_cube(self, vchannels, velocity2d, intensity2d, linewidth2d, lineslope2d, make_convolve=True, #Should be in model class
                 nchan=None, rms=None, tb={'nu': False, 'beam': False, 'full': True}, return_data_only=False, header=None, dpc=None, **kwargs_line):

        #from .cube import Cube as Cube2 #header should already be known by here, i.e. it should be an input when Model is initialised
        
        vel2d, int2d, linew2d, lineb2d = velocity2d, {}, {}, {}
        line_profile = self.line_profile
        if nchan is None: nchan=len(vchannels)
        int2d_shape = np.shape(velocity2d['upper'])
        
        if isinstance(intensity2d, numbers.Number): int2d['upper'] = int2d['lower'] = intensity2d
        else: int2d = intensity2d
        if isinstance(linewidth2d, numbers.Number): linew2d['upper'] = linew2d['lower'] = linewidth2d
        else: linew2d = linewidth2d
        if isinstance(lineslope2d, numbers.Number): lineb2d['upper'] = lineb2d['lower'] = lineslope2d
        else: lineb2d = lineslope2d

        """
        int2d_near_nan = np.isnan(int2d['upper']) #~int2d['upper'].mask
        int2d_far_nan = np.isnan(int2d['lower']) #~int2d['lower'].mask
        if self.subpixels:
            vel2d_near_nan = np.isnan(vel2d[self.sub_centre_id]['upper'])
            vel2d_far_nan = np.isnan(vel2d[self.sub_centre_id]['lower'])
        else:
            vel2d_near_nan = np.isnan(vel2d['upper']) #~vel2d['upper'].mask
            vel2d_far_nan = np.isnan(vel2d['lower']) #~vel2d['lower'].mask
        """

        cube = []
        noise = 0.0
        #for _ in itertools.repeat(None, nchan):
        for vchan in vchannels:
            v_near, v_far = self.get_line_profile(vchan, vel2d, linew2d, lineb2d, **kwargs_line)
            """
            v_near_clean = np.where(vel2d_near_nan, -np.inf, v_near)
            v_far_clean = np.where(vel2d_far_nan, -np.inf, v_far)
            #vmap_full = np.array([v_near_clean, v_far_clean]).max(axis=0)
            int2d_near = np.where(int2d_near_nan, -np.inf, int2d['upper'] * v_near_clean)
            int2d_far = np.where(int2d_far_nan, -np.inf, int2d['lower'] * v_far_clean)
            """
            int2d_near = int2d['upper'] * v_near
            int2d_far = int2d['lower'] * v_far
            #vel nans might differ from Int nans when a z surf is zero and SG is active, then nanmax must be used: 
            int2d_full = np.nanmax([int2d_near, int2d_far], axis=0) 
            if rms is not None:
                noise = np.random.normal(scale=rms, size=int2d_shape)
                int2d_full += noise

            if make_convolve and self.beam_kernel:
                int2d_full[np.isnan(int2d_full)] = noise #np.where(inf_mask, noise, int2d_full)                
                int2d_full = self.beam_area*convolve(int2d_full, self.beam_kernel, preserve_nan=False)
            else:
                int2d_full *= self.beam_area
                int2d_full[~np.isfinite(int2d_full)] = noise
                
            cube.append(int2d_full)
            
        if return_data_only: return np.asarray(cube)
        #else: return Cube(nchan, vchannels, np.asarray(cube), beam=self.beam_info, beam_kernel=self.beam_kernel, tb=tb)
        else: return Cube(np.asarray(cube), header, vchannels, dpc, beam=self.beam_info, filename="./cube_model.fits")
        #else: return Cube2(np.asarray(cube), header, vchannels, beam=self.beam_info)

    @staticmethod
    def make_channels_movie(vchan0, vchan1, velocity2d, intensity2d, linewidth2d, lineslope2d, nchans=30, folder='./movie_channels/', **kwargs):
        channels = np.linspace(vchan0, vchan1, num=nchans)
        int2d_cube = []
        for i, vchan in enumerate(channels):
            int2d = Intensity.get_channel(velocity2d, intensity2d, linewidth2d, lineslope2d, vchan, **kwargs)
            int2d_cube.append(int2d)
            extent = [-600, 600, -600, 600]
            plt.imshow(int2d, cmap='binary', extent=extent, origin='lower', vmax=np.max(linewidth2d['upper']))
            plt.xlabel('au')
            plt.ylabel('au')
            plt.text(200, 500, '%.1f km/s'%vchan)
            cbar = plt.colorbar()
            cbar.set_label(r'Brightness Temperature [K]')
            plt.contour(velocity2d['upper'], levels=[vchan], colors='red', linestyles='--', linewidths=1.3, extent = extent)
            plt.contour(velocity2d['lower'], levels=[vchan], colors='red', linestyles=':', linewidths=1.3, extent = extent)
            plt.plot([None],[None], color='red', linestyle='--', linewidth=2, label='Near side') 
            plt.plot([None],[None], color='red', linestyle=':', linewidth=2, label='Far side') 
            plt.legend(loc='upper left')
            plt.savefig(folder+'int2d_chan%04d'%i)
            print ('Saving channel %d'%i)
            plt.close()

        os.chdir(folder)
        print ('Making channels movie...')
        os.system('convert -delay 10 *int2d* cube_channels.gif')
        return np.array(int2d_cube)

    
class Mcmc:
    @staticmethod
    def _get_params2fit(mc_params, boundaries):
        header = []
        kind = []
        params_indices = {}
        boundaries_list = []
        check_param2fit = lambda val: val and isinstance(val, bool)
        i = 0
        for key in mc_params:
            if isinstance(mc_params[key], dict):
                params_indices[key] = {}
                for key2 in mc_params[key]: 
                    if check_param2fit(mc_params[key][key2]):
                        header.append(key2)
                        kind.append(key)
                        boundaries_list.append(boundaries[key][key2])
                        params_indices[key][key2] = i
                        i+=1
            else: raise InputError(mc_params, 'Wrong input parameters. Base keys in mc_params must be categories; parameters of a category must be within a dictionary as well.')

        return header, kind, len(header), boundaries_list, params_indices
    
    @staticmethod
    def plot_walkers(samples, best_params, nstats=None, header=None, kind=None, tag=''):
        npars, nsteps, nwalkers = samples.shape
        if kind is not None:
            ukind, neach = np.unique(kind, return_counts=True)
            ncols = len(ukind)
            nrows = np.max(neach)
        else:
            ukind = [''] 
            ncols = 1
            nrows = npars
            kind = ['' for i in range(nrows)] 
        
        if header is not None:
            if len(header) != npars: raise InputError(header, 'Number of headers must be equal to number of parameters')
            
        kind_col = {ukind[i]: i for i in range(ncols)}
        col_count = np.zeros(ncols).astype('int')

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
        x0_hline = 0
        for k, key in enumerate(kind):
            j = kind_col[key]
            i = col_count[j] 
            for walker in samples[k].T:
                if ncols == 1: axij = ax[i]
                elif nrows==1: axij = ax[j]
                else: axij = ax[i][j]
                axij.plot(walker, alpha=0.1, lw=1.0, color='k')
                if header is not None: 
                    #axij.set_ylabel(header[k])
                    axij.text(0.1, 0.1, header[k], va='center', ha='left', fontsize=MEDIUM_SIZE+2, transform=axij.transAxes, rotation=0) #0.06, 0.95, va top, rot 90
            if i==0: axij.set_title(key, pad=10, fontsize=MEDIUM_SIZE+2)
            if nstats is not None: 
                axij.axvline(nsteps-nstats, ls=':', lw=2, color='r')
                x0_hline = nsteps-nstats

            axij.plot([x0_hline, nsteps], [best_params[k]]*2, ls='-', lw=3, color='dodgerblue')
            axij.text((nsteps-1)+0.03*nsteps, best_params[k], '%.3f'%best_params[k], va='center', color='dodgerblue', fontsize=MEDIUM_SIZE+1, rotation=90) 
            axij.tick_params(axis='y', which='major', labelsize=SMALL_SIZE, rotation=45)
            axij.set_xlim(None, nsteps-1 + 0.01*nsteps)
            col_count[j]+=1

        for j in range(ncols):
            i_last = col_count[j]-1
            if ncols==1: ax[i_last].set_xlabel('Steps')
            elif nrows==1: ax[j].set_xlabel('Steps') 
            else: ax[i_last][j].set_xlabel('Steps')
            if nrows>1 and i_last<nrows-1: #Remove empty axes
                for k in range((nrows-1)-i_last): ax[nrows-1-k][j].axis('off')

        plt.tight_layout()
        plt.savefig('mc_walkers_%s_%dwalkers_%dsteps.png'%(tag, nwalkers, nsteps), dpi=300)
        plt.close()

    @staticmethod
    def plot_corner(samples, labels=None, quantiles=None):
        """Plot corner plot to check parameter correlations. Requires the 'corner' module"""
        import corner
        quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles
        corner.corner(samples, labels=labels, title_fmt='.4f', bins=30,
                      quantiles=quantiles, show_titles=True)
        
    def ln_likelihood(self, new_params, **kwargs):
        for i in range(self.mc_nparams):
            #Assuming uniform prior likelihood (within boundaries) for all parameters
            if not (self.mc_boundaries_list[i][0] < new_params[i] < self.mc_boundaries_list[i][1]): return -np.inf
            else: self.params[self.mc_kind[i]][self.mc_header[i]] = new_params[i]

        vel2d, int2d, linew2d, lineb2d = self.make_model(**kwargs)

        lnx2=0    
        model_cube = self.get_cube(self.channels, vel2d, int2d, linew2d, lineb2d, nchan=self.nchan, return_data_only=True)#, tb = {'nu': 230, 'beam': self.beam_info})
        for i in range(self.nchan):
            model_chan = model_cube[i]
            mask_data = np.isfinite(self.data[i])
            mask_model = np.isfinite(model_chan)
            data = np.where(np.logical_and(mask_model, ~mask_data), 0, self.data[i])
            model = np.where(np.logical_and(mask_data, ~mask_model), 0, model_chan)
            mask = np.logical_and(mask_data, mask_model)
            lnx =  np.where(mask, np.power((data - model)/self.noise_stddev, 2), 0) 
            lnx2 += -0.5 * np.sum(lnx)
            
        #print (new_params, "\nLOG LIKELIHOOD %.4e"%lnx2)
        return lnx2 if np.isfinite(lnx2) else -np.inf
    

class Model(Height, Velocity, Intensity, Linewidth, Lineslope, Tools, Mcmc): #Inheritance should only be from Intensity and Mcmc, the others contain just staticmethods...
    def __init__(self, datacube, Rmax, Rmin=1.0, prototype=False, subpixels=False):        
        """
        Initialise discminer model object.

        Parameters
        ----------
        datacube : `~discminer.disc2d.cube.Cube` object
            Datacube to be modelled with discminer through MCMC methods (see `~discminer.disc2d.Model.run_mcmc`). It can also be used as a reference cube to make a prototype model, which can then be employed as an initial guess for the MCMC parameter search.

        Rmax : `~astropy.units.Quantity`
            Maximum radial extent of the model in physical units. Not to be confused with the disc outer radius.

        Rmin : float or `~astropy.units.Quantity`
            Inner radius to mask out from the model.

            - If float, computes inner radius in number of beams. Default is 1.0.

            - If `~astropy.units.Quantity`, takes the value provided, assumed in physical units.

        prototype : bool, optional
            Compute a prototype model. This is useful for quick inspection of channels given a set of parameters, which can then be used as seeding parameters for the MCMC fit. It is also helpful for analysis purposes once a best-fit model has been found. Defaults to False.
        
        subpixels : bool, optional
            Subdivide original grid pixels into smaller pixels (subpixels) to account for large velocity gradients in the disc. This allows for more precise calculations of line-of-sight velocities in regions where velocity gradients across individual pixels can be large, e.g. near the centre of the disc. Defaults to False.

        Attributes
        ----------
        skygrid : dict
            Dictionary with useful information of the sky grid where the disc observable properties are merged for visualisation. This grid matches the spatial grid of the input datacube in physical units.

        grid : dict
            Disc grid where the model disc properties are computed. Why are there two different grids? Sometimes, the maximum (deprojected) extent of the disc is larger than the rectangular size of the sky grid. Therefore, having independent grids is needed to make sure that the deprojected disc properties do not display sharp boundaries at R=skygrid_extent. In other cases, the maximum extent of the disc is smaller than the sky grid, in which case it's useful to employ a (smaller) independent grid to save computing time.
            
        """
        
        FrontendUtils._print_logo()        
        self.prototype = prototype

        mgrid = ModelGrid(datacube, Rmax, Rmin=Rmin) #Make model grid (disc and sky grids)
        grid = mgrid.discgrid        
        skygrid = mgrid.skygrid

        self.Rmax = mgrid.Rmax
        self.Rmin = mgrid.Rmin

        self.Rmax_m = mgrid.Rmax.to('m').value
        self.Rmin_m = mgrid.Rmin.to('m').value        

        self.vchannels = mgrid.vchannels
        self.header = mgrid.header
        self.dpc = mgrid.dpc
        
        self.beam = datacube.beam
        self.beam_info = datacube.beam
        self.beam_size = datacube.beam_size        
        self.beam_area = datacube.beam_area
        self.beam_kernel = datacube.beam_kernel
        
        self._z_upper_func = Model.z_cone
        self._z_lower_func = Model.z_cone_neg
        self._velocity_func = Model.keplerian
        self._intensity_func = Model.intensity_powerlaw
        self._linewidth_func = Model.linewidth_powerlaw
        self._lineslope_func = Model.lineslope_powerlaw
        self._line_profile = Model.line_profile_bell
        self._use_temperature = False
        self._use_full_channel = False
 
        x_true, y_true = grid['x'], grid['y']
        self.x_true, self.y_true = x_true, y_true
        self.phi_true = grid['phi'] #From 0 to 2pi, old sf3d version: -pi, pi
        self.R_true = grid['R']         
        self.mesh = skygrid['meshgrid'] #disc grid will be interpolated onto this sky grid in make_model(). Must match data shape for mcmc. 
        self.grid = grid
        self.skygrid = skygrid
        self.extent = skygrid['extent']
        self.projected_coords = None #to be computed by get_projected_coords
    
        self.R_1d = None #will be modified if selfgravity is considered

        if subpixels and isinstance(subpixels, int):
            if subpixels%2 == 0: subpixels+=1 #If input even becomes odd to contain pxl centre
            pix_size = grid['step']
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
            self.mc_header, self.mc_kind, self.mc_nparams, self.mc_boundaries_list, self.mc_params_indices = Model._get_params2fit(self.mc_params, self.mc_boundaries)
            #print ('Default parameter header for mcmc fitting:', self.mc_header)
            #print ('Default parameters to fit and fixed parameters:', self.mc_params)

    def plot_quick_attributes(self, R_in=10, R_out=300, surface='upper', fig_width=80, fig_height=25,
                              height=True, velocity=True, linewidth=True, peakintensity=True, **kwargs_plot):                              
        import termplotlib as tpl  # pip install termplotlib. Requires gnuplot: brew install gnuplot (for OSX users)
        kwargs = dict(plot_command="plot '-' w steps", xlabel='Offset [au]', label=None, xlim=None, ylim=None) #plot_command: lines, steps, dots, points, boxes
        kwargs.update(kwargs_plot)
        R = np.linspace(R_in, R_out, 100)
        if surface=='upper': coords = {'R': R*sfu.au, 'z': self.z_upper_func({'R': R*sfu.au}, **self.params['height_upper'])}
        if surface=='lower': coords = {'R': R*sfu.au, 'z': self.z_lower_func({'R': R*sfu.au}, **self.params['height_lower'])}        
        def make_plot(func, kind, tag=None, val_unit=1, surface=surface):
            if tag is None: tag=kind
            fig = tpl.figure()
            val = func(coords, **self.params[kind])
            fig.plot(R, val/val_unit, width=fig_width, height=fig_height, title=tag+' '+surface, **kwargs)
            fig.show()
        if height and surface=='upper': make_plot(self.z_upper_func, 'height_upper', val_unit=sfu.au, surface='')
        if height and surface=='lower': make_plot(self.z_lower_func, 'height_lower', val_unit=sfu.au, surface='')
        if velocity: make_plot(self.velocity_func, 'velocity')
        if linewidth: make_plot(self.linewidth_func, 'linewidth')
        if peakintensity: make_plot(self.intensity_func, 'intensity', tag='peak intensity')
        
    def run_mcmc(self, data, channels, p0_mean=[], p0_stddev=1e-3, noise_stddev=1.0,
                 nwalkers=30, nsteps=100, frac_stats=0.5, frac_stddev=1e-3, 
                 nthreads=None,
                 backend=None, #emcee
                 use_zeus=False,
                 #custom_header={}, custom_kind={}, mc_layers=1,
                 z_mirror=False, 
                 plot_walkers=True, plot_corner=True, tag='',
                 mpi=False,
                 **kwargs_model): 
        #p0: list of initial guesses. In the future will support 'optimize', 'min_bound', 'max_bound'
        self.data = data
        self.channels = channels
        self.nchan = len(channels)
        self.noise_stddev = noise_stddev
        if use_zeus: import zeus as sampler_id
        else: import emcee as sampler_id
            
        kwargs_model.update({'z_mirror': z_mirror})
        if z_mirror: 
            for key in self.mc_params['height_lower']: self.mc_params['height_lower'][key] = 'height_upper_mirror'
        self.mc_header, self.mc_kind, self.mc_nparams, self.mc_boundaries_list, self.mc_params_indices = Model._get_params2fit(self.mc_params, self.mc_boundaries)
        self.params = copy.deepcopy(self.mc_params)

        #if p0_mean == 'optimize': p0_mean = optimize_p0()
        if isinstance(p0_mean, (list, tuple, np.ndarray)): 
            if len(p0_mean) != self.mc_nparams: raise InputError(p0_mean, 'Length of input p0_mean must be equal to the number of parameters to fit: %d'%self.mc_nparams)
            else: pass

        nstats = int(round(frac_stats*(nsteps-1))) #python2 round returns float, python3 returns int
        ndim = self.mc_nparams

        p0_stddev = [frac_stddev*(self.mc_boundaries_list[i][1] - self.mc_boundaries_list[i][0]) for i in range(self.mc_nparams)]
        p0 = np.random.normal(loc=p0_mean,
                              scale=p0_stddev,
                              size=(nwalkers, ndim)
                              )

        _break_line()
        print ('Initialising MCMC routines with the following (%d) parameters:\n'%self.mc_nparams)
        if found_termtables:
            bound_left, bound_right = np.array(self.mc_boundaries_list).T
            tt_header = ['Attribute', 'Parameter', 'Mean initial guess', 'Par stddev', 'Lower bound', 'Upper bound']
            tt_data = np.array([self.mc_kind, self.mc_header, p0_mean, p0_stddev, bound_left, bound_right]).T
            termtables.print(
                tt_data,
                header=tt_header,
                style=termtables.styles.markdown,
                padding=(0, 1),
                alignment="lcllll"
                )
        else:
            print ('Parameter header set for mcmc model fitting:', self.mc_header)
            print ('Parameters to fit and fixed parameters:')
            pprint.pprint(self.mc_params)
            print ('Number of mc parameters:', self.mc_nparams)
            print ('Parameter attributes:', self.mc_kind)
            print ('Parameter boundaries:')
            pprint.pprint(self.mc_boundaries_list)
            print ('Mean for initial guess p0:', p0_mean)
            print ('p0 pars stddev:', p0_stddev)
        _break_line(init='\n', end='\n\n')

        if mpi: #Needs schwimmbad library: $ pip install schwimmbad 
            from schwimmbad import MPIPool

            with MPIPool() as pool:
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)
               
                sampler = sampler_id.EnsembleSampler(nwalkers, ndim, self.ln_likelihood, pool=pool, backend=backend, kwargs=kwargs_model)                                                        
                start = time.time()
                if backend is not None and backend.iteration!=0:
                    sampler.run_mcmc(None, nsteps, progress=True)
                else:
                    sampler.run_mcmc(p0, nsteps, progress=True)
                end = time.time()
                multi_time = end - start
                print("MPI multiprocessing took {0:.1f} seconds".format(multi_time))

        else:
            with Pool(processes=nthreads) as pool:
                sampler = sampler_id.EnsembleSampler(nwalkers, ndim, self.ln_likelihood, pool=pool, backend=backend, kwargs=kwargs_model)                                                        
                start = time.time()
                if backend is not None and backend.iteration!=0:
                    sampler.run_mcmc(None, nsteps, progress=True)
                else:
                    sampler.run_mcmc(p0, nsteps, progress=True)
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
            
        sampler_chain = sampler.chain
        if use_zeus: sampler_chain = np.swapaxes(sampler.chain, 0, 1) #zeus chains shape (nsteps, nwalkers, npars) must be swapped
        samples = sampler_chain[:, -nstats:] #3d matrix, chains shape (nwalkers, nstats, npars)
        samples = samples.reshape(-1, samples.shape[-1]) #2d matrix, shape (nwalkers*nstats, npars). With -1 numpy guesses the x dimensionality
        best_params = np.median(samples, axis=0)
        self.mc_sampler = sampler
        self.mc_samples = samples
        self.best_params = best_params

        samples_all = sampler_chain[:, :] #3d matrix, chains shape (nwalkers, nsteps, npars)
        samples_all = samples_all.reshape(-1, samples.shape[-1]) #2d matrix, shape (nwalkers*nsteps, npars)
        self.mc_samples_all = samples_all
        
        #Errors: +- 68.2 percentiles
        errpos, errneg = [], []
        for i in range(self.mc_nparams):
            tmp = best_params[i]
            indpos = samples[:,i] > tmp
            indneg = samples[:,i] < tmp
            val = samples[:,i][indpos] - tmp
            errpos.append(np.percentile(val, [68.2])) #1 sigma (2x perc 34.1), positive pars
            val = np.abs(samples[:,i][indneg] - tmp)
            errneg.append(np.percentile(val, [68.2])) 
        self.best_params_errpos = np.asarray(errpos).squeeze()
        self.best_params_errneg = np.asarray(errneg).squeeze()
        
        best_fit_dict = np.array([p0_mean, best_params, self.best_params_errneg, self.best_params_errpos]).T
        best_fit_dict = {key+'_'+self.mc_kind[i]: str(best_fit_dict[i].tolist())[1:-1] for i,key in enumerate(self.mc_header)}
        self.best_fit_dict = best_fit_dict
        
        _break_line(init='\n')
        print ('Median from parameter walkers for the last %d steps:\n'%nstats)        
        if found_termtables:
            tt_header = ['Parameter', 'Best-fit value', 'error [-]', 'error [+]']
            tt_data = np.array([self.mc_header, self.best_params, self.best_params_errneg, self.best_params_errpos]).T
            termtables.print(
                tt_data,
                header=tt_header,
                style=termtables.styles.markdown,
                padding=(0, 1),
                alignment="clll"
                )
        else:
            print (list(zip(self.mc_header, best_params)))
        _break_line(init='\n', end='\n\n')

        #************
        #PLOTTING
        #************
        #for key in custom_header: self.mc_header[key] = custom_header[key]
        #for key in custom_kund: self.mc_kind[key] = custom_kind[key]
        if plot_walkers: 
            Mcmc.plot_walkers(sampler_chain.T, best_params, header=self.mc_header, kind=self.mc_kind, nstats=nstats, tag=tag)
        if plot_corner: 
            Mcmc.plot_corner(samples, labels=self.mc_header)
            plt.savefig('mc_corner_%s_%dwalkers_%dsteps.png'%(tag, nwalkers, nsteps))
            plt.close()

    def _get_attribute_func(self, attribute):
        return {
            'intensity': self.intensity_func,
            'linewidth': self.linewidth_func,
            'lineslope': self.lineslope_func,
            'velocity': self.velocity_func,
        }[attribute]
    
    def get_attribute_map(self, coords, attribute, surface='upper'):

        for key in coords:
            if isinstance(coords[key], u.Quantity):
                if key=='R': coords[key] = coords[key].to('m').value
                if key=='phi': coords[key] = coords[key].to('radian').value
                if key=='z': coords[key] = coords[key].to('m').value

        if 'z' not in coords.keys():
            if surface=='upper':
                z_map = self.z_upper_func(coords, **self.params['height_upper'])
            elif surface=='lower':
                z_map = self.z_lower_func(coords, **self.params['height_lower'])
            else:
                raise InputError(surface, "Only 'upper' or 'lower' are valid surfaces.")
            print ("Updating input coords dictionary with %s surface height values..."%surface)
            coords.update({'z': z_map})
            
        attribute_func = self._get_attribute_func(attribute)
        return attribute_func(coords, **self.params[attribute])

    @staticmethod
    def orientation(incl=np.pi/4, PA=0.0, xc=0.0, yc=0.0):
        xc = xc*sfu.au
        yc = yc*sfu.au
        return incl, PA, xc, yc
    
    def get_projected_coords(self, z_mirror=False, writebinaries=True, 
                             R_nan_val=0, phi_nan_val=10*np.pi, z_nan_val=0):
            
        if self.prototype: 
            _break_line()
            print ('Computing disc upper and lower surface coordinates, projected on the sky plane...')
            print ('Using height and orientation parameters from prototype model:\n')
            pprint.pprint({key: self.params[key] for key in ['height_upper', 'height_lower', 'orientation']})
            
        incl, PA, xc, yc = Model.orientation(**self.params['orientation'])
        cos_incl, sin_incl = np.cos(incl), np.sin(incl)

        #*******************************************
        #MAKE TRUE GRID FOR UPPER AND LOWER SURFACES
        z_true = {}
        z_true['upper'] = self.z_upper_func({'R': self.R_true, 'phi': self.phi_true}, **self.params['height_upper'])

        if z_mirror: z_true['lower'] = -z_true['upper']
        else: z_true['lower'] = self.z_lower_func({'R': self.R_true, 'phi': self.phi_true}, **self.params['height_lower']) 

        grid_true = {'upper': [self.x_true, self.y_true, z_true['upper'], self.R_true, self.phi_true], 
                     'lower': [self.x_true, self.y_true, z_true['lower'], self.R_true, self.phi_true]}
        
        #***********************************
        #PROJECT PROPERTIES ON THE SKY PLANE        
        R, phi, z = {}, {}, {}
        for side in ['upper', 'lower']:
            xt, yt, zt = grid_true[side][:3]
            x_pro, y_pro, z_pro = self._project_on_skyplane(xt, yt, zt, cos_incl, sin_incl)
            if PA: x_pro, y_pro = self._rotate_sky_plane(x_pro, y_pro, PA)             
            x_pro = x_pro+xc
            y_pro = y_pro+yc
            R[side] = griddata((x_pro, y_pro), self.R_true, (self.mesh[0], self.mesh[1]), method='linear')
            x_grid = griddata((x_pro, y_pro), xt, (self.mesh[0], self.mesh[1]), method='linear')
            y_grid = griddata((x_pro, y_pro), yt, (self.mesh[0], self.mesh[1]), method='linear')
            phi[side] = np.arctan2(y_grid, x_grid) #-np.pi, np.pi output for user 
            #Since this one is periodic it has to be recalculated, otherwise the interpolation will screw up things at the boundary -np.pi->np.pi
            # When plotting contours there seems to be in any case some sort of interpolation, so there is still problems at the boundary
            #phi[side] = griddata((x_pro, y_pro), self.phi_true, (self.mesh[0], self.mesh[1]), method='linear')
            z[side] = griddata((x_pro, y_pro), z_true[side], (self.mesh[0], self.mesh[1]), method='linear')
            #r[side] = hypot_func(R[side], z[side])
            if self.Rmax_m is not None: 
                for prop in [R, phi, z]: prop[side] = np.where(np.logical_and(R[side]<self.Rmax_m, R[side]>self.Rmin_m), prop[side], np.nan)

            if writebinaries:
                print ('Saving projected R,phi,z disc coordinates for %s emission surface into .npy binaries...'%side)
                np.save('%s_R.npy'%side, R[side])
                np.save('%s_phi.npy'%side, phi[side])
                np.save('%s_z.npy'%side, z[side])
                
        R_nonan, phi_nonan, z_nonan = None, None, None
        if R_nan_val is not None: R_nonan = {side: np.where(np.isnan(R[side]), R_nan_val, R[side]) for side in ['upper', 'lower']} #Use np.nan_to_num instead
        if phi_nan_val is not None: phi_nonan = {side: np.where(np.isnan(phi[side]), phi_nan_val, phi[side]) for side in ['upper', 'lower']}
        if z_nan_val is not None: z_nonan = {side: np.where(np.isnan(z[side]), z_nan_val, z[side]) for side in ['upper', 'lower']}
        _break_line()

        self.projected_coords = {'R': R,
                                 'phi': phi,
                                 'z': z,
                                 'R_nonan': R_nonan,
                                 'phi_nonan': phi_nonan,
                                 'z_nonan': z_nonan
        }

        #return self.projected_coords
        return R, phi, z, R_nonan, phi_nonan, z_nonan

#can be generalised and put outside this class, would require incl, PA and z_func as args.    
    def make_disc_axes(self, ax, Rmax=None, surface='upper'): 
        if Rmax is None:
            Rmax = self.Rmax.to('au')
        else:
            Rmax = Rmax.to('au')
        R_daxes = np.linspace(0, Rmax, 50)
        
        incl, PA, xc, yc = Model.orientation(**self.params['orientation'])
        xc /= sfu.au
        yc /= sfu.au        
        
        if surface=='upper':
            z_daxes = self.z_upper_func({'R': R_daxes.to('m').value}, **self.params['height_upper'])/sfu.au 
        elif surface=='lower':
            z_daxes = self.z_lower_func({'R': R_daxes.to('m').value}, **self.params['height_lower'])/sfu.au
        else:
            raise InputError(surface, "Only 'upper' or 'lower' are valid surfaces.")

        Contours.disc_axes(ax, R_daxes.value, z_daxes, incl, PA, xc=xc, yc=yc)

    def make_emission_surface(self, ax, R_lev=None, phi_lev=None,
                              proj_offset=None, which='both',
                              kwargs_R={}, kwargs_phi={}):
        R = self.projected_coords['R']
        phi = self.projected_coords['phi']
        X, Y = self.skygrid['meshgrid']
        Contours.emission_surface(
            ax, R, phi, self.extent, X=X, Y=Y,
            R_lev=R_lev, phi_lev=phi_lev, proj_offset=proj_offset,
            which=which, kwargs_R=kwargs_R, kwargs_phi=kwargs_phi
        )
        
            
    def make_model(self, z_mirror=False, **kwargs_get_cube):                   
        if self.prototype: 
            _break_line()
            print ('Running prototype model with the following parameters:\n')
            pprint.pprint(self.params)
            _break_line(init='\n')

        incl, PA, xc, yc = Model.orientation(**self.params['orientation'])
        int_kwargs = self.params['intensity']
        vel_kwargs = self.params['velocity']
        lw_kwargs = self.params['linewidth']
        ls_kwargs = self.params['lineslope']

        cos_incl, sin_incl = np.cos(incl), np.sin(incl)

        #*******************************************
        #MAKE TRUE GRID FOR UPPER AND LOWER SURFACES
        z_true = self.z_upper_func({'R': self.R_true, 'phi': self.phi_true}, **self.params['height_upper'])

        if z_mirror: z_true_far = -z_true
        else: z_true_far = self.z_lower_func({'R': self.R_true, 'phi': self.phi_true}, **self.params['height_lower']) 

        if (self.velocity_func is Velocity.keplerian_vertical_selfgravity or
            self.velocity_func is Velocity.keplerian_vertical_selfgravity_pressure):
            z_1d = self.z_upper_func({'R': self.R_1d*sfu.au}, **self.params['height_upper'])/sfu.au
            if z_mirror: z_far_1d = -z_1d
            else: z_far_1d = self.z_lower_func({'R': self.R_1d*sfu.au}, **self.params['height_lower'])/sfu.au
        else: z_1d = z_far_1d = None

        grid_true = {'upper': [self.x_true, self.y_true, z_true, self.R_true, self.phi_true, self.R_1d, z_1d], 
                     'lower': [self.x_true, self.y_true, z_true_far, self.R_true, self.phi_true, self.R_1d, z_far_1d]}

        #*******************************
        #COMPUTE PROPERTIES ON SKY GRID #This will no longer be necessary as all four functions will always be called
        avai_kwargs = [vel_kwargs, int_kwargs, lw_kwargs, ls_kwargs]
        avai_funcs = [self.velocity_func, self.intensity_func, self.linewidth_func, self.lineslope_func]
        true_kwargs = [isinstance(kwarg, dict) for kwarg in avai_kwargs]
        prop_kwargs = [kwarg for i, kwarg in enumerate(avai_kwargs) if true_kwargs[i]]
        prop_funcs = [func for i, func in enumerate(avai_funcs) if true_kwargs[i]]
       
        if self.subpixels:
            subpix_vel = []
            for i in range(self.subpixels):
                for j in range(self.subpixels):
                    z_true = self.z_upper_func({'R': self.sub_R_true[i][j]}, **self.params['height_upper'])
                    
                    if z_mirror: z_true_far = -z_true
                    else: z_true_far = self.z_lower_func({'R': self.sub_R_true[i][j]}, **self.params['height_lower']) 

                    subpix_grid_true = {'upper': [self.sub_x_true[j], self.sub_y_true[i], z_true, self.sub_R_true[i][j], self.sub_phi_true[i][j]], 
                                        'lower': [self.sub_x_true[j], self.sub_y_true[i], z_true_far, self.sub_R_true[i][j], self.sub_phi_true[i][j]]}
                    subpix_vel.append(self._compute_prop(subpix_grid_true, [self.velocity_func], [vel_kwargs])[0])

            ang_fac = sin_incl * np.cos(self.phi_true) 
            for i in range(self.subpixels_sq):
                for side in ['upper', 'lower']:
                    subpix_vel[i][side] *= ang_fac
                    subpix_vel[i][side] += vel_kwargs['vsys']
                    
            props = self._compute_prop(grid_true, prop_funcs[1:], prop_kwargs[1:])
            props.insert(0, subpix_vel)
            
        else: 
            props = self._compute_prop(grid_true, prop_funcs, prop_kwargs)
            if true_kwargs[0]: #Convention: positive vel (+) means gas receding from observer
                phi_fac = sin_incl * np.cos(self.phi_true) #phi component
                for side in ['upper', 'lower']:
                    if len(props[0][side])==3: #3D vel
                        v3d = props[0][side]
                        r_fac = sin_incl * np.sin(self.phi_true)
                        z_fac = cos_incl
                        props[0][side] = v3d[0]*phi_fac+v3d[1]*r_fac+v3d[2]*z_fac
                    else: #1D vel, assuming vphi only
                        props[0][side] *= phi_fac 
                    props[0][side] += vel_kwargs['vsys']

        #***********************************
        #PROJECT PROPERTIES ON THE SKY PLANE        
        x_pro_dict = {}
        y_pro_dict = {}
        z_pro_dict = {}
        for side in ['upper', 'lower']:
            xt, yt, zt = grid_true[side][:3]
            x_pro, y_pro, z_pro = self._project_on_skyplane(xt, yt, zt, cos_incl, sin_incl)
            if PA: x_pro, y_pro = self._rotate_sky_plane(x_pro, y_pro, PA)             
            x_pro = x_pro+xc
            y_pro = y_pro+yc
            if self.Rmax_m is not None: R_grid = griddata((x_pro, y_pro), self.R_true, (self.mesh[0], self.mesh[1]), method='linear')
            x_pro_dict[side] = x_pro
            y_pro_dict[side] = y_pro
            z_pro_dict[side] = z_pro

            if self.subpixels:
                for i in range(self.subpixels_sq): #Subpixels are projected on the same plane where true grid is projected
                    props[0][i][side] = griddata((x_pro, y_pro), props[0][i][side], (self.mesh[0], self.mesh[1]), method='linear') #subpixels velocity
                for prop in props[1:]:
                    prop[side] = griddata((x_pro, y_pro), prop[side], (self.mesh[0], self.mesh[1]), method='linear')
                    if self.Rmax_m is not None: prop[side] = np.where(np.logical_and(R_grid<self.Rmax_m, R_grid>self.Rmin_m), prop[side], np.nan) #Todo: allow for R_in as well
            else:
                for prop in props:
                    if not isinstance(prop[side], numbers.Number): prop[side] = griddata((x_pro, y_pro), prop[side], (self.mesh[0], self.mesh[1]), method='linear')
                    if self.Rmax_m is not None: prop[side] = np.where(np.logical_and(R_grid<self.Rmax_m, R_grid>self.Rmin_m), prop[side], np.nan)
            
        #*************************************
        if self.prototype:
            self.get_projected_coords(z_mirror=z_mirror) #TODO: enable kwargs for this method
            self.props = props
            #Rail.__init__(self, self.projected_coords, self.skygrid)
            return self.get_cube(self.vchannels, *props, header=self.header, dpc=self.dpc, **kwargs_get_cube)
        else:
            return props

General2d = Model #Backcompat

class _Rosenfeld2d(Velocity, Intensity, Linewidth, Tools): #Deprecated, haven't tested it in a while
    """
    Host class for the Rosenfeld+2013 model which describes the velocity field of a flared disc in 2D. 
    This model assumes a (Keplerian) double cone to account for the near and far sides of the disc 
    and solves analytical equations to find the line-of-sight velocity v_obs projected on the sky-plane from both sides. 
    
    Parameters
    ----------
    grid : array_like, shape (nrows, ncols)
       (x', y') map of the sky-plane onto which the disc velocity field will be projected.

    Attributes
    ----------
    velocity_func : function(coord, **kwargs) 
       Velocity function describing the kinematics of the disc. The argument coord is a dictionary
       of coordinates (e.g. 'x', 'y', 'z', 'r', 'R', 'theta', 'phi') where the function will be evaluated. 
       Additional arguments are optional and depend upon the function definition, e.g. Mstar=1.0*sfu.Msun
    """

    def __init__(self, grid):
        self.grid = grid
        self._velocity_func = Rosenfeld2d.keplerian
        self._intensity_func = Rosenfeld2d.intensity_powerlaw
        self._linewidth_func = Rosenfeld2d.linewidth_powerlaw
        self._line_profile = Rosenfeld2d.line_profile_v_sigma
        self._use_temperature = False
        self._use_full_channel = False


    def _get_t(self, A, B, C):
        t = []
        for i in range(self.grid['ncells']):
            p = [A, B[i], C[i]]
            t.append(np.sort(np.roots(p)))
        return np.array(t)

    def make_model(self, incl, psi, PA=0.0, int_kwargs={}, vel_kwargs={}, lw_kwargs=None, ls_kwargs=None):
        """
        Executes the Rosenfeld+2013 model.
        The sum of incl+psi must be < 90, otherwise the quadratic equation will have imaginary roots as some portions of the cone (which has finite extent)
        do not intersect with the sky plane.  

        Parameters
        ----------
        incl : scalar
           Inclination of the disc midplane with respect to the x'y' plane; pi/2 radians is edge-on.
    
        psi : scalar
           Opening angle of the cone describing the velocity field of the gas emitting layer in the disc; 
           0 radians returns the projected velocity field of the disc midplane (i.e no conic emission). 

        PA : scalar, optional
           Position angle in radians. Measured from North (+y) to East (-x).

        Attributes
        ----------
        velocity : array_like, size (n,)
           Velocity field computed using the Rosenfeld+2013 model.

        velocity2d : array_like, size (nx, ny)
           If set get_2d=True: Velocity field computed using the Rosenfeld+2013 model, reshaped to 2D to facilitate plotting.
        """
        if PA: x_plane, y_plane = Rosenfeld2d._rotate_sky_plane(self.x_true, self.y_true, -PA)
        else: x_plane, y_plane = self.x_true, self.y_true

        cos_incl = np.cos(incl)
        sin_incl = np.sin(incl)
        y_plane_cos_incl = y_plane/cos_incl

        #**********************
        #ROSENFELD COEFFICIENTS
        fac = -2*np.sin(psi)**2
        A = np.cos(2*incl) + np.cos(2*psi)
        B = fac * 2*(sin_incl/cos_incl) * y_plane
        C = fac * (x_plane**2 + (y_plane_cos_incl)**2)
        t = self._get_t(A,B,C).T

        #****************************
        #ROSENFELD CONVERSION X<-->X'
        x_true_near = x_plane
        y_true_near = y_plane_cos_incl + t[1]*sin_incl
            
        x_true_far = x_plane
        y_true_far = y_plane_cos_incl + t[0]*sin_incl
        
        #np.hypot 2x faster than np.linalg.norm([x,y], axis=0)
        R_true_near = hypot_func(x_true_near, y_true_near) 
        R_true_far = hypot_func(x_true_far, y_true_far)

        z_true_near = t[1] * cos_incl
        z_true_far = t[0] * cos_incl 

        phi_true_near = np.arctan2(y_true_near, x_true_near)        
        phi_true_far = np.arctan2(y_true_far, x_true_far)        

        #****************************
            
        grid_true =  {'upper': [x_true_near, y_true_near, z_true_near, R_true_near, phi_true_near], 
                      'lower': [x_true_far, y_true_far, z_true_far, R_true_far, phi_true_far]}

        #*******************************
        #COMPUTE PROPERTIES ON TRUE GRID
        avai_kwargs = [vel_kwargs, int_kwargs, lw_kwargs, ls_kwargs]
        avai_funcs = [self.velocity_func, self.intensity_func, self.linewidth_func, self.lineslope_func]
        true_kwargs = [isinstance(kwarg, dict) for kwarg in avai_kwargs]
        prop_kwargs = [kwarg for i, kwarg in enumerate(avai_kwargs) if true_kwargs[i]]
        prop_funcs = [func for i, func in enumerate(avai_funcs) if true_kwargs[i]]
        props = self._compute_prop(grid_true, prop_funcs, prop_kwargs)
        #Positive vel is positive along z, i.e. pointing to the observer, for that reason imposed a (-) factor to convert to the standard convention: (+) receding  
        if true_kwargs[0]:
            ang_fac_near = -sin_incl * np.cos(phi_true_near)
            ang_fac_far = -sin_incl * np.cos(phi_true_far)
            props[0]['upper'] *= ang_fac_near 
            props[0]['lower'] *= ang_fac_far
                
        #*************************************

        return [{side: prop[side].reshape([self.grid['nx']]*2) for side in ['upper', 'lower']} for prop in props]
