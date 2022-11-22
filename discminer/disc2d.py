"""
2D Disc models
==============
Classes: Rosenfeld2d, General2d, Velocity, Intensity, Cube, Tools
"""

#TODO in v1.0: migrate to astropy units
#TODO in General2d: Implement irregular grids (see e.g.  meshio from nschloe on github) for the disc grid. It could be useful for better refinement of e.g. disc centre w/o wasting too much in the outer region.
#TODO in General2d: Compute props in the interpolated grid (not in the original grid) to avoid interpolation of props and save time.
#TODO in General2d: Allow the lower surface to have independent intensity and line width parametrisations.
#TODO in make_model(): Allow for warped emitting surfaces, check notes for ideas as to how to solve for multiple intersections between l.o.s and emission surface.
#TODO in __main__ file: show intro message when python -m disc2d
#TODO in run_mcmc(): use get() methods instead of allowing the user to use self obj attributes.
#TODO in General2d: Initialise R_inner and R_disc in General2d
#TODO in make_model(): Enable 3D velocities too when subpixel algorithm is used
#TODO in make_model(): Find a smart way (e.g. having a dict per attribute) to pass only the coords needed by a prop attribute, i.e. not all coordinates need to be passed to compute e.g. keplerian velocities.
#TODO in show(): use text labels on line profiles to distinguish profiles when more than 2 cubes are shown.
#TODO in make_model(): Save/load bestfit/input parameters in json files. These should store relevant info in separate dicts (e.g. nwalkers, attribute functions). 
#TODO in run_mcmc(): Implement other minimisation kernels (i.e. Delta_v). Only one kernel currently: difference of intensities on each pixel, on each channel.
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
from matplotlib import ticker
from radio_beam import Beam
from scipy.integrate import quad
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit
from scipy.special import ellipe, ellipk
from .tools.utils import InputError
from sf3dmodels.utils import constants as sfc
from sf3dmodels.utils import units as sfu
from .core import Model

#from .cart import Intensity

os.environ["OMP_NUM_THREADS"] = "1"

try: 
    import termtables
    found_termtables = True
except ImportError:
    print ("\n*** For nicer outputs we recommend installing 'termtables' by typing in terminal: pip install termtables ***")
    found_termtables = False

#warnings.filterwarnings("error")
__all__ = ['Cube', 'Tools', 'Intensity', 'Velocity', 'General2d', 'Rosenfeld2d']
path_icons = os.path.dirname(os.path.realpath(__file__))+'/icons/'

"""
matplotlib.rcParams['font.family'] = 'monospace'
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['axes.linewidth'] = 3.0
matplotlib.rcParams['xtick.major.width']=1.6
matplotlib.rcParams['ytick.major.width']=1.6

matplotlib.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of x and y labels
matplotlib.rc('xtick', labelsize=MEDIUM_SIZE-2)    # fontsize of y tick labels
matplotlib.rc('ytick', labelsize=MEDIUM_SIZE-2)    # fontsize of x tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE-1)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of figure title

params = {'xtick.major.size': 6.5,
          'ytick.major.size': 6.5
          }

matplotlib.rcParams.update(params)
"""
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
        return xp+xc, yp+yc, zp #Missing +xc, +yc

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
        from astropy import units as u
        from astropy.io import fits
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
    def mod_nticks_cbars(cbars, nbins=5):
        for cb in cbars:
            cb.locator = ticker.MaxNLocator(nbins=nbins)
            cb.update_ticks()

    @staticmethod
    def mod_major_ticks(ax, axis='both', nbins=6):
        ax.locator_params(axis=axis, nbins=nbins)

    @staticmethod
    def mod_minor_ticks(ax):
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2)) #1 minor tick per major interval
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    @classmethod
    def make_up_ax(cls, ax, xlims=(None, None), ylims=(None, None), 
                   mod_minor=True, mod_major=True, **kwargs_tick_params):
        kwargs_t = dict(labeltop=True, labelbottom=False, top=True, right=True, which='both', direction='in')
        kwargs_t.update(kwargs_tick_params)
        if mod_major: cls.mod_major_ticks(ax)
        if mod_minor: cls.mod_minor_ticks(ax)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.tick_params(**kwargs_t)
                
    @staticmethod
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    @staticmethod
    def get_cmap_from_color(color, lev=3):
        cmap = matplotlib.colors.to_rgba(color)
        newcolors = np.tile(cmap, lev).reshape(lev,4) #Repeats the colour lev times
        newcolors[:,-1] = np.linspace(0.25, 0.95, lev) #Modifies alpha only
        new_cmap = ListedColormap(newcolors)
        return new_cmap

    @staticmethod
    def mask_cmap_interval(cmap, cmap_lims, mask_lims, mask_color=np.ones(4), append=False):
        if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        cmap0, cmap1 = cmap_lims
        mask0, mask1 = mask_lims
        c0 = (mask0-cmap0)/(cmap1-cmap0)
        c1 = (mask1-cmap0)/(cmap1-cmap0)
        id0 = int(round(c0*(cmap.N)))
        id1 = int(round(c1*(cmap.N)))
        new_cmap = copy.copy(cmap)
        new_cmap._init()
        """#The following does not work, plt does not know where to locate the newly added colorss
        if append:
           mask_color_arr = np.broadcast_to(mask_color, (id1-id0, 4))
           new_cmap._lut = np.insert(new_cmap._lut, id0, mask_color_arr, axis=0)
           new_cmap.N = cmap.N + id1-id0
           #Next line redoes the continuous linearsegmented colormap, thus the masked color block is reduced to a single color  
           #new_cmap = new_cmap._resample(new_cmap.N) 
        """
        new_cmap._lut[id0:id1,:] = mask_color 
        return new_cmap

    @staticmethod
    def get_continuous_cmap(hex_list, float_list=None):                                                                               
        """
        Taken from https://github.com/KerryHalupka/custom_colormap 
        creates and returns a color map that can be used in heat map figures.                                                             
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
                                    
        Parameters                                                                                        
        ----------                                                                                          
        hex_list: list of hex code strings                                                                
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns     
        ----------
        matplotlib cmap

        Examples
        ----------
        fig, ax = plt.subplots(1,1)
        hex_list = ['#0091ad', '#fffffc', '#ffd166']
        x, y = np.mgrid[-5:5:0.05, -5:5:0.05]                                
        z = (np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2))
        im = ax.imshow(z, cmap=get_continuous_cmap(hex_list))                                                         
        fig.colorbar(im)                                                                                                                                                                      
        ax.yaxis.set_major_locator(plt.NullLocator()) # remove y axis ticks                                                                                                                               
        ax.xaxis.set_major_locator(plt.NullLocator()) # remove x axis ticks
        plt.show()
        """

        rgb_list = [matplotlib.colors.to_rgb(i) for i in hex_list]
        if float_list is None: float_list = np.linspace(0,1,len(rgb_list))
                                                                                                                                          
        cdict = dict()                                                                                
        for num, col in enumerate(['red', 'green', 'blue']):                                               
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmap_new = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmap_new
        
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


class Contours(PlotTools):
    @staticmethod
    def emission_surface(ax, R, phi, R_lev=None, phi_lev=None, extent=None,
                         proj_offset=None, X=None, Y=None, which='both',
                         kwargs_R={}, kwargs_phi={}):
        kwargs_phif = dict(linestyles=':', linewidths=1.0, colors='k')
        kwargs_Rf = dict(linewidths=1.4, colors='k')
        kwargs_phif.update(kwargs_phi)        
        kwargs_Rf.update(kwargs_R)

        near_nonan = ~np.isnan(R['upper'])

        Rmax = np.max(R['upper'][near_nonan])
        if extent is None:
            extent = np.array([-Rmax, Rmax, -Rmax, Rmax])/sfu.au
        kwargs_phif.update({'extent': extent})
        kwargs_Rf.update({'extent': extent})

        if R_lev is None: R_lev = np.linspace(0.06, 0.97, 4)*Rmax
        else: R_lev = np.sort(R_lev)
        if phi_lev is None: phi_lev = np.linspace(-np.pi*0.95, np.pi, 11, endpoint=False)

        #Splitting phi into pos and neg to try and avoid ugly contours close to -pi and pi
        phi_lev_neg = phi_lev[phi_lev<0] 
        phi_lev_pos = phi_lev[phi_lev>0]
        phi_neg_near = np.where((phi['upper']<0) & (R['upper']>R_lev[0]) & (R['upper']<R_lev[-1]), phi['upper'], np.nan)
        phi_pos_near = np.where((phi['upper']>0) & (R['upper']>R_lev[0]) & (R['upper']<R_lev[-1]), phi['upper'], np.nan)
        phi_neg_far = np.where((phi['lower']<0) & (R['lower']>R_lev[0]) & (R['lower']<R_lev[-1]), phi['lower'], np.nan)
        phi_pos_far = np.where((phi['lower']>0) & (R['lower']>R_lev[0]) & (R['lower']<R_lev[-1]), phi['lower'], np.nan)

        if proj_offset is not None: #For 3d projections
            ax.contour(X, Y, R['upper'], offset=proj_offset, levels=R_lev, **kwargs_Rf)
            ax.contour(X, Y, np.where(near_nonan, np.nan, R['lower']), offset=proj_offset, levels=R_lev, **kwargs_Rf)
            ax.contour(X, Y, phi_pos_near, offset=proj_offset, levels=phi_lev_pos, **kwargs_phif)
            ax.contour(X, Y, phi_neg_near, offset=proj_offset, levels=phi_lev_neg, **kwargs_phif)
            ax.contour(X, Y, np.where(near_nonan, np.nan, phi_pos_far), offset=proj_offset, levels=phi_lev_pos, **kwargs_phif)
            ax.contour(X, Y, np.where(near_nonan, np.nan, phi_neg_far), offset=proj_offset, levels=phi_lev_neg, **kwargs_phif)
            
        else:
            if which=='both':
                ax.contour(R['upper'], levels=R_lev, **kwargs_Rf)
                ax.contour(np.where(near_nonan, np.nan, R['lower']), levels=R_lev, **kwargs_Rf)
                ax.contour(phi_pos_near, levels=phi_lev_pos, **kwargs_phif)
                ax.contour(phi_neg_near, levels=phi_lev_neg, **kwargs_phif)
                ax.contour(np.where(near_nonan, np.nan, phi_pos_far), levels=phi_lev_pos, **kwargs_phif)
                ax.contour(np.where(near_nonan, np.nan, phi_neg_far), levels=phi_lev_neg, **kwargs_phif)
            elif which=='upper':
                ax.contour(R['upper'], levels=R_lev, **kwargs_Rf)
                ax.contour(phi_pos_near, levels=phi_lev_pos, **kwargs_phif)
                ax.contour(phi_neg_near, levels=phi_lev_neg, **kwargs_phif)
            elif which=='lower':
                ax.contour(R['lower'], levels=R_lev, **kwargs_Rf)
                ax.contour(phi_pos_far, levels=phi_lev_pos, **kwargs_phif)
                ax.contour(phi_neg_far, levels=phi_lev_neg, **kwargs_phif)

    #The following method can be optimised if the contour finding process is separated from the plotting
    # by returning coords_list and inds_cont first, which will allow the user use the same set of contours to plot different props.
    @staticmethod
    def prop_along_coords(ax, prop, coords, coord_ref, coord_levels, 
                          ax2=None, X=None, Y=None, 
                          PA=0,
                          acc_threshold=0.05,
                          max_prop_threshold=np.inf,
                          color_bounds=[np.pi/5, np.pi/2],
                          colors=['k', 'dodgerblue', (0,1,0), (1,0,0)],
                          lws=[2, 0.5, 0.2, 0.2], lw_ax2_factor=1,
                          subtract_quadrants=False,
                          subtract_func=np.subtract):
        """
        Compute radial/azimuthal contours according to the model disc geometry 
        to get and plot information from the input 2D property ``prop``.    

        Parameters
        ----------
        ax : `matplotlib.axes` instance, optional
           ax instance to make the plot. 

        prop : array_like, shape (nx, ny)
           Input 2D field to extract information along the computed contours.
        
        coords : list, shape (2,)
           coords[0] [array_like, shape (nx, ny)], is the coordinate 2D map onto which contours will be computed using the input ``coord_levels``;
           coords[1] [array_like, shape (nx, ny)], is the coordinate 2D map against which the ``prop`` values are plotted. The output plot is prop vs coords[1]       
           
        coord_ref : scalar
           Reference coordinate (referred to ``coords[0]``) to highlight among the other contours.
           
        coord_levels : array_like, shape (nlevels,)
           Contour levels to be extracted from ``coords[0]``.
        
        ax2 : `matplotlib.axes` instance (or list of instances), optional
           Additional ax(s) instance(s) to plot the location of contours in the disc. 
           If provided, ``X`` and ``Y`` must also be passed.
           
        X : array_like, shape (nx, ny), optional
           Meshgrid of the model x coordinate (see `numpy.meshgrid`). Required if ax2 instance(s) is provided.

        Y : array_like, shape (nx, ny), optional
           Meshgrid of the model y coordinate (see `numpy.meshgrid`). Required if ax2 instance(s) is provided.
        
        PA : scalar, optional
           Reference position angle.
           
        acc_threshold : float, optional 
           Threshold to accept points on contours at constant coords[0]. If obtained level at a point is such that np.abs(level-level_reference)<acc_threshold the point is accepted

        max_prop_threshold : float, optional 
           Threshold to accept points of contours. Rejects residuals of the contour if they are < max_prop_threshold. Useful to reject hot pixels.

        color_bounds : array_like, shape (nbounds,), optional
           Colour bounds with respect to the reference contour coord_ref.
           
        colors : array_like, shape (nbounds+2,), optional
           Contour colors. (i=0) is reserved for the reference contour coord_ref, 
           (i>0) for contour colors according to the bounds in color_bounds. 
           
        lws : array_like, shape (nbounds+2), optional
           Contour linewidths. Similarly, (i=0) is reserved for coord_ref and 
           (i>0) for subsequent bounds.

        subtract_quadrants : bool, optional
           If True, subtract residuals by folding along the projected minor axis of the disc. Currently working for azimuthal contours only.
           
        subtract_func : function, optional
           If subtract_quadrants, this function is used to operate between folded quadrants. Defaults to np.subtract.
        """
        from skimage import measure 

        coord_list, lev_list, resid_list, color_list = [], [], [], []
        if np.sum(coord_levels==coord_ref)==0: coord_levels = np.append(coord_levels, coord_ref)
        for lev in coord_levels:
            contour = measure.find_contours(coords[0], lev) #, fully_connected='high', positive_orientation='high')
            if len(contour)==0:
                print ('no contours found for phi =', lev)
                continue
            ind_good = np.argmin([np.abs(lev-coords[0][tuple(np.round(contour[i][0]).astype(np.int))]) for i in range(len(contour))]) #getting ind of closest contour to lev
            inds_cont = np.round(contour[ind_good]).astype(np.int)
            inds_cont = [tuple(f) for f in inds_cont]
            first_cont = np.array([coords[0][i] for i in inds_cont])
            second_cont = np.array([coords[1][i] for i in inds_cont])
            prop_cont = np.array([prop[i] for i in inds_cont])
            corr_inds = np.abs(first_cont-lev) < acc_threshold
            if lev == coord_ref: zorder=10
            else: zorder=np.random.randint(0,10)

            lw = lws[-1]
            color = colors[-1]
            for i,bound in enumerate(color_bounds):
                if lev == coord_ref: 
                    lw = lws[0]
                    color = colors[0]
                    zorder = 10
                    break
                if np.abs(coord_ref - lev) < bound:
                    lw = lws[i+1]
                    color = colors[i+1]
                    break

            if subtract_quadrants:
                #if lev < color_bounds[0]: continue
                ref_pos = PA+90 #Reference axis for positive angles
                ref_neg = PA-90
                angles = second_cont[corr_inds]
                prop_ = prop_cont[corr_inds]
                angles_pos = angles[angles>=0]
                angles_neg = angles[angles<0]
                relative_diff_pos = ref_pos - angles_pos
                relative_diff_neg = ref_neg - angles_neg
                angle_diff_pos, prop_diff_pos = [], []
                angle_diff_neg, prop_diff_neg = [], []

                for i,diff in enumerate(relative_diff_pos):
                    #Finding where the difference matches that of the current analysis angle
                    #The -1 flips the sign so that the number on the other side of the symmetry axis is found                
                    ind = np.argmin(np.abs(-1*relative_diff_pos - diff))  
                    mirror_ind = angles==angles_pos[ind]
                    current_ind = angles==angles_pos[i]
                    prop_diff = subtract_func(prop_[current_ind][0], prop_[mirror_ind][0])
                    angle_diff_pos.append(angles_pos[i])
                    prop_diff_pos.append(prop_diff)
                angle_diff_pos = np.asarray(angle_diff_pos)
                prop_diff_pos = np.asarray(prop_diff_pos)

                if len(angle_diff_pos)>1:
                    ind_sort_pos = np.argsort(angle_diff_pos)
                    plot_ang_diff_pos = angle_diff_pos[ind_sort_pos]
                    plot_prop_diff_pos = prop_diff_pos[ind_sort_pos]
                    ind_prop_pos = np.abs(plot_prop_diff_pos)<max_prop_threshold
                    ax.plot(plot_ang_diff_pos[ind_prop_pos], plot_prop_diff_pos[ind_prop_pos], color=color, lw=lw, zorder=zorder)
                    coord_list.append(plot_ang_diff_pos[ind_prop_pos])
                    resid_list.append(plot_prop_diff_pos[ind_prop_pos])
                    color_list.append(color)
                    lev_list.append(lev)
                else: 
                    plot_ang_diff_pos = []
                    plot_prop_diff_pos = []

                for i,diff in enumerate(relative_diff_neg):
                    ind = np.argmin(np.abs(-1*relative_diff_neg - diff))
                    mirror_ind = angles==angles_neg[ind]
                    current_ind = angles==angles_neg[i]
                    prop_diff = subtract_func(prop_[current_ind][0], prop_[mirror_ind][0])
                    angle_diff_neg.append(angles_neg[i])
                    prop_diff_neg.append(prop_diff)
                angle_diff_neg = np.asarray(angle_diff_neg)
                prop_diff_neg = np.asarray(prop_diff_neg)

                if len(angle_diff_neg)>1:
                    ind_sort_neg = np.argsort(np.abs(angle_diff_neg))
                    plot_ang_diff_neg = angle_diff_neg[ind_sort_neg]
                    plot_prop_diff_neg = prop_diff_neg[ind_sort_neg]
                    ind_prop_neg = np.abs(plot_prop_diff_neg)<max_prop_threshold
                    ax.plot(plot_ang_diff_neg[ind_prop_neg], plot_prop_diff_neg[ind_prop_neg], color=color, lw=lw, zorder=zorder)
                    coord_list.append(plot_ang_diff_neg[ind_prop_neg])
                    resid_list.append(plot_prop_diff_neg[ind_prop_neg])
                    color_list.append(color)
                    lev_list.append(lev)
                else: 
                    plot_ang_diff_neg = []
                    plot_prop_diff_neg = []

                """
                if len(angle_diff_pos)>1 or len(angle_diff_neg)>1:
                    coord_list.append(np.append(plot_ang_diff_pos, plot_ang_diff_neg))
                    resid_list.append(np.append(plot_prop_diff_pos, plot_prop_diff_neg))
                    color_list.append(color)
                    lev_list.append(lev)
                """
            else:
                coord_list.append(second_cont[corr_inds])
                resid_list.append(prop_cont[corr_inds])
                color_list.append(color)
                lev_list.append(lev)
                ind_sort = np.argsort(second_cont[corr_inds]) #sorting by azimuth to avoid 'joint' boundaries in plot
                ax.plot(second_cont[corr_inds][ind_sort], 
                        prop_cont[corr_inds][ind_sort], 
                        color=color, lw=lw, zorder=zorder)

            if ax2 is not None:
                x_cont = np.array([X[i] for i in inds_cont])
                y_cont = np.array([Y[i] for i in inds_cont])
            if isinstance(ax2, matplotlib.axes._subplots.Axes): 
                ax2.plot(x_cont[corr_inds], y_cont[corr_inds], color=color, lw=lw*lw_ax2_factor)
            elif isinstance(ax2, list):
                for axi in ax2: 
                    if isinstance(axi, matplotlib.axes._subplots.Axes): axi.plot(x_cont[corr_inds], y_cont[corr_inds], color=color, lw=lw*lw_ax2_factor)

        return [np.asarray(tmp) for tmp in [coord_list, resid_list, color_list, lev_list]]

    @staticmethod
    def make_substructures(ax, twodim=False, gaps=[], rings=[], kinks=[], make_labels=False,
                           kwargs_gaps={}, kwargs_rings={}, kwargs_kinks={}):
        '''Overlay ring-like (if twodim) or vertical lines (if not twodim) to illustrate the radial location of substructures in the disc'''
        kwargs_g = dict(color='0.2', ls='--', lw=1.7, alpha=0.9)
        kwargs_r = dict(color='0.2', ls='-', lw=1.7, alpha=0.9)
        kwargs_k = dict(color='purple', ls=':', lw=2.6, alpha=0.9)
        kwargs_g.update(kwargs_gaps)
        kwargs_r.update(kwargs_rings)
        kwargs_k.update(kwargs_kinks)        
        if twodim:
            phi = np.linspace(0, 2*np.pi, 50)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            for R in gaps: ax.plot(R*cos_phi, R*sin_phi, **kwargs_g)
            for R in rings: ax.plot(R*cos_phi, R*sin_phi, **kwargs_r)
            for R in kinks: ax.plot(R*cos_phi, R*sin_phi, **kwargs_k)
        else:
            for R in gaps: ax.axvline(R, **kwargs_g)
            for R in rings: ax.axvline(R, **kwargs_r)
            for R in kinks: ax.axvline(R, **kwargs_k)
        if make_labels and len(gaps)>0: ax.plot([None], [None], label='Gaps', **kwargs_g)
        if make_labels and len(rings)>0: ax.plot([None], [None], label='Rings', **kwargs_r)
        if make_labels and len(kinks)>0: ax.plot([None], [None], label='Kinks', **kwargs_k)
            
        return ax
        
    @staticmethod
    def make_contour_lev(prop, lev, X, Y, acc_threshold=20): 
        from skimage import measure 
        contour = measure.find_contours(prop, lev)
        inds_cont = np.round(contour[-1]).astype(np.int)
        inds_cont = [tuple(f) for f in inds_cont]
        first_cont = np.array([prop[i] for i in inds_cont])
        corr_inds = np.abs(first_cont-lev) < acc_threshold
        x_cont = np.array([X[i] for i in inds_cont])
        y_cont = np.array([Y[i] for i in inds_cont])
        return x_cont[corr_inds], y_cont[corr_inds], inds_cont, corr_inds

    @staticmethod
    def beams_along_ring(lev, Rgrid, beam_size, X, Y):
        xc, yc, _, _ = Contours.make_contour_lev(Rgrid, lev, X, Y)
        try:
            rc = hypot_func(xc, yc)
            a = np.max(rc)
            b = np.min(rc)
            ellipse_perim = np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b))) #Assuming that disc vertical extent does not distort much the ellipse
            return ellipse_perim/beam_size
        except ValueError: #No contour was found
            return np.inf

    @staticmethod
    def get_average_east_west(resid_list, coord_list, lev_list, 
                              Rgrid, beam_size, X, Y,
                              av_func=np.nanmean, mask_ang=0, resid_thres='3sigma',
                              error_func=True, error_unit=1.0, error_thres=np.inf):
        #resid_thres: None, '3sigma', or list of thresholds with size len(lev_list)        
        nconts = len(lev_list)
        if resid_thres is None: resid_thres = [np.inf]*nconts
        elif resid_thres == '3sigma': resid_thres = [3*np.nanstd(resid_list[i]) for i in range(nconts)] #anything higher than 3sigma is rejected from annulus
        # -np.pi<coord_list<np.pi
        ind_west = [((coord_list[i]<90-mask_ang) & (coord_list[i]>-90+mask_ang)) & (np.abs(resid_list[i]-np.nanmean(resid_list[i])) < resid_thres[i]) for i in range(nconts)]
        ind_east = [((coord_list[i]>90+mask_ang) | (coord_list[i]<-90-mask_ang)) & (np.abs(resid_list[i]-np.nanmean(resid_list[i])) < resid_thres[i]) for i in range(nconts)]
        av_west = np.array([av_func(resid_list[i][ind_west[i]]) for i in range(nconts)])
        av_east = np.array([av_func(resid_list[i][ind_east[i]]) for i in range(nconts)])
        
        if error_func is None: av_west_error, av_east_error = None, None
        else:
            beams_ring_sqrt = np.sqrt([0.5*Contours.beams_along_ring(lev, Rgrid, beam_size, X, Y) for lev in lev_list]) #0.5 because we split the disc in halves
            if callable(error_func): #if error map provided, compute average error per radius, divided by sqrt of number of beams (see Michiel Hogerheijde notes on errors)
                av_west_error, av_east_error = np.zeros(nconts), np.zeros(nconts)
                for i in range(nconts):
                    x_west, y_west, __ = Tools.get_sky_from_disc_coords(lev_list[i], coord_list[i][ind_west[i]]) #MISSING z, incl, PA for the function to work
                    x_east, y_east, __ = Tools.get_sky_from_disc_coords(lev_list[i], coord_list[i][ind_east[i]]) #MISSING z, incl, PA for the function to work
                    error_west = np.array(list(map(error_func, x_west, y_west))).T[0]
                    error_east = np.array(list(map(error_func, x_east, y_east))).T[0]
                    sigma2_west = np.where((np.isfinite(error_west)) & (error_unit*error_west<error_thres) & (error_west>0), (error_unit*error_west)**2, 0)
                    sigma2_east = np.where((np.isfinite(error_east)) & (error_unit*error_east<error_thres) & (error_east>0), (error_unit*error_east)**2, 0)
                    Np_west = len(coord_list[i][ind_west[i]])
                    Np_east = len(coord_list[i][ind_east[i]])
                    av_west_error[i] = np.sqrt(np.nansum(sigma2_west)/Np_west)/beams_ring_sqrt[i]  
                    av_east_error[i] = np.sqrt(np.nansum(sigma2_east)/Np_east)/beams_ring_sqrt[i]        
            else: #compute standard error of mean value
                av_west_error = np.array([np.std(resid_list[i][ind_west[i]], ddof=1) for i in range(nconts)])/beams_ring_sqrt
                av_east_error = np.array([np.std(resid_list[i][ind_east[i]], ddof=1) for i in range(nconts)])/beams_ring_sqrt
                
        return av_west, av_east, av_west_error, av_east_error

    @staticmethod
    def get_average(resid_list, coord_list, lev_list, 
                    Rgrid, beam_size, X, Y,
                    av_func=np.nanmean, mask_ang=0, resid_thres='3sigma',
                    error_func=True, error_unit=1.0, error_thres=np.inf):
        #mask_ang: +- angles to reject around minor axis (i.e. phi=+-90) 
        #resid_thres: None, '3sigma', or list of thresholds with size len(lev_list)        
        frac_annulus = 1.0 #if halves, 0.5; if quadrants, 0.25
        nconts = len(lev_list)
        if resid_thres is None: resid_thres = [np.inf]*nconts #consider all values for the average
        elif resid_thres == '3sigma': resid_thres = [3*np.nanstd(resid_list[i]) for i in range(nconts)] #anything higher than 3sigma is rejected from annulus
        # -np.pi<coord_list<np.pi        
        ind_accep = [(((coord_list[i]<90-mask_ang) & (coord_list[i]>-90+mask_ang)) |
                      ((coord_list[i]>90+mask_ang) | (coord_list[i]<-90-mask_ang))) &
                     (np.abs(resid_list[i]-np.nanmean(resid_list[i]))<resid_thres[i])
                     for i in range(nconts)]
        av_annulus = np.array([av_func(resid_list[i][ind_accep[i]]) for i in range(nconts)])
        
        if error_func is None: av_error = None
        else:
            beams_ring_sqrt = np.sqrt([frac_annulus*Contours.beams_along_ring(lev, Rgrid, beam_size, X, Y) for lev in lev_list])
            if callable(error_func): #if error map provided, compute average error per radius, divided by sqrt of number of beams (see Michiel Hogerheijde notes on errors)
                av_error = np.zeros(nconts)
                for i in range(nconts):
                    x_accep, y_accep, __ = get_sky_from_disc_coords(lev_list[i], coord_list[i][ind_accep[i]]) #MISSING z, incl, PA for the function to work
                    error_accep = np.array(list(map(error_func, x_accep, y_accep))).T[0]
                    sigma2_accep = np.where((np.isfinite(error_accep)) & (error_unit*error_accep<error_thres) & (error_accep>0), (error_unit*error_accep)**2, 0)
                    Np_accep = len(coord_list[i][ind_accep[i]])
                    av_error[i] = np.sqrt(np.nansum(sigma2_accep)/Np_accep)/beams_ring_sqrt[i]  
            else: #compute standard error of mean value
                av_error = np.array([np.std(resid_list[i][ind_accep[i]], ddof=1) for i in range(nconts)])/beams_ring_sqrt
                
        return av_annulus, av_error

    @staticmethod
    def get_average_zones(resid_list, coord_list, lev_list, Rgrid, beam_size, X, Y,
                          az_zones=[[-30, 30], [150,  -150]], join_zones=False, av_func=np.nanmean,
                          resid_thres='3sigma', error_func=True, error_unit=1.0, error_thres=np.inf):
                          
        #resid_thres: None, '3sigma', or list of thresholds with size len(lev_list)
        nconts = len(lev_list)
        nzones = len(az_zones)
        
        if resid_thres is None: resid_thres = [np.inf]*nconts
        elif resid_thres == '3sigma': resid_thres = [3*np.nanstd(resid_list[i]) for i in range(nconts)] #anything higher than 3sigma is rejected from annulus

        make_or = lambda az0, az1: [((coord_list[i]>az0) | (coord_list[i]<az1)) & (np.abs(resid_list[i]-np.nanmean(resid_list[i])) < resid_thres[i]) for i in range(nconts)]
        make_and = lambda az0, az1: [((coord_list[i]>az0) & (coord_list[i]<az1)) & (np.abs(resid_list[i]-np.nanmean(resid_list[i])) < resid_thres[i]) for i in range(nconts)]

        def get_portion_inds(az):
            az0, az1 = az
            if (az0 > az1):
                inds = make_or(az0, az1)
            else:
                inds = make_and(az0, az1)
            return inds

        def get_portion_percent(az):
            az0, az1 = az
            if (az0 > az1):
                if az0 < 0: perc = 1 - (az0-az1)/360.
                else: perc = (180-az0 + 180-np.abs(az1))/360.
            else:
                perc = (az1-az0)/360.
            return perc

        #inds containts lists of indices, one list per zone. Each is a list of lists, with as many lists as nconts (number of radii). Each sublist has different number of indices, the larger the radius (i.e. larger path) the more indices.        
        inds = [get_portion_inds(zone) for zone in az_zones] 
        az_percent = np.array([get_portion_percent(zone) for zone in az_zones])

        if join_zones and nzones>1:
            concat = lambda x,y: x+y
            inds = [[functools.reduce(concat, [ind[i] for ind in inds]) for i in range(nconts)]] #concatenates indices from zones, per radius.
            az_percent = np.sum(az_percent)[None] #array of single number
            nzones = 1
            
        av_on_inds = [np.array([av_func(resid_list[i][ind[i]]) for i in range(nconts)]) for ind in inds]        

        beams_ring_full = [Contours.beams_along_ring(lev, Rgrid, beam_size, X, Y) for lev in lev_list]
        beams_zone_sqrt = [np.sqrt(az_percent*br) for br in beams_ring_full]

        if error_func is None: av_error = None
        else:
            if callable(error_func): #Not yet tested
                #if error map provided, compute average error per radius, divided by sqrt of number of beams (see Michiel Hogerheijde notes on errors)  
                av_error = []
                for i in range(nconts):
                    r_ind = [Tools.get_sky_from_disc_coords(lev_list[i], coord_list[i][ind[i]]) for ind in inds] #MISSING z, incl, PA for the function to work
                    error_ind = [np.array(list(map(error_func, r_ind[j][0], r_ind[j][1]))).T[0] for j in range(nzones)]
                    sigma2_ind = [np.where((np.isfinite(error_ind[j])) & (error_unit*error_ind[j]<error_thres) & (error_ind[j]>0),
                                           (error_unit*error_ind[j])**2,
                                           0)
                                  for j in range(nzones)]
                    np_ind = [len(coord_list[i][ind[i]]) for ind in inds]
                    av_error.append([np.sqrt(np.nansum(sigma2_ind[j])/np_ind[j])/beams_zone_sqrt[i][j] for j in range(nzones) in np_ind])
            else: #compute standard error of mean value 
                av_error = [np.array([np.std(resid_list[i][inds[j][i]], ddof=1)/beams_zone_sqrt[i][j] for i in range(nconts)]) for j in range(nzones)]

        return av_on_inds, av_error    

    
    def make_filaments(prop_2D, R_nonan_up_au, R_inner_au, beam_size_au, distance_pc, dpix_arcsec, **kwargs):
        #FIND FILAMENTS
        #adapt_thresh is the width of the element used for the adaptive thresholding mask.
        # This is primarily the step that picks out the filamentary structure. The element size should be similar to the width of the expected filamentary structure

        #kw_fil_mask = dict(verbose=False, adapt_thresh=50*apu.au, smooth_size=1*beam_size_au*apu.au, size_thresh=100*apu.pix**2, border_masking=False, fill_hole_size=0.01*apu.arcsec**2)
        from fil_finder import FilFinder2D
        from astropy import units as apu

        distance=distance_pc*apu.pc
        ang_scale=dpix_arcsec*apu.arcsec
        R_min=R_inner_au
        
        kw_fil_mask = dict(verbose=False, adapt_thresh=1*beam_size_au*apu.au, smooth_size=0.2*beam_size_au*apu.au, size_thresh=500*apu.pix**2, border_masking=False, fill_hole_size=0.01*apu.arcsec**2)
        kw_fil_mask.update(kwargs)
        Rgrid = R_nonan_up_au
        Rind = (Rgrid>R_min) #& (Rgrid<R_max)
        fil_pos = FilFinder2D(np.where(Rind & (prop_2D>0), np.abs(prop_2D), 0), ang_scale=ang_scale, distance=distance)
        fil_pos.preprocess_image(skip_flatten=True) 
        fil_pos.create_mask(**kw_fil_mask)
        fil_pos.medskel(verbose=False)
        
        fil_neg = FilFinder2D(np.where(Rind & (prop_2D<0), np.abs(prop_2D), 0), ang_scale=ang_scale, distance=distance)
        fil_neg.preprocess_image(skip_flatten=True) 
        fil_neg.create_mask(**kw_fil_mask)
        fil_neg.medskel(verbose=False)
        
        fil_pos.analyze_skeletons(prune_criteria='length')
        fil_neg.analyze_skeletons(prune_criteria='length')
        return fil_pos, fil_neg

    
    
    
class Cube(object):
    def __init__(self, nchan, channels, data, beam=False, beam_kernel=False, tb={'nu': False, 'beam': False, 'full': True}):
        self.nchan = nchan
        self.channels = channels
        self.data = data
        self.point = self.cursor
        self._interactive = self.cursor
        self._interactive_path = self.curve
        if isinstance(beam, Beam): self.beam_info = beam
        if beam_kernel: self.beam_kernel = beam_kernel
        if isinstance(tb, dict): #Should be deprecated and removed from __init__
            if tb['nu'] and tb['beam']: self.data = Tools.get_tb(self.data, tb['nu'], tb['beam'], full=tb['full'])

    @property
    def interactive(self): 
        return self._interactive
          
    @interactive.setter 
    def interactive(self, func): 
        print('Setting interactive function to', func) 
        self._interactive = func

    @interactive.deleter 
    def interactive(self): 
        print('Deleting interactive function') 
        del self._interactive

    @property
    def interactive_path(self): 
        return self._interactive_path
          
    @interactive_path.setter 
    def interactive_path(self, func): 
        print('Setting interactive_path function to', func) 
        self._interactive_path = func

    @interactive_path.deleter 
    def interactive_path(self): 
        print('Deleting interactive_path function') 
        del self._interactive_path

    def ellipse(self):
        pass
    
    def _plot_spectrum_region(self, x0, x1, y0, y1, ax, extent=None, compare_cubes=[], stat_func=np.mean, **kwargs):
        kwargs_spec = dict(where='mid', linewidth=2.5, label=r'x0:%d,x1:%d'%(x0,x1))
        kwargs_spec.update(kwargs)
        v0, v1 = self.channels[0], self.channels[-1]
        def get_ji(x,y):
            pass
        if extent is None:
            j0, i0 = int(x0), int(y0)
            j1, i1 = int(x1), int(y1)
        else: 
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j0 = int(nx*(x0-extent[0])/dx)
            i0 = int(ny*(y0-extent[2])/dy)
            j1 = int(nx*(x1-extent[0])/dx)
            i1 = int(ny*(y1-extent[2])/dy)

        slice_cube = self.data[:,i0:i1,j0:j1]
        spectrum = np.array([stat_func(chan) for chan in slice_cube])
        ncubes = len(compare_cubes)
        if ncubes > 0: 
            slice_comp = [compare_cubes[i].data[:,i0:i1,j0:j1] for i in range(ncubes)]
            cubes_spec = [np.array([stat_func(chan) for chan in slice_comp[i]]) for i in range(ncubes)]

        if np.logical_or(np.isinf(spectrum), np.isnan(spectrum)).all(): return False
        else:
            plot_spec = ax.step(self.channels, spectrum, **kwargs_spec)
            if ncubes > 0:
                alpha = 0.2
                dalpha = -alpha/ncubes
                for i in range(ncubes):
                    ax.fill_between(self.channels, cubes_spec[i], color=plot_spec[0].get_color(), step='mid', alpha=alpha)
                    alpha+=dalpha
            else: ax.fill_between(self.channels, spectrum, color=plot_spec[0].get_color(), step='mid', alpha=0.2)
            return plot_spec
   
    def box(self, fig, ax, extent=None, compare_cubes=[], stat_func=np.mean, **kwargs):
        from matplotlib.widgets import RectangleSelector        
        def onselect(eclick, erelease):
            if eclick.inaxes is ax[0]:
                plot_spec = self._plot_spectrum_region(eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata, 
                                                       ax[1], extent=extent, compare_cubes=compare_cubes, 
                                                       stat_func=stat_func, **kwargs) 
                                                       
                if plot_spec:
                    print('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
                    print('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
                    print('used button  : ', eclick.button)
                    xc, yc = eclick.xdata, eclick.ydata #Left, bottom corner
                    dx, dy = erelease.xdata-eclick.xdata, erelease.ydata-eclick.ydata
                    rect = patches.Rectangle((xc,yc), dx, dy, lw=2, edgecolor=plot_spec[0].get_color(), facecolor='none')
                    ax[0].add_patch(rect)
                    ax[1].legend()
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        def toggle_selector(event):
            print('Key pressed.')
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                print('RectangleSelector deactivated.')
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                print('RectangleSelector activated.')
                toggle_selector.RS.set_active(True)

        rectprops = dict(facecolor='none', edgecolor = 'white',
                         alpha=0.8, fill=False)

        lineprops = dict(color='white', linestyle='-',
                         linewidth=3, alpha=0.8)

        toggle_selector.RS = RectangleSelector(ax[0], onselect, drawtype='box', rectprops=rectprops, lineprops=lineprops)
        cid = fig.canvas.mpl_connect('key_press_event', toggle_selector)
        return toggle_selector.RS

    def _plot_spectrum_cursor(self, x, y, ax, extent=None, compare_cubes=[], **kwargs):
        kwargs_spec = dict(where='mid', linewidth=2.5, label=r'%d,%d'%(x,y))
        kwargs_spec.update(kwargs)
        def get_ji(x,y):
            pass
        if extent is None:
            j, i = int(x), int(y)
        else: 
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j = int(nx*(x-extent[0])/dx)
            i = int(ny*(y-extent[2])/dy)
            
        spectrum = self.data[:,i,j]
        v0, v1 = self.channels[0], self.channels[-1]
        if np.logical_or(np.isinf(spectrum), np.isnan(spectrum)).all(): return False
        else:
            #plot_fill = ax.fill_between(self.channels, spectrum, alpha=0.1)
            plot_spec = ax.step(self.channels, spectrum, **kwargs_spec)
            ncubes = len(compare_cubes)
            if ncubes > 0:
                alpha = 0.2
                dalpha = -alpha/ncubes
                for cube in compare_cubes: 
                    ax.fill_between(self.channels, cube.data[:,i,j], color=plot_spec[0].get_color(), step='mid', alpha=alpha)
                    alpha+=dalpha
            else: ax.fill_between(self.channels, spectrum, color=plot_spec[0].get_color(), step='mid', alpha=0.2)
            return plot_spec
        
    #def point(self, *args, **kwargs):
     #   return self.cursor(*args, **kwargs)

    def cursor(self, fig, ax, extent=None, compare_cubes=[], **kwargs):
        def onclick(event):
            if event.button==3: 
                print ('Right click. Disconnecting click event...')
                fig.canvas.mpl_disconnect(cid)
            elif event.inaxes is ax[0]:
                plot_spec = self._plot_spectrum_cursor(event.xdata, event.ydata, ax[1], extent=extent, 
                                                       compare_cubes=compare_cubes, **kwargs) 
                if plot_spec:
                    print('%s click: button=%d, xdata=%f, ydata=%f' %
                          ('double' if event.dblclick else 'single', event.button,
                           event.xdata, event.ydata))
                    ax[0].scatter(event.xdata, event.ydata, marker='D', s=50, facecolor=plot_spec[0].get_color(), edgecolor='k')
                    ax[1].legend(frameon=False, handlelength=0.7, fontsize=MEDIUM_SIZE-1)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        return cid

    def _plot_beam(self, ax):
        x_fwhm = self.beam_kernel.model.x_fwhm
        y_fwhm = self.beam_kernel.model.y_fwhm
        ny_pix, nx_pix = np.shape(self.data[0])
        ellipse = patches.Ellipse(xy = (0.05,0.05), angle = 90+self.beam_info.pa.value,
                                  width=x_fwhm/nx_pix, height=y_fwhm/ny_pix, lw=1, fill=True, 
                                  fc='gray', ec='k', transform=ax.transAxes)
        ax.add_artist(ellipse)
        
    def surface(self, ax, *args, **kwargs): return Contours.emission_surface(ax, *args, **kwargs)

    def show(self, extent=None, chan_init=0, compare_cubes=[], cursor_grid=True, cmap='gnuplot2_r',
             int_unit=r'Intensity [mJy beam$^{-1}$]', pos_unit='Offset [au]', vel_unit=r'km s$^{-1}$',
             show_beam=False, surface={'args': (), 'kwargs': {}}, **kwargs):
        from matplotlib.widgets import Button, Cursor, Slider
        v0, v1 = self.channels[0], self.channels[-1]
        dv = v1-v0
        fig, ax = plt.subplots(ncols=2, figsize=(12,5))
        plt.subplots_adjust(wspace=0.25)

        y0, y1 = ax[1].get_position().y0, ax[1].get_position().y1
        axcbar = plt.axes([0.47, y0, 0.03, y1-y0])
        max_data = np.nanmax([self.data]+[comp.data for comp in compare_cubes])
        ax[0].set_xlabel(pos_unit)
        ax[0].set_ylabel(pos_unit)
        ax[1].set_xlabel('l.o.s velocity [%s]'%vel_unit)
        PlotTools.mod_major_ticks(ax[0], axis='both', nbins=5)
        ax[0].tick_params(direction='out')
        ax[1].tick_params(direction='in', right=True, labelright=False, labelleft=False)
        axcbar.tick_params(direction='out')
        ax[1].set_ylabel(int_unit, labelpad=15)
        ax[1].yaxis.set_label_position('right')
        ax[1].set_xlim(v0-0.1, v1+0.1)
        vmin, vmax = -1*max_data/100, 0.7*max_data#0.8*max_data#
        ax[1].set_ylim(vmin, vmax)
        #ax[1].grid(lw=1.5, ls=':')
        cmap = copy.copy(plt.get_cmap(cmap))
        cmap.set_bad(color=(0.9,0.9,0.9))

        if show_beam and self.beam_kernel: self._plot_beam(ax[0])

        img = ax[0].imshow(self.data[chan_init], cmap=cmap, extent=extent, origin='lower', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img, cax=axcbar)
        img.cmap.set_under('w')
        current_chan = ax[1].axvline(self.channels[chan_init], color='black', lw=2, ls='--')
        text_chan = ax[1].text((self.channels[chan_init]-v0)/dv, 1.02, #Converting xdata coords to Axes coords 
                               '%4.1f %s'%(self.channels[chan_init], vel_unit), ha='center', 
                               color='black', transform=ax[1].transAxes)

        if cursor_grid: cg = Cursor(ax[0], useblit=True, color='lime', linewidth=1.5)

        def get_interactive(func):
            return func(fig, ax, extent=extent, compare_cubes=compare_cubes, **kwargs)
        
        interactive_obj = [get_interactive(self.interactive)]
        #***************
        #SLIDERS
        #***************
        def update_chan(val):
            chan = int(val)
            vchan = self.channels[chan]
            img.set_data(self.data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan-v0)/dv)
            text_chan.set_text('%4.1f %s'%(vchan, vel_unit))
            fig.canvas.draw_idle()

        def update_cubes(val):
            i = int(slider_cubes.val)
            chan = int(slider_chan.val)
            vchan = self.channels[chan]
            if i==0: img.set_data(self.data[chan])
            else: img.set_data(compare_cubes[i-1].data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan-v0)/dv)
            text_chan.set_text('%4.1f km/s'%vchan)
            fig.canvas.draw_idle()

        ncubes = len(compare_cubes)
        if ncubes>0:
            axcubes = plt.axes([0.2, 0.90, 0.24, 0.025], facecolor='0.7')
            axchan = plt.axes([0.2, 0.95, 0.24, 0.025], facecolor='0.7')
            slider_cubes = Slider(axcubes, 'Cube id', 0, ncubes, 
                                  valstep=1, valinit=0, valfmt='%1d', color='dodgerblue')                                  
            slider_chan = Slider(axchan, 'Channel', 0, self.nchan-1, 
                                 valstep=1, valinit=chan_init, valfmt='%2d', color='dodgerblue')        
            slider_cubes.on_changed(update_cubes)
            slider_chan.on_changed(update_cubes)
        else: 
            axchan = plt.axes([0.2, 0.9, 0.24, 0.05], facecolor='0.7')
            slider_chan = Slider(axchan, 'Channel', 0, self.nchan-1, 
                                 valstep=1, valinit=chan_init, valfmt='%2d', color='dodgerblue')        
            slider_chan.on_changed(update_chan)
    
        #*************
        #BUTTONS
        #*************
        def go2cursor(event):
            if self.interactive == self.cursor or self.interactive == self.point: return 0
            interactive_obj[0].set_active(False)
            self.interactive = self.cursor
            interactive_obj[0] = get_interactive(self.interactive)
        def go2box(event):
            if self.interactive == self.box: return 0
            fig.canvas.mpl_disconnect(interactive_obj[0])
            self.interactive = self.box
            interactive_obj[0] = get_interactive(self.interactive)
        def go2trash(event):
            print ('Cleaning interactive figure...')
            plt.close()
            chan = int(slider_chan.val)
            self.show(extent=extent, chan_init=chan, compare_cubes=compare_cubes, 
                      cursor_grid=cursor_grid, int_unit=int_unit, pos_unit=pos_unit, 
                      vel_unit=vel_unit, surface=surface, show_beam=show_beam, **kwargs)
        def go2surface(event):
            self.surface(ax[0], *surface['args'], **surface['kwargs'])
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        box_img = plt.imread(path_icons+'button_box.png')
        cursor_img = plt.imread(path_icons+'button_cursor.jpeg')
        trash_img = plt.imread(path_icons+'button_trash.jpg') 
        surface_img = plt.imread(path_icons+'button_surface.png') 
        axbcursor = plt.axes([0.05, 0.779, 0.05, 0.05])
        axbbox = plt.axes([0.05, 0.72, 0.05, 0.05])
        axbtrash = plt.axes([0.05, 0.661, 0.05, 0.05], frameon=True, aspect='equal')
        bcursor = Button(axbcursor, '', image=cursor_img)
        bcursor.on_clicked(go2cursor)
        bbox = Button(axbbox, '', image=box_img)
        bbox.on_clicked(go2box)
        btrash = Button(axbtrash, '', image=trash_img, color='white', hovercolor='lime')
        btrash.on_clicked(go2trash)
        if len(surface['args'])>0:
            axbsurf = plt.axes([0.005, 0.759, 0.07, 0.07], frameon=True, aspect='equal')
            bsurf = Button(axbsurf, '', image=surface_img)
            bsurf.on_clicked(go2surface)
        plt.show()

    def show_side_by_side(self, cube1, extent=None, chan_init=0, cursor_grid=True, cmap='gnuplot2_r',
                          int_unit=r'Intensity [mJy beam$^{-1}$]', pos_unit='Offset [au]', vel_unit=r'km s$^{-1}$',
                          show_beam=False, surface={'args': (), 'kwargs': {}}, **kwargs):
        from matplotlib.widgets import Button, Cursor, Slider
        compare_cubes = [cube1]
        v0, v1 = self.channels[0], self.channels[-1]
        dv = v1-v0
        fig, ax = plt.subplots(ncols=3, figsize=(17,5))
        plt.subplots_adjust(wspace=0.25)

        y0, y1 = ax[2].get_position().y0, ax[2].get_position().y1
        axcbar = plt.axes([0.63, y0, 0.015, y1-y0])
        max_data = np.nanmax([self.data]+[comp.data for comp in compare_cubes])
        ax[0].set_xlabel(pos_unit)
        ax[0].set_ylabel(pos_unit)
        ax[2].set_xlabel('l.o.s velocity [%s]'%vel_unit)
        PlotTools.mod_major_ticks(ax[0], axis='both', nbins=5)
        ax[0].tick_params(direction='out')
        ax[2].tick_params(direction='in', right=True, labelright=False, labelleft=False)
        axcbar.tick_params(direction='out')
        ax[2].set_ylabel(int_unit, labelpad=15)
        ax[2].yaxis.set_label_position('right')
        ax[2].set_xlim(v0-0.1, v1+0.1)
        vmin, vmax = -1*max_data/100, 0.7*max_data#0.8*max_data#
        ax[2].set_ylim(vmin, vmax)
        cmap = copy.copy(plt.get_cmap(cmap))
        cmap.set_bad(color=(0.9,0.9,0.9))

        if show_beam and self.beam_kernel: self._plot_beam(ax[0])

        img = ax[0].imshow(self.data[chan_init], cmap=cmap, extent=extent, origin='lower', vmin=vmin, vmax=vmax)
        img1 = ax[1].imshow(cube1.data[chan_init], cmap=cmap, extent=extent, origin='lower', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img, cax=axcbar)
        img.cmap.set_under('w')
        img1.cmap.set_under('w')
        current_chan = ax[2].axvline(self.channels[chan_init], color='black', lw=2, ls='--')
        text_chan = ax[2].text((self.channels[chan_init]-v0)/dv, 1.02, #Converting xdata coords to Axes coords 
                               '%4.1f %s'%(self.channels[chan_init], vel_unit), ha='center', 
                               color='black', transform=ax[2].transAxes)

        if cursor_grid: cg = Cursor(ax[0], useblit=True, color='lime', linewidth=1.5)

        def get_interactive(func):
            return func(fig, [ax[0], ax[2]], extent=extent, compare_cubes=compare_cubes, **kwargs)
        
        interactive_obj = [get_interactive(self.interactive)]
        #***************
        #SLIDERS
        #***************
        def update_chan(val):
            chan = int(val)
            vchan = self.channels[chan]
            img.set_data(self.data[chan])
            img1.set_data(cube1.data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan-v0)/dv)
            text_chan.set_text('%4.1f %s'%(vchan, vel_unit))
            fig.canvas.draw_idle()

        ncubes = len(compare_cubes)
        axchan = plt.axes([0.2, 0.9, 0.24, 0.05], facecolor='0.7')
        slider_chan = Slider(axchan, 'Channel', 0, self.nchan-1, 
                             valstep=1, valinit=chan_init, valfmt='%2d', color='dodgerblue')        
        slider_chan.on_changed(update_chan)
    
        #*************
        #BUTTONS
        #*************
        def go2cursor(event):
            if self.interactive == self.cursor or self.interactive == self.point: return 0
            interactive_obj[0].set_active(False)
            self.interactive = self.cursor
            interactive_obj[0] = get_interactive(self.interactive)
        def go2box(event):
            if self.interactive == self.box: return 0
            fig.canvas.mpl_disconnect(interactive_obj[0])
            self.interactive = self.box
            interactive_obj[0] = get_interactive(self.interactive)
        def go2trash(event):
            print ('Cleaning interactive figure...')
            plt.close()
            chan = int(slider_chan.val)
            self.show_side_by_side(cube1, extent=extent, chan_init=chan,
                                   cursor_grid=cursor_grid, int_unit=int_unit, pos_unit=pos_unit, 
                                   vel_unit=vel_unit, surface=surface, show_beam=show_beam, **kwargs)
        def go2surface(event):
            self.surface(ax[0], *surface['args'], **surface['kwargs'])
            self.surface(ax[1], *surface['args'], **surface['kwargs'])
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        box_img = plt.imread(path_icons+'button_box.png')
        cursor_img = plt.imread(path_icons+'button_cursor.jpeg')
        trash_img = plt.imread(path_icons+'button_trash.jpg') 
        surface_img = plt.imread(path_icons+'button_surface.png') 
        axbcursor = plt.axes([0.05, 0.779, 0.05, 0.05])
        axbbox = plt.axes([0.05, 0.72, 0.05, 0.05])
        axbtrash = plt.axes([0.05, 0.661, 0.05, 0.05], frameon=True, aspect='equal')
        bcursor = Button(axbcursor, '', image=cursor_img)
        bcursor.on_clicked(go2cursor)
        bbox = Button(axbbox, '', image=box_img)
        bbox.on_clicked(go2box)
        btrash = Button(axbtrash, '', image=trash_img, color='white', hovercolor='lime')
        btrash.on_clicked(go2trash)
        if len(surface['args'])>0:
            axbsurf = plt.axes([0.005, 0.759, 0.07, 0.07], frameon=True, aspect='equal')
            bsurf = Button(axbsurf, '', image=surface_img)
            bsurf.on_clicked(go2surface)
        plt.show()
        
    """
    #Lasso functions under development
    def _plot_lasso(self, ax, x, y, chan, color=False, show_path=True, extent=None, compare_cubes=[], **kwargs): 
        if len(self._lasso_path) == 0: return
        #for i in range(len(self.lasso_path))
        if extent is None:
            j = x.astype(np.int)
            i = y.astype(np.int)
        else: 
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j = (nx*(x-extent[0])/dx).astype(np.int)
            i = (ny*(y-extent[2])/dy).astype(np.int)
        
        if color: self._plot_path = ax[1].step(np.arange(len(i)), self.data[chan,i,j], color=color)
        else: self._plot_path = ax[1].step(np.arange(len(i)), self.data[chan,i,j])
        self._plot_color = self._plot_path[0].get_color()
        if show_path: self._path_on_cube = ax[0].plot(x,y, color=self._plot_color)
        else: self._path_on_cube = None

    def lasso(self, fig, ax, chan, color=False, show_path=True, extent=None, compare_cubes=[], **kwargs): 
        from matplotlib.widgets import LassoSelector
        canvas = ax[0].figure.canvas        
        def onselect(verts):
            #path = Path(verts)
            canvas.draw_idle()
            self._lasso_path.append(np.array(verts).T)
            self._plot_lasso(ax, *np.array(verts).T, chan, color, show_path, extent, compare_cubes, **kwargs)
            print (verts)
        def disconnect():
            self._lasso_obj.disconnect_events()
            canvas.draw_idle()
        self._lasso_obj = LassoSelector(ax[0], onselect, lineprops={'color': 'lime'})
        def onclick(event):
            if event.button == 3:
                print ('Right click. Disconnecting click event...')
                disconnect()
                fig.canvas.draw()
        cid = fig.canvas.mpl_connect('button_press_event', onclick) 
    """

    def curve(self, ax, x, y, chan, color=False, show_path=True, extent=None, compare_cubes=[], **kwargs): 
        kwargs_curve = dict(linewidth=2.5)#, label=r'x0:%d,x1:%d'%(x0,x1))
        kwargs_curve.update(kwargs)

        if extent is None:
            j = x.astype(np.int)
            i = y.astype(np.int)
        else: 
            nz, ny, nx = np.shape(self.data)
            dx = extent[1] - extent[0]
            dy = extent[3] - extent[2]
            j = (nx*(x-extent[0])/dx).astype(np.int)
            i = (ny*(y-extent[2])/dy).astype(np.int)

        pix_ids = np.arange(len(i))
        path_val = self.data[chan,i,j]
        if color: plot_path = ax[1].step(pix_ids, path_val, where='mid', color=color, **kwargs_curve)
        else: plot_path = ax[1].step(pix_ids, path_val, where='mid', **kwargs_curve)
        plot_color = plot_path[0].get_color()
        if show_path: path_on_cube = ax[0].plot(x, y, color=plot_color, **kwargs_curve)
        else: path_on_cube = None

        cube_fill = []
        plot_fill = None
        ncubes = len(compare_cubes)        
        if ncubes > 0:
            alpha = 0.2
            dalpha = -alpha/ncubes
            for cube in compare_cubes: 
                cube_fill.append(ax[1].fill_between(pix_ids, cube.data[chan,i,j], color=plot_color, step='mid', alpha=alpha))
                alpha+=dalpha
        else: plot_fill = ax[1].fill_between(pix_ids, path_val, color=plot_color, step='mid', alpha=0.2)

        return path_on_cube, plot_path, plot_color, plot_fill, cube_fill

    def show_path(self, x, y, extent=None, chan_init=20, compare_cubes=[], cursor_grid=True,
                  int_unit=r'Intensity [mJy beam$^{-1}$]', pos_unit='au', vel_unit=r'km s$^{-1}$',
                  show_beam=False, **kwargs):
        from matplotlib.widgets import Button, Cursor, Slider
        v0, v1 = self.channels[0], self.channels[-1]
        dv = v1-v0
        fig, ax = plt.subplots(ncols=2, figsize=(12,5))
        plt.subplots_adjust(wspace=0.25)

        y0, y1 = ax[1].get_position().y0, ax[1].get_position().y1
        axcbar = plt.axes([0.47, y0, 0.03, y1-y0])
        max_data = np.max(self.data)
        ax[0].set_xlabel(pos_unit)
        ax[0].set_ylabel(pos_unit)
        ax[1].set_xlabel('Pixel id along path')
        ax[1].tick_params(direction='in', right=True, labelright=False, labelleft=False)
        axcbar.tick_params(direction='out')
        ax[1].set_ylabel(int_unit, labelpad=15)
        ax[1].yaxis.set_label_position('right')
        #ax[1].set_xlim(v0-0.1, v1+0.1)
        #ax[1].set_ylim(-1, max_data)
        vmin, vmax = -max_data/30, max_data
        ax[1].set_ylim(vmin, vmax)
        ax[1].grid(lw=1.5, ls=':')
        cmap = plt.get_cmap('brg')
        cmap.set_bad(color=(0.9,0.9,0.9))

        if show_beam and self.beam_kernel: self._plot_beam(ax[0])

        img = ax[0].imshow(self.data[chan_init], cmap=cmap, extent=extent, origin='lower', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img, cax=axcbar)
        text_chan = ax[1].text(0.15, 1.04, #Converting xdata coords to Axes coords 
                               r'v$_{\rmchan}$=%4.1f %s'%(self.channels[chan_init], vel_unit), ha='center', 
                               color='black', transform=ax[1].transAxes)

        if cursor_grid: cg = Cursor(ax[0], useblit=True, color='lime', linewidth=1.5)
        box_img = plt.imread(path_icons+'button_box.png')
        cursor_img = plt.imread(path_icons+'button_cursor.jpeg')

        def get_interactive(func, chan=chan_init, color=False, show_path=True):
            return func(ax, x, y, chan, color=color, show_path=show_path, extent=extent, compare_cubes=compare_cubes, **kwargs)

        interactive_obj = [get_interactive(self.interactive_path)]
        #***************
        #SLIDERS
        #***************
        def update_chan(val):
            chan = int(val)
            vchan = self.channels[chan]
            img.set_data(self.data[chan])
            text_chan.set_text(r'v$_{\rmchan}$=%4.1f %s'%(vchan, vel_unit))
            path_on_cube, plot_path, plot_color, plot_fill, cube_fill = interactive_obj[0]
            plot_path[0].remove()
            if plot_fill is not None: plot_fill.remove()
            for cbfill in cube_fill: cbfill.remove()
            interactive_obj[0] = get_interactive(self.interactive_path, chan, color=plot_color, show_path=False)
            fig.canvas.draw_idle()

        def update_cubes(val):
            i = int(slider_cubes.val)
            chan = int(slider_chan.val)
            vchan = self.channels[chan]
            if i==0: img.set_data(self.data[chan])
            else: img.set_data(compare_cubes[i-1].data[chan])
            text_chan.set_text(r'v$_{\rmchan}$=%4.1f %s'%(vchan, vel_unit))
            path_on_cube, plot_path, plot_color, plot_fill, cube_fill = interactive_obj[0]
            plot_path[0].remove()
            if plot_fill is not None: plot_fill.remove()
            for cbfill in cube_fill: cbfill.remove()
            interactive_obj[0] = get_interactive(self.interactive_path, chan, color=plot_color, show_path=False)
            fig.canvas.draw_idle()

        ncubes = len(compare_cubes)
        if ncubes>0:
            axcubes = plt.axes([0.2, 0.90, 0.24, 0.025], facecolor='0.7')
            axchan = plt.axes([0.2, 0.95, 0.24, 0.025], facecolor='0.7')
            slider_cubes = Slider(axcubes, 'Cube id', 0, ncubes, 
                                  valstep=1, valinit=0, valfmt='%1d', color='dodgerblue')                                  
            slider_chan = Slider(axchan, 'Channel', 0, self.nchan-1, 
                                 valstep=1, valinit=chan_init, valfmt='%2d', color='dodgerblue')        
            slider_cubes.on_changed(update_cubes)
            slider_chan.on_changed(update_cubes)
        else: 
            axchan = plt.axes([0.2, 0.9, 0.24, 0.05], facecolor='0.7')
            slider_chan = Slider(axchan, 'Channel', 0, self.nchan-1, 
                                 valstep=1, valinit=chan_init, valfmt='%2d', color='dodgerblue')        
            slider_chan.on_changed(update_chan)

        plt.show()

        """
        self._path_on_cube, self._plot_path, self._plot_color = None, None, None
        self._lasso_path = []
        self.interactive_path(fig, ax, chan_init, color=False, show_path=True, extent=extent, compare_cubes=compare_cubes, **kwargs)

        def get_interactive(func, chan=chan_init, color=False, show_path=True):
            #func(fig, ax, chan, color=color, show_path=show_path, extent=extent, compare_cubes=compare_cubes, **kwargs)
            if func == self.lasso:
                return self._plot_lasso(ax, True, True, chan, color=color, show_path=show_path, extent=extent, compare_cubes=compare_cubes, **kwargs)
        
        #interactive_obj = [get_interactive(self.interactive_path)]
        #print (interactive_obj)
        #***************
        #SLIDERS
        #***************
        def update_chan(val):
            chan = int(val)
            vchan = self.channels[chan]
            img.set_data(self.data[chan])
            current_chan.set_xdata(vchan)
            text_chan.set_x((vchan-v0)/dv)
            text_chan.set_text('%4.1f km/s'%vchan)
            #path_on_cube, plot_path, plot_color = interactive_obj[0]
            if self._path_on_cube is not None: 
                self._plot_path[0].remove()
                get_interactive(self.interactive_path, chan, color=self._plot_color, show_path=False)
            fig.canvas.draw_idle()
        """

    def make_fits(self, output, **kw_header):
        from astropy.io import fits
        hdr = fits.Header()
        hdr.update(**kw_header)
        data = np.where(np.isfinite(self.data), self.data, 0)
        fits.writeto(output, data, hdr, overwrite=True)
    
    def make_gif(self, folder='./movie/', extent=None, velocity2d=None, 
                 unit=r'Brightness Temperature [K]',
                 gif_command='convert -delay 10 *int2d* cube_channels.gif'):
        cwd = os.getcwd()
        if folder[-1] != '/': folder+='/'
        os.system('mkdir %s'%folder)
        max_data = np.max(self.data)

        clear_list, coll_list = [], []
        fig, ax = plt.subplots()
        contour_color = 'red'
        cmap = plt.get_cmap('binary')
        cmap.set_bad(color=(0.9,0.9,0.9))
        ax.plot([None],[None], color=contour_color, linestyle='--', linewidth=2, label='Upper surface') 
        ax.plot([None],[None], color=contour_color, linestyle=':', linewidth=2, label='Lower surface') 
        ax.set_xlabel('au')
        ax.set_ylabel('au')
        for i in range(self.nchan):
            vchan = self.channels[i]
            int2d = ax.imshow(self.data[i], cmap=cmap, extent=extent, origin='lower', vmax=max_data)
            cbar = plt.colorbar(int2d)
            cbar.set_label(unit)
            if velocity2d is not None:
                vel_near=ax.contour(velocity2d['upper'], levels=[vchan], colors=contour_color, linestyles='--', linewidths=1.3, extent = extent)
                vel_far=ax.contour(velocity2d['lower'], levels=[vchan], colors=contour_color, linestyles=':', linewidths=1.3, extent = extent)
                coll_list = [vel_near, vel_far]
            text_chan = ax.text(0.7, 1.02, '%4.1f km/s'%vchan, color='black', transform=ax.transAxes)
            ax.legend(loc='upper left')
            plt.savefig(folder+'int2d_chan%04d'%i)
            #print ('Saved channel %d'%i)
            #plt.cla()
            clear_list = [cbar, int2d, text_chan]
            for obj in clear_list: obj.remove()
            for obj in coll_list: 
                for coll in obj.collections:
                    coll.remove()
        plt.close()
        os.chdir(folder)
        print ('Making movie...')
        os.system(gif_command)
        os.chdir(cwd)


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
        x_stddev = beam_kernel.model.x_stddev.value
        y_stddev = beam_kernel.model.y_stddev.value
        self._beam_area = 2*np.pi*x_stddev*y_stddev #see https://en.wikipedia.org/wiki/Gaussian_function, and https://science.nrao.edu/facilities/vla/proposing/TBconv
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
            int2d_full = self._beam_area*convolve(int2d_full, self.beam_kernel, preserve_nan=False)

        return int2d_full

    def get_cube(self, vchannels, velocity2d, intensity2d, linewidth2d, lineslope2d, make_convolve=True, 
                 nchan=None, rms=None, tb={'nu': False, 'beam': False, 'full': True}, return_data_only=False, header=None, **kwargs):

        #from .cube import Cube as Cube2 #header should already be known by here, i.e. it should be an input when General2d is initialised
        
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
            v_near, v_far = self.get_line_profile(vchan, vel2d, linew2d, lineb2d, **kwargs)
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
                """
                inf_mask = np.isinf(int2d_full)
                """
                inf_mask = np.isnan(int2d_full) # Use np.nan_to_num instead
                int2d_full = np.where(inf_mask, noise, int2d_full)                
                int2d_full = self._beam_area*convolve(int2d_full, self.beam_kernel, preserve_nan=False)

            cube.append(int2d_full)
            
        if return_data_only: return np.asarray(cube)
        else: return Cube(nchan, vchannels, np.asarray(cube), beam=self.beam_info, beam_kernel=self.beam_kernel, tb=tb)
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
            model_chan = model_cube[i] #model_cube.data[i] #self.get_channel(vel2d, int2d, linew2d, lineb2d, self.channels[i])
            mask_data = np.isfinite(self.data[i])
            mask_model = np.isfinite(model_chan)
            data = np.where(np.logical_and(mask_model, ~mask_data), 0, self.data[i])
            model = np.where(np.logical_and(mask_data, ~mask_model), 0, model_chan)
            mask = np.logical_and(mask_data, mask_model)
            lnx =  np.where(mask, np.power((data - model)/self.noise_stddev, 2), 0) 
            #lnx = -0.5 * np.sum(lnx2[~np.isnan(lnx2)] * 0.00001)# * self.ivar)
            lnx2 += -0.5 * np.sum(lnx)
            
        #print (new_params, "\nLOG LIKELIHOOD %.4e"%lnx2)
        return lnx2 if np.isfinite(lnx2) else -np.inf
    
     
class General2d(Height, Velocity, Intensity, Linewidth, Lineslope, Tools, Mcmc): #Inheritance should only be from Intensity and Mcmc, the others contain just staticmethods...
    #def __init__(self, grid, prototype=False, subpixels=False, beam=None, skygrid=None, kwargs_beam={}):
    def __init__(self, datacube, dpc, Rmax, Rmin=1.0, prototype=False, subpixels=False, beam=None, kwargs_beam={}):        
        Tools._print_logo()        
        self.flags = {'disc': True, 'env': False}
        self.prototype = prototype

        mgrid = Model(datacube, dpc, Rmax, Rmin=Rmin, prototype=prototype, subpixels=subpixels) #Make model grid (disc and sky grids)
        grid = mgrid.discgrid        
        skygrid = mgrid.skygrid

        self.Rmax = mgrid.Rmax
        self.Rmin = mgrid.Rmin

        self.Rmax_m = mgrid.Rmax.to('m').value
        self.Rmin_m = mgrid.Rmin.to('m').value        
        
        self._beam_info = False #Should be None; on if statements use isinstance(beam, Beam) instead
        self._beam_from = False #Should be deprecated
        self._beam_kernel = False #Should be None
        self._beam_area = False 
        if beam is not None: 
            self.beam_info, self.beam_kernel = Tools._get_beam_from(beam, dpix=grid['step'], **kwargs_beam)

        self._z_upper_func = General2d.z_cone
        self._z_lower_func = General2d.z_cone_neg
        self._velocity_func = General2d.keplerian
        self._intensity_func = General2d.intensity_powerlaw
        self._linewidth_func = General2d.linewidth_powerlaw
        self._lineslope_func = General2d.lineslope_powerlaw
        self._line_profile = General2d.line_profile_bell
        self._use_temperature = False
        self._use_full_channel = False
 
        x_true, y_true = grid['x'], grid['y']
        self.x_true, self.y_true = x_true, y_true
        self.phi_true = grid['phi']
        self.R_true = grid['R']         
        self.mesh = skygrid['meshgrid'] #disc grid will be interpolated onto this sky grid in make_model(). Must match data shape for mcmc. 
        self.grid = grid
        self.skygrid = skygrid
        
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
            self.mc_header, self.mc_kind, self.mc_nparams, self.mc_boundaries_list, self.mc_params_indices = General2d._get_params2fit(self.mc_params, self.mc_boundaries)
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
        self.mc_header, self.mc_kind, self.mc_nparams, self.mc_boundaries_list, self.mc_params_indices = General2d._get_params2fit(self.mc_params, self.mc_boundaries)
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

        Tools._break_line()
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
        Tools._break_line(init='\n', end='\n\n')

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
        
        Tools._break_line(init='\n')
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
        Tools._break_line(init='\n', end='\n\n')

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

    @staticmethod
    def orientation(incl=np.pi/4, PA=0.0, xc=0.0, yc=0.0):
        xc = xc*sfu.au
        yc = yc*sfu.au
        return incl, PA, xc, yc

    def get_projected_coords(self, z_mirror=False, writebinaries=True, 
                             R_nan_val=0, phi_nan_val=10*np.pi, z_nan_val=0):
            
        if self.prototype: 
            Tools._break_line()
            print ('Computing disc upper and lower surface coordinates, projected on the sky plane...')
            print ('Using height and orientation parameters from prototype model:\n')
            pprint.pprint({key: self.params[key] for key in ['height_upper', 'height_lower', 'orientation']})
            Tools._break_line(init='\n')
        
        incl, PA, xc, yc = General2d.orientation(**self.params['orientation'])
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
            phi[side] = np.arctan2(y_grid, x_grid) 
            #Since this one is periodic it has to be recalculated, otherwise the interpolation will screw up things at the boundary -np.pi->np.pi
            # When plotting contours there seems to be in any case some sort of interpolation, so there is still problems at the boundary
            #phi[side] = griddata((x_pro, y_pro), self.phi_true, (self.mesh[0], self.mesh[1]), method='linear')
            z[side] = griddata((x_pro, y_pro), z_true[side], (self.mesh[0], self.mesh[1]), method='linear')
            #r[side] = hypot_func(R[side], z[side])
            if self.Rmax_m is not None: 
                for prop in [R, phi, z]: prop[side] = np.where(np.logical_and(R[side]<self.Rmax_m, R[side]>self.Rmin_m), prop[side], np.nan)

            if writebinaries:
                np.save('%s_R.npy'%side, R[side])
                np.save('%s_phi.npy'%side, phi[side])
                np.save('%s_z.npy'%side, z[side])
                
        R_nonan, phi_nonan, z_nonan = None, None, None
        if R_nan_val is not None: R_nonan = {side: np.where(np.isnan(R[side]), R_nan_val, R[side]) for side in ['upper', 'lower']} #Use np.nan_to_num instead
        if phi_nan_val is not None: phi_nonan = {side: np.where(np.isnan(phi[side]), phi_nan_val, phi[side]) for side in ['upper', 'lower']}
        if z_nan_val is not None: z_nonan = {side: np.where(np.isnan(z[side]), z_nan_val, z[side]) for side in ['upper', 'lower']}

        return R, phi, z, R_nonan, phi_nonan, z_nonan

    def make_disc_axes(self, ax, Rmax=None, surface='upper'): #can be generalised and put outside this class, would require incl, PA and z_func as args.
        if Rmax is None:
            Rmax = self.Rmax.to('au')
        else:
            Rmax = Rmax.to('au')

        R_daxes = np.linspace(0, Rmax, 50)            
        phi_daxes_0 = np.zeros(50)
        phi_daxes_90 = np.zeros(50)+np.pi/2
        phi_daxes_180 = np.zeros(50)+np.pi
        phi_daxes_270 = np.zeros(50)-np.pi/2

        incl, PA, xc, yc = General2d.orientation(**self.params['orientation'])
        xc /= sfu.au
        yc /= sfu.au        

        if surface=='upper':
            z_daxes = self.z_upper_func({'R': R_daxes.to('m').value}, **self.params['height_upper'])/sfu.au 
        elif surface=='lower':
            z_daxes = self.z_lower_func({'R': R_daxes.to('m').value}, **self.params['height_lower'])/sfu.au
        else:
            raise InputError(surface, "Only 'upper' or 'lower' are valid surfaces.")

        kwargs_axes = dict(color='k', ls=':', lw=1.5, dash_capstyle='round', dashes=(0.5, 1.5), alpha=0.7)        
        make_ax = lambda x, y: ax.plot(x, y, **kwargs_axes)
        
        for phi_dax in [phi_daxes_0, phi_daxes_90, phi_daxes_180, phi_daxes_270]:            
            x_cont, y_cont,_ = Tools.get_sky_from_disc_coords(R_daxes.value, phi_dax, z_daxes, incl, PA, xc, yc)
            make_ax(x_cont, y_cont)
    
    def make_model(self, z_mirror=False):                   
        if self.prototype: 
            Tools._break_line()
            print ('Running prototype model with the following parameters:\n')
            pprint.pprint(self.params)
            Tools._break_line(init='\n')

        incl, PA, xc, yc = General2d.orientation(**self.params['orientation'])
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
        return props

    
class Rosenfeld2d(Velocity, Intensity, Linewidth, Tools):
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
        self.flags = {'disc': True, 'env': False}
        self.grid = grid
        self._velocity_func = Rosenfeld2d.keplerian
        self._intensity_func = Rosenfeld2d.intensity_powerlaw
        self._linewidth_func = Rosenfeld2d.linewidth_powerlaw
        self._line_profile = General2d.line_profile_v_sigma
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
