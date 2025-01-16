"""
rail module
===========
Classes:
"""

from .grid import GridTools
from .plottools import make_substructures as make_substruct
from .tools.utils import hypot_func
from . import constants as sfc
from . import units as sfu
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.axes import Axes
import numpy as np
from astropy import units as u
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from collections.abc import Iterable
from skimage import measure
import numbers

get_sky_from_disc_coords = GridTools.get_sky_from_disc_coords

class Rail(object):
    def __init__(self, model, prop, coord_levels=None):
        """
        Initialise Rail class. This class provides useful methods to compute azimuthal/radial contours and/or azimuthal averages from an input ``prop`` map projected on the sky.
        The methods in this class use the model disc vertical structure to convert sky to disc coordinates, and therefore all quantities returned are referred to the disc reference frame.

        Parameters
        ----------
        model : `.disc2d.General2d` instance
           Model to get the disc vertical structure from.

        prop : array_like, shape (nx, ny)
           Observable to be analysed, as projected on the sky.
        
        coord_levels : array_like, shape (nlevels,), optional
           Radial or azimuthal levels where *prop* will be mapped.

              - If None, coord_levels are assumed to vary linearly and radially as np.arange(model.Rmin, model.Rmax, beam/4)
              - Does not have any effect on the `Rail.make_filaments` method
        """
        
        _rail_coords = model.projected_coords
        X, Y = model.skygrid['meshgrid'] #regular square grid
        self.X = X/sfu.au 
        self.Y = Y/sfu.au

        self.R = _rail_coords['R'] #disc coord
        self.phi = _rail_coords['phi'] #disc coord
        self.R_nonan = _rail_coords['R_nonan']                
        self.extent = model.skygrid['extent']
        self.dpc = model.dpc
        self.Rmin = model.Rmin
        self.Rmax = model.Rmax        

        self.beam_size = model.beam_size
        self.header = model.header

        self.prop = prop        
        if coord_levels is None:
            self.coord_levels = np.arange(
                model.Rmin.to('au').value,
                model.Rmax.to('au').value,
                model.beam_size.to('au').value/4
            )
        else:
            self.coord_levels = coord_levels

        self._lev_list = None
        self._coord_list = None
        self._resid_list = None
        self._color_list = None

        self.model = model

    @staticmethod
    def make_contour_lev(prop, lev, X, Y, acc_threshold=20): 
        contour = measure.find_contours(prop, lev)
        ind_good = np.argmin([np.abs(lev-prop[tuple(np.round(contour[i][0]).astype(int))]) for i in range(len(contour))]) #get contour id closest to lev              
        inds_cont = np.round(contour[ind_good]).astype(int)
        inds_cont = [tuple(f) for f in inds_cont]
        first_cont = np.array([prop[i] for i in inds_cont])
        corr_inds = np.abs(first_cont-lev) < acc_threshold
        x_cont = np.array([X[i] for i in inds_cont])
        y_cont = np.array([Y[i] for i in inds_cont])
        return x_cont[corr_inds], y_cont[corr_inds], inds_cont, corr_inds
    
    @staticmethod
    def beams_along_annulus(lev, Rgrid, beam_size, X, Y):
        xc, yc, _, _ = Rail.make_contour_lev(Rgrid, lev, X, Y)
        try:
            rc = hypot_func(xc, yc)
            a = np.max(rc)
            b = np.min(rc)
            ellipse_perim = np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b))) #Assumes elliptical path is not affected much by the disc vertical structure
            return ellipse_perim/beam_size
        except ValueError: #No contour found
            return np.inf

    def make_interpolated_lev(self, prop, lev, phi_beam_frac=1/4.):
        phi_step = self.beam_size.to('au').value*phi_beam_frac/lev
        phi_list = np.arange(-np.pi, np.pi, phi_step)
        zp = self.model.z_upper_func({'R': lev*u.au.to('m')}, **self.model.params['height_upper'])*u.m.to('au')
        x_samples, y_samples, z_samples = GridTools.get_sky_from_disc_coords(lev, phi_list, zp, **self.model.params['orientation']) 
    
        x_min, x_max, y_min, y_max = self.extent
        j_samples = (x_samples - x_min) / (x_max - x_min) * (self.X.shape[1] - 1)
        i_samples = (y_samples - y_min) / (y_max - y_min) * (self.X.shape[0] - 1)

        coords = np.vstack([i_samples, j_samples])
        prop_interp = map_coordinates(prop, coords, order=1, mode='nearest')

        return prop_interp, np.degrees(phi_list), x_samples, y_samples

    def prop_along_coords(self,
                          coord_ref=None,
                          surface='upper',
                          ax=None,
                          ax2=None,
                          acc_threshold=10, #0.05
                          max_prop_threshold=np.inf,
                          color_bounds=[200, 400],
                          colors=['red', 'dodgerblue', '#FFB000'],
                          lws=[0.3, 0.3, 0.3], lw_ax2_factor=1,
                          interpgrid=False,
                          fold=False,
                          fold_func=np.subtract):
        """
        Compute azimuthal contours according to the model disc geometry 
        to retrieve and plot information from the input two-dimensional map ``prop``.    

        Parameters
        ----------
        coord_ref : scalar, optional
           The contour at this coordinate will be highlighted in bold black.

        ax : `matplotlib.axes` instance, optional
           ax instance where computed contours are to be shown
                   
        ax2 : `matplotlib.axes` instance (or list of instances), optional
           Additional ax(s) instance(s) to plot the location of contours in the disc. 
           
        acc_threshold : float, optional 
           Threshold to accept points on contours at a given coord_level. If coord_level obtained for a pixel is such that np.abs(level_pixel-level_reference)<acc_threshold the pixel value is accepted.

        max_prop_threshold : float, optional 
           Threshold to accept points of contours. Rejects residuals of the contour if they are >= max_prop_threshold. Useful to reject hot pixels.

        color_bounds : array_like, shape (nbounds,), optional
           Color bounds for contour colors.
           
        colors : array_like, shape (nbounds+1,), optional
           Contour colors within bounds. 

        lws : array_like, shape (nbounds+1), optional
           Contour linewidths within bounds.

        lw_ax2_factor : float, optional
           Linewidth fraction for contours in ax2 with respect to those in ax1.

        interpgrid : bool, optional
           If True, interpolate prop values from native grid into the path of a perfect annulus. Else, fetch prop values from native grid along annular contours found with skimage.measure.

        fold : bool, optional
           If True, subtract residuals by folding along the projected minor axis of the disc. Currently working for azimuthal contours only.
           
        fold_func : function, optional
           If fold, this function is used to operate between folded quadrants. Defaults to np.subtract.
        """

        prop, coord_levels = self.prop, self.coord_levels
        _rail_phi = self.phi[surface] #np.where(self.phi['upper'] < 0.98*np.pi, self.phi['upper'], np.nan)
        coords_azimuthal = [self.R_nonan[surface]/sfu.au, np.degrees(_rail_phi)]
        coords_radial = [] #Missing this, must add the option to allow the user switch between azimuthal or radial contours
        
        X, Y = self.X, self.Y
        coords = coords_azimuthal

        if isinstance(color_bounds, Iterable):
            color_bounds = np.insert([0, np.inf], 1, color_bounds)
            nbounds = len(color_bounds)
        else:
            nbounds = 0
            
        coord_list, lev_list, resid_list, color_list = [], [], [], []

        if np.sum(coord_levels==coord_ref)==0 and coord_ref is not None:
            coord_levels = np.append(coord_levels, coord_ref)

        for levi, lev in enumerate(coord_levels):

            if interpgrid:
                prop_cont, second_cont, x_cont, y_cont = self.make_interpolated_lev(prop, lev)
                corr_inds = slice(None)

            else:
                contour = measure.find_contours(coords[0], lev) #, fully_connected='high', positive_orientation='high')

                if len(contour)==0:
                    print ('no contours found for phi =', lev)
                    continue
                
                ind_good = np.argmin([np.abs(lev-coords[0][tuple(np.round(contour[i][0]).astype(int))]) for i in range(len(contour))]) #get contour id closest to lev
                inds_cont = np.round(contour[ind_good]).astype(int)
                inds_cont = [tuple(f) for f in inds_cont]
                first_cont = np.array([coords[0][i] for i in inds_cont])
                second_cont = np.array([coords[1][i] for i in inds_cont])
                prop_cont = np.array([prop[i] for i in inds_cont])
            
                corr_inds = np.abs(first_cont-lev) < acc_threshold #clean based on acc_threshold
                
                _, tmp = np.unique(second_cont, return_index=True) 
                unique_inds = np.zeros_like(second_cont).astype(bool)
                unique_inds[tmp] = True
                corr_inds = corr_inds & unique_inds #make sure points are not repeated


            if lev == coord_ref:
                zorder=10
            else:
                zorder=np.random.randint(0,10)
            
            if color_bounds is None:
                if isinstance(colors, Iterable) and not isinstance(colors, str):
                    color = colors[levi]
                else:
                    color = colors
                    
                if isinstance(lws, Iterable):
                    lw = lws[levi]
                else:
                    lw = lws
            else:
                for i,bound in enumerate(color_bounds):
                    if lev == coord_ref: 
                        lw = 2.0
                        color = 'k'
                        zorder = 10
                        break
                    elif i!=(nbounds-1):
                        if color_bounds[i] <= lev <= color_bounds[i+1]:
                            lw = lws[i]
                            color = colors[i]                                            
                            break
                    else:
                        lw = lws[i]
                        color = colors[i]                    
                        break
                            
            if fold:
                #if lev < color_bounds[0]: continue
                ref_pos = 90 #Reference axis for positive angles
                ref_neg = -90
                angles = second_cont[corr_inds]
                prop_ = prop_cont[corr_inds]
                angles_pos = angles[angles>=0] #0, 180
                angles_neg = angles[angles<0] #-180, 0
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
                    prop_diff = fold_func(prop_[current_ind][0], prop_[mirror_ind][0])
                    angle_diff_pos.append(angles_pos[i])
                    prop_diff_pos.append(prop_diff)
                angle_diff_pos = np.asarray(angle_diff_pos)
                prop_diff_pos = np.asarray(prop_diff_pos)

                if len(angle_diff_pos)>1:
                    ind_sort_pos = np.argsort(angle_diff_pos)
                    plot_ang_diff_pos = angle_diff_pos[ind_sort_pos]
                    plot_prop_diff_pos = prop_diff_pos[ind_sort_pos]
                    ind_prop_pos = np.abs(plot_prop_diff_pos)<max_prop_threshold
                    if ax is not None:
                        ax.plot(plot_ang_diff_pos[ind_prop_pos], plot_prop_diff_pos[ind_prop_pos],
                                color=color, lw=lw, zorder=zorder)
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
                    prop_diff = fold_func(prop_[current_ind][0], prop_[mirror_ind][0])
                    angle_diff_neg.append(angles_neg[i])
                    prop_diff_neg.append(prop_diff)
                angle_diff_neg = np.asarray(angle_diff_neg)
                prop_diff_neg = np.asarray(prop_diff_neg)

                if len(angle_diff_neg)>1:
                    ind_sort_neg = np.argsort(np.abs(angle_diff_neg))
                    plot_ang_diff_neg = angle_diff_neg[ind_sort_neg]
                    plot_prop_diff_neg = prop_diff_neg[ind_sort_neg]
                    ind_prop_neg = np.abs(plot_prop_diff_neg)<max_prop_threshold
                    if ax is not None:
                        ax.plot(plot_ang_diff_neg[ind_prop_neg], plot_prop_diff_neg[ind_prop_neg],
                                color=color, lw=lw, zorder=zorder)
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
                if interpgrid:
                    coord_list.append(second_cont)
                    resid_list.append(prop_cont)
                    corr_inds = ind_sort = slice(None)
                else:
                    ind_sort = np.argsort(second_cont[corr_inds]) #sorting by azimuth/radius to avoid 'joint' boundaries in plot
                    coord_list.append(second_cont[corr_inds][ind_sort])
                    resid_list.append(prop_cont[corr_inds][ind_sort])

                color_list.append(color)
                lev_list.append(lev)

                if ax is not None:
                    ax.plot(second_cont[corr_inds][ind_sort], prop_cont[corr_inds][ind_sort], color=color, lw=lw, zorder=zorder)

            if ax2 is not None:
                if not interpgrid:
                    x_cont = np.array([X[i] for i in inds_cont])
                    y_cont = np.array([Y[i] for i in inds_cont])
            if isinstance(ax2, Axes):
                ax2.plot(x_cont[corr_inds], y_cont[corr_inds], color=color, lw=lw*lw_ax2_factor)
            elif isinstance(ax2, list):
                for axi in ax2: 
                    if isinstance(axi, Axes):
                        axi.plot(x_cont[corr_inds], y_cont[corr_inds], color=color, lw=lw*lw_ax2_factor)

        self._lev_list = lev_list
        self._coord_list = coord_list
        self._resid_list = resid_list
        self._color_list = color_list

        return [np.asarray(tmp, dtype=dty)
                for tmp,dty in zip([lev_list, coord_list, resid_list, color_list],
                                   [float, object, object, object]
                )
        ]
        
    def get_average(self, surface='upper', 
                    av_func=np.nanmean,
                    mask_ang=0,
                    sigma_thres=np.inf,
                    mask_from_map=None, mask_perc_init=0.2,
                    plot_diagnostics=False, tag='',
                    forward_error=False, error_func=False, error_unit=1.0, error_thres=np.inf,                    
                    **kwargs_along_coords):
        #mask_ang: +- angles to reject around minor axis (i.e. phi=+-90) 

        if self._lev_list is None:
            kwargs_along_coords.update({'surface': surface})
            self.prop_along_coords(**kwargs_along_coords)

        lev_list, coord_list, resid_list = self._lev_list, self._coord_list, self._resid_list
        Rgrid = self.R_nonan[surface]/sfu.au
        X = self.X
        Y = self.Y
        beam_size = self.beam_size.to('au').value
        
        frac_annulus = 1.0 #if halves, 0.5; if quadrants, 0.25
        nconts = len(lev_list)

        if mask_from_map is not None:
            mrail = Rail(self.model, mask_from_map, lev_list)
            _, coord_list2, resid_list2, _ = mrail.prop_along_coords(surface=surface)
            
            resid_thres = []
            ind_accep = []
            mean_list, sigma_list = [], []
            
            for i in range(nconts):
                resid2_nonan = resid_list2[i][~np.isnan(resid_list2[i])]
                psig = int(mask_perc_init*len(resid2_nonan))
                isort = np.argsort(resid2_nonan)
                sigma = np.nanstd(resid2_nonan[isort][psig:-psig])
                mean_val = np.nanmean(resid2_nonan[isort][psig:-psig])
                ind = np.abs(resid_list2[i]-mean_val)<sigma_thres*sigma

                ind_accep.append(
                    (
                        ((coord_list[i]<90-mask_ang) & (coord_list[i]>-90+mask_ang)) |
                        ((coord_list[i]>90+mask_ang) | (coord_list[i]<-90-mask_ang))
                    )
                    & ind
                    )
                mean_list.append(mean_val)
                sigma_list.append(sigma)
                
            if plot_diagnostics:
                fig, ax = plt.subplots(nrows=2, figsize=(12,10))                
                ax0, ax1 = ax
                idiag = (np.array([0.3, 0.6, 0.9])*nconts).astype(int) #ind of radii to be plotted

                for i in idiag:
                    ax0.plot(coord_list2[i], resid_list2[i], lw=2.5, alpha=0.5, label='R=%.1f'%lev_list[i])
                    ax0.scatter(coord_list2[i][ind_accep[i]], resid_list2[i][ind_accep[i]], ec='k', fc='none', s=40, lw=1.2)
                    ax1.plot(coord_list[i], resid_list[i], lw=2.5, alpha=0.5)
                    ax1.scatter(coord_list[i][ind_accep[i]], resid_list[i][ind_accep[i]], ec='k', fc='none', s=40, lw=1.2)

                i = idiag[0]
                yfac = 10                
                ax0.set_ylim(mean_list[idiag[-1]]-yfac*sigma_list[idiag[-1]], mean_list[i]+yfac*sigma_list[i])
                mean_eval_map = np.nanmean(resid_list[i][ind_accep[i]])
                sigma_eval_map = np.nanstd(resid_list[i][ind_accep[i]])
                ax1.set_ylim(mean_eval_map-yfac*sigma_eval_map, mean_eval_map+yfac*sigma_eval_map)

                tick_angles = np.arange(-150, 180, 30)
                ax0.legend(frameon=False, fontsize=15)

                for axi in ax:
                    axi.set_xticks(tick_angles)
                    axi.set_xlabel(r'Azimuth [deg]')                        

                fig.savefig('diagnostics_mask_average_from_map_%s.png'%tag)
                plt.close()
                
        else:
            #anything higher than sigma_thres will be rejected from annulus
            resid_thres = [sigma_thres*np.nanstd(resid_list[i]) for i in range(nconts)] 

            ind_accep = [(((coord_list[i]<90-mask_ang) & (coord_list[i]>-90+mask_ang)) |
                          ((coord_list[i]>90+mask_ang) | (coord_list[i]<-90-mask_ang))) &
                         (np.abs(resid_list[i]-np.nanmean(resid_list[i]))<resid_thres[i])
                         for i in range(nconts)]

        av_annulus = np.array([av_func(resid_list[i][ind_accep[i]]) for i in range(nconts)])

        if error_func is None:
            av_error = None
            
        else:
            beams_ring_sqrt = np.sqrt([frac_annulus*Rail.beams_along_annulus(lev, Rgrid, beam_size, X, Y) for lev in lev_list])
            if callable(error_func): #if error map provided, compute average error per radius, divided by sqrt of number of beams (see Michiel Hogerheijde notes on errors)
                av_error = np.zeros(nconts)
                for i in range(nconts):
                    x_accep, y_accep, __ = get_sky_from_disc_coords(lev_list[i], coord_list[i][ind_accep[i]]) #MISSING z, incl, PA for the function to work
                    error_accep = np.array(list(map(error_func, x_accep, y_accep))).T[0]
                    sigma2_accep = np.where((np.isfinite(error_accep)) & (error_unit*error_accep<error_thres) & (error_accep>0), (error_unit*error_accep)**2, 0)
                    Np_accep = len(coord_list[i][ind_accep[i]])
                    av_error[i] = np.sqrt(np.nansum(sigma2_accep)/Np_accep)/beams_ring_sqrt[i]  
            else: #compute standard error of mean value
                if mask_from_map is not None and forward_error:
                    resid = resid_list2
                else:
                    resid = resid_list
                av_error = np.array([np.std(resid[i][ind_accep[i]], ddof=1) for i in range(nconts)])/beams_ring_sqrt

        return av_annulus, av_error


    def get_average_zones(self, surface='upper',
                          av_func=np.nanmean,
                          az_zones=[[-30, 30], [150,  -150]],
                          fast=True, #If not fast use np.trapz integral instead of av_func to get average values
                          join_zones=False, #Get a single averaged profile from all values in the input zones
                          sigma_thres=np.inf, error_func=False, error_unit=1.0, error_thres=np.inf,
                          **kwargs_along_coords):
                          
        if self._lev_list is None:
            kwargs_along_coords.update({'surface': surface})
            self.prop_along_coords(**kwargs_along_coords)

        lev_list, coord_list, resid_list = self._lev_list, self._coord_list, self._resid_list
        Rgrid = self.R_nonan[surface]/sfu.au
        X = self.X
        Y = self.Y
        beam_size = self.beam_size.to('au').value
        
        nconts = len(lev_list)
        nzones = len(az_zones)

        resid_thres = [sigma_thres*np.nanstd(resid_list[i]) for i in range(nconts)] 

        make_or = lambda az0, az1: [((coord_list[i]>az0) | (coord_list[i]<az1))
                                    & (np.abs(resid_list[i]-np.nanmean(resid_list[i])) < resid_thres[i])
                                    for i in range(nconts)]
        
        make_and = lambda az0, az1: [((coord_list[i]>az0) & (coord_list[i]<az1))
                                     & (np.abs(resid_list[i]-np.nanmean(resid_list[i])) < resid_thres[i])
                                     for i in range(nconts)]

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
            if not fast:
                warnings.warn('Using standard (fast) algorithm to compute average of values within zones altogether')

            #concatenate indices from zones, per radius
            inds = [[functools.reduce(lambda x,y: x+y, [ind[i] for ind in inds]) for i in range(nconts)]] 
            az_percent = np.atleast_1d(np.sum(az_percent))
            nzones = 1

        if fast: #compute 'arithmetic' average using av_func
            av_on_inds = [np.array([av_func(resid_list[i][ind[i]]) for i in range(nconts)]) for ind in inds]        

        else: #Compute average using integral definition (trapezoid seems to do better than simpson)
            av_integral = lambda y,x,dT: np.trapz(y, x=x)/dT # or trapezoid from scipy.integrate
            av_on_inds = []

            for ind in inds:

                av_annulus = []

                for i in range(nconts):
                    ii = ind[i]
                    coords_ii = coord_list[i][ii]
                    resid_ii = resid_list[i][ii]

                    if not len(coords_ii):
                        trap = None
                    else:
                        trap = av_integral(resid_ii, coords_ii, coords_ii[-1]-coords_ii[0]) #dT assumes coords_list is sorted (no matter if ascending or descending)
                    av_annulus.append(trap) 
                av_on_inds.append(np.array(av_annulus))
            
        beams_ring_full = [Rail.beams_along_annulus(lev, Rgrid, beam_size, X, Y) for lev in lev_list]
        beams_zone_sqrt = [np.sqrt(az_percent*br) for br in beams_ring_full]

        if error_func is None:
            av_error = None

        else:
            if callable(error_func): #Not yet tested
                #if error map provided, compute average error per radius, divided by sqrt of number of beams (see Michiel Hogerheijde notes on errors)  
                av_error = []
                for i in range(nconts):
                    r_ind = [get_sky_from_disc_coords(lev_list[i], coord_list[i][ind[i]]) for ind in inds] #MISSING z, incl, PA for the function to work
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

    
    def make_2d_map(self, prop_thres=np.inf, return_coords=False): 
        #compute x,y coords from azimuthal contours and interpolate onto 2D grid
        n_conts = len(self._coord_list)
        R_list = np.array([np.repeat(self._lev_list[i], len(self._coord_list[i]))
                           for i in range(n_conts)], dtype=object)
        R = np.concatenate(R_list).ravel()
        phi = np.radians(np.concatenate(self._coord_list).ravel())
        prop = np.concatenate(self._resid_list).ravel()
        prop[np.abs(prop)>prop_thres] = 0.0
        
        for p in [phi, R, prop]:
            p = np.nan_to_num(p, nan=0.0)
            
        x = R*np.cos(phi)
        y = R*np.sin(phi)
        #Interpolate onto the same regular grid used for the sky plane; could use any other grid 
        prop2D = griddata((x, y), prop, (self.X, self.Y), method='linear') 
        
        if return_coords:
            return x, y, prop, prop2D
        else:
            return prop2D

    def make_filaments(self, surface='upper', fill_neg=np.nan, **kwargs):
        #FIND FILAMENTS
        #kwargs docs at https://fil-finder.readthedocs.io/en/latest/tutorial.html#masking

        from fil_finder import FilFinder2D

        kw_fil_mask = dict(
            verbose=False,
            border_masking=False,
            adapt_thresh=self.beam_size,
            smooth_size=0.2*self.beam_size,
            size_thresh=500*u.pix**2,
            fill_hole_size=0.01*u.arcsec**2
        )
        kw_fil_mask.update(kwargs)
        
        Rgrid = self.R_nonan[surface]        
        R_min_m=self.Rmin.to('m').value
        ang_scale = np.abs(self.header['CDELT1'])*u.Unit(self.header['CUNIT1']) #pix size
                       
        Rind = (Rgrid>R_min_m) #& (Rgrid<R_max)
        fil_pos = FilFinder2D(np.where(Rind & (self.prop>0), np.abs(self.prop), fill_neg), ang_scale=ang_scale, distance=self.dpc)
        fil_pos.preprocess_image(skip_flatten=True) 
        fil_pos.create_mask(**kw_fil_mask)
        fil_pos.medskel(verbose=False)
        
        fil_neg = FilFinder2D(np.where(Rind & (self.prop<0), np.abs(self.prop), fill_neg), ang_scale=ang_scale, distance=self.dpc)
        fil_neg.preprocess_image(skip_flatten=True) 
        fil_neg.create_mask(**kw_fil_mask)
        fil_neg.medskel(verbose=False)

        fil_pos.analyze_skeletons(prune_criteria='length')
        fil_neg.analyze_skeletons(prune_criteria='length')
        return fil_pos, fil_neg

    
class Contours(object):
    @staticmethod
    def emission_surface(ax, R, phi, extent, R_lev=None, phi_lev=None,
                         proj_offset=None, X=None, Y=None, which='both',
                         kwargs_R={}, kwargs_phi={}):
        kwargs_phif = dict(linestyles=':', linewidths=0.4, colors='k')
        kwargs_Rf = dict(linestyles=':', linewidths=0.4, colors='k')
        kwargs_phif.update(kwargs_phi)        
        kwargs_Rf.update(kwargs_R)

        near_nonan = ~np.isnan(R['upper'])

        Rmax = np.nanmax(R['upper'])

        kwargs_phif.update({'extent': extent})
        kwargs_Rf.update({'extent': extent})

        if R_lev is None: R_lev = np.linspace(0.06, 0.97, 4)*Rmax
        else: R_lev = np.sort(R_lev)
        if phi_lev is None: phi_lev = np.radians(np.arange(-170, 171, 20))

        #Splitting phi into pos and neg to try and avoid ugly contours close to -pi and pi
        phi_lev_neg = phi_lev[phi_lev<0] 
        phi_lev_pos = phi_lev[phi_lev>=0]
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
                ax.contour(np.where(R['upper']<R_lev[-1], np.nan, R['lower']), levels=R_lev, **kwargs_Rf)
                ax.contour(phi_pos_near, levels=phi_lev_pos, **kwargs_phif)
                ax.contour(phi_neg_near, levels=phi_lev_neg, **kwargs_phif)
                ax.contour(np.where(near_nonan, np.nan, phi_pos_far), levels=phi_lev_pos, **kwargs_phif)
                ax.contour(np.where(near_nonan, np.nan, phi_neg_far), levels=phi_lev_neg, **kwargs_phif)
            elif which in ['upper', 'up']:
                ax.contour(R['upper'], levels=R_lev, **kwargs_Rf)
                ax.contour(phi_pos_near, levels=phi_lev_pos, **kwargs_phif)
                ax.contour(phi_neg_near, levels=phi_lev_neg, **kwargs_phif)
            elif which in ['lower', 'low']:
                ax.contour(R['lower'], levels=R_lev, **kwargs_Rf)
                ax.contour(phi_pos_far, levels=phi_lev_pos, **kwargs_phif)
                ax.contour(phi_neg_far, levels=phi_lev_neg, **kwargs_phif)
                
    #The following method can be optimised if the contour finding process is separated from the plotting
    # by returning coords_list and inds_cont first, which will allow the user use the same set of contours to plot different props.

    @staticmethod
    def disc_axes(ax, R_list, z_list, incl, PA, xc=0, yc=0, **kwargs_axes):

        kwargs_ax = dict(color='k', ls=':', lw=1.5, dash_capstyle='round', dashes=(0.5, 1.5), alpha=0.7)
        kwargs_ax.update(kwargs_axes)
        
        phi_daxes_0 = np.zeros_like(R_list)
        phi_daxes_90 = np.zeros_like(R_list)+np.pi/2
        phi_daxes_180 = np.zeros_like(R_list)+np.pi
        phi_daxes_270 = np.zeros_like(R_list)-np.pi/2

        isort = np.argsort(R_list)
        make_ax = lambda x, y: ax.plot(x, y, **kwargs_ax)        
        
        for phi_dax in [phi_daxes_0, phi_daxes_90, phi_daxes_180, phi_daxes_270]:            
            x_cont, y_cont,_ = GridTools.get_sky_from_disc_coords(R_list[isort], phi_dax, z_list[isort], incl, PA, xc, yc)
            make_ax(x_cont, y_cont)

    @staticmethod
    def make_substructures(*args, **kwargs): #Backcompat
        __doc__ = make_substruct.__doc__
        return make_substruct(*args, **kwargs)
        
    @staticmethod
    def make_contour_lev(prop, lev, X, Y, acc_threshold=20): 
        contour = measure.find_contours(prop, lev)
        ind_good = np.argmin([np.abs(lev-prop[tuple(np.round(contour[i][0]).astype(int))]) for i in range(len(contour))]) #get contour id closest to lev              
        inds_cont = np.round(contour[ind_good]).astype(int)
        inds_cont = [tuple(f) for f in inds_cont]
        first_cont = np.array([prop[i] for i in inds_cont])
        corr_inds = np.abs(first_cont-lev) < acc_threshold
        x_cont = np.array([X[i] for i in inds_cont]) #Get x sky coords of contour
        y_cont = np.array([Y[i] for i in inds_cont]) #Get y sky coords of contour
        return x_cont[corr_inds], y_cont[corr_inds], inds_cont, corr_inds

    @staticmethod
    def beams_along_ring(lev, Rgrid, beam_size, X, Y):
        xc, yc, _, _ = Contours.make_contour_lev(Rgrid, lev, X, Y)
        try:
            rc = hypot_func(xc, yc)
            a = np.max(rc)
            b = np.min(rc)
            ellipse_perim = np.pi*(3*(a+b)-np.sqrt((3*a+b)*(a+3*b))) #Assumes that the disc vertical extent does not distort much the ellipse
            return ellipse_perim/beam_size
        except ValueError: #No contour found
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
                    x_west, y_west, __ = get_sky_from_disc_coords(lev_list[i], coord_list[i][ind_west[i]]) #MISSING z, incl, PA for the function to work
                    x_east, y_east, __ = get_sky_from_disc_coords(lev_list[i], coord_list[i][ind_east[i]]) #MISSING z, incl, PA for the function to work
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
    def get_average_zones(resid_list, coord_list, lev_list, Rgrid, beam_size, X, Y, fast=True, 
                          az_zones=[[-30, 30], [150,  -150]], join_zones=False, av_func=np.nanmean,
                          resid_thres='3sigma', error_func=True, error_unit=1.0, error_thres=np.inf):
                          
        #resid_thres: None, '3sigma', or list of thresholds with size len(lev_list)
        nconts = len(lev_list)
        nzones = len(az_zones)
        if not fast: join_zones=False
        
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

        if fast: #Compute usual average
            av_on_inds = [np.array([av_func(resid_list[i][ind[i]]) for i in range(nconts)]) for ind in inds]        
        else: #Compute average using integral definition (trapezoid seems to succeed better than simpson)
            av_integral = lambda y,x,dT: np.trapz(y, x=x)/dT # or trapezoid from scipy.integrate
            av_on_inds = []
            for ind in inds:
                av_annulus = []
                for i in range(nconts):
                    ii = ind[i]
                    coords_ii = coord_list[i][ii]
                    resid_ii = resid_list[i][ii]
                    if not len(coords_ii): trap=None
                    else: trap = av_integral(resid_ii, coords_ii, coords_ii[-1]-coords_ii[0]) #dT assumes coords_list is sorted (no matter if ascending or descending)
                    av_annulus.append(trap) 
                av_on_inds.append(np.array(av_annulus))
            
        beams_ring_full = [Contours.beams_along_ring(lev, Rgrid, beam_size, X, Y) for lev in lev_list]
        beams_zone_sqrt = [np.sqrt(az_percent*br) for br in beams_ring_full]

        if error_func is None: av_error = None
        else:
            if callable(error_func): #Not yet tested
                #if error map provided, compute average error per radius, divided by sqrt of number of beams (see Michiel Hogerheijde notes on errors)  
                av_error = []
                for i in range(nconts):
                    r_ind = [get_sky_from_disc_coords(lev_list[i], coord_list[i][ind[i]]) for ind in inds] #MISSING z, incl, PA for the function to work
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
