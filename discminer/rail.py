"""
rail module
===========
Classes:
"""

from .grid import GridTools

from . import constants as sfc
from . import units as sfu
import matplotlib.pyplot as plt
import numpy as np
import numbers

get_sky_from_disc_coords = GridTools.get_sky_from_disc_coords

class Contours(object):
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
    def make_substructures(ax, twodim=False, polar=False, gaps=[], rings=[], kinks=[], make_labels=False,
                           kwargs_gaps={}, kwargs_rings={}, kwargs_kinks={}, func1d='axvline'):
        '''Overlay ring-like (if twodim) or vertical lines (if not twodim) to illustrate the radial location of substructures in the disc'''
        kwargs_g = dict(color='0.2', ls='--', lw=1.7, dash_capstyle='round', dashes=(3.0, 2.5), alpha=1.0)
        kwargs_r = dict(color='0.2', ls='-', lw=1.7, dash_capstyle='round', alpha=1.0)
        kwargs_k = dict(color='purple', ls=':', lw=2.5, dash_capstyle='round', dashes=(0.5, 1.5), alpha=0.9)
        kwargs_g.update(kwargs_gaps)
        kwargs_r.update(kwargs_rings)
        kwargs_k.update(kwargs_kinks)        
        if twodim:
            nphi = 100
            phi = np.linspace(0, 2*np.pi, nphi)
            if polar:
                for R in gaps: ax.plot(phi, [R]*nphi, **kwargs_g)
                for R in rings: ax.plot(phi, [R]*nphi, **kwargs_r)
                for R in kinks: ax.plot(phi, [R]*nphi, **kwargs_k)                
            else:
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                for R in gaps: ax.plot(R*cos_phi, R*sin_phi, **kwargs_g)
                for R in rings: ax.plot(R*cos_phi, R*sin_phi, **kwargs_r)
                for R in kinks: ax.plot(R*cos_phi, R*sin_phi, **kwargs_k)
        else:
            if func1d=='axvline': func1d=ax.axvline
            elif func1d=='axhline': func1d=ax.axhline            
            for R in gaps: func1d(R, **kwargs_g)
            for R in rings: func1d(R, **kwargs_r)
            for R in kinks: func1d(R, **kwargs_k)
        if make_labels and len(gaps)>0: ax.plot([None], [None], label='Gaps', **kwargs_g)
        if make_labels and len(rings)>0: ax.plot([None], [None], label='Rings', **kwargs_r)
        if make_labels and len(kinks)>0: ax.plot([None], [None], label='Kinks', **kwargs_k)
            
        return ax
        
    @staticmethod
    def make_contour_lev(prop, lev, X, Y, acc_threshold=20): 
        from skimage import measure 
        contour = measure.find_contours(prop, lev)
        inds_cont = np.round(contour[-1]).astype(int)
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

    @staticmethod
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
