"""
Pick module
===========
Classes: Pick
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as st
from .rail import Rail
from .tools.utils import weighted_std

import copy
import warnings

def get_neighbour_peaks(var_x, pos_x, var_y, n_clusters=8, detect_thres=3):
    #Everything is referred to the variance sorted wrt x
    ind_x_sort = np.argsort(pos_x)
    var_x = var_x[ind_x_sort]
    pos_x = pos_x[ind_x_sort]
    var_y = var_y[ind_x_sort]
    ind_main = np.argmax(var_y)
    var_nomain = np.delete(var_y, ind_main)
    acc_neigh = []
    delete_list = [ind_main]
    ind_centre = ind_main
    first_neighs = np.sort(var_y[np.abs(np.arange(0, n_clusters) - ind_main)==1])[::-1]

    while(True):
        ind_neigh, = np.where(var_y==first_neighs[0])[0]
        delete_list.append(ind_neigh)
        var_noneigh = np.delete(var_y, delete_list)
        var_std_noneigh = np.std(var_noneigh)
        var_mean_noneigh = np.mean(var_noneigh)
        if first_neighs[0] >= var_mean_noneigh+detect_thres*var_std_noneigh:
            acc_neigh.append(ind_neigh)
            sign = int(np.sign(ind_neigh-ind_main)) #left or right neighbour?
            if len(first_neighs)<2: first_neighs = np.array([var_y[ind_neigh+sign]]) #If already at a border
            elif ind_neigh+sign>=n_clusters or ind_neigh+sign<0: first_neighs = np.array([var_y[ind_centre-sign]]) #If neighbour out of the border
            else:
                first_neighs = np.sort([var_y[ind_neigh+sign], var_y[ind_centre-sign]])[::-1] #else if exactly two neighbours
                ind_centre, = np.where(var_y==first_neighs[1])[0]
                ind_centre = ind_centre - int(np.sign(ind_centre-ind_main)) #If the smallest neigh is on the right of ind_main, then ind_centre=ind_smallest-1
        else: break #if first_neighs[i] < 3sigma then the other peak is also smaller

    if len(acc_neigh)==0:
        var_std = np.std(var_nomain)
        var_mean = np.mean(var_nomain)
    else:
        var_std = var_std_noneigh
        var_mean = var_mean_noneigh

    acc_neigh = np.asarray(acc_neigh).astype(int)
    pos_left = pos_x[ind_main]
    pos_right = pos_x[ind_main]
    var_left, var_right = var_x[ind_main], var_x[ind_main]
    if var_y[ind_main] > var_mean+detect_thres*var_std:
        acc_neigh = np.append(acc_neigh, ind_main)
    if len(acc_neigh)>1:
        ind_left = np.argmin(pos_x[acc_neigh])
        ind_right = np.argmax(pos_x[acc_neigh])
        pos_left = pos_x[acc_neigh[ind_left]]
        var_left = var_x[acc_neigh[ind_left]]
        pos_right = pos_x[acc_neigh[ind_right]]
        var_right = var_x[acc_neigh[ind_right]]

    #accepted peaks are sorted from lowest to highest peak. The returned indices refer to the list of peaks sorted wrt the input x coord
    return acc_neigh, var_std, [pos_left, var_left, pos_right, var_right]


class Pick(Rail):
    def __init__(self, model, prop, coord_levels, **kw_prop_along_coords):
        super().__init__(model, prop, coord_levels)
        kw_pac = dict(fold=True)
        kw_pac.update(kw_prop_along_coords)
        lev, coord, resid, color = self.prop_along_coords(**kw_pac)

        self.lev_list = copy.copy(lev)        
        self.coord_list = copy.copy(coord)
        self.resid_list = copy.copy(resid)
        self.color_list = copy.copy(color)

    def find_peaks(self, phi_min=-85, phi_max=85, detect_thres=3, clean_thres=np.inf, clean_histogram=True, fig_ax_histogram=None, av_global=np.median):
        len_res = len(self.resid_list)
        peak_angle = np.zeros(len_res)
        peak_resid = np.zeros(len_res)
        peak_sign = np.zeros(len_res)
        peak_error = np.zeros(len_res)
        
        for i in np.arange(0, len_res): #1 radius per quadrant (0,90), (-90,0)
            arg90 = (self.coord_list[i] >= phi_min) & (self.coord_list[i] <= phi_max)  #-90, 90
            if np.sum(arg90) == 0: abs_resid = [np.nan] #if arg90 is empty
            else: abs_resid = np.abs(self.resid_list[i][arg90])

            argpeak = np.argmax(abs_resid)
            peak_resid[i] = abs_resid[argpeak]

            if np.sum(arg90) == 0: #if arg90 is empty
                peak_angle[i] = np.nan
                peak_sign[i] = np.nan
            else:
                peak_angle[i] = self.coord_list[i][arg90][argpeak]
                peak_sign[i] = np.sign(self.resid_list[i][arg90][argpeak])

        if clean_thres is not None and np.isfinite(clean_thres):

            got_runtimeerror = False
            
            if clean_histogram:

                from discminer.tools.fit_kernel import _gauss
                from scipy.optimize import curve_fit

                yfunc = lambda x, A, sigma: _gauss(x, A, 0, sigma)
                
                peak_hist = peak_resid[peak_resid < 5] #Initial threshold of 5 km/s, not suited for folded intensities 
                counts, bins = np.histogram(peak_hist, bins=4*int(round(len(peak_resid)**(1/3.)))-1 )
                                
                mbins = 0.5*(bins[1:] + bins[:-1])
                
                try:
                    popt, pcov = curve_fit(yfunc, mbins, counts, p0=[np.max(counts), 0.2])
                    popt = np.abs(popt) #Make sure A and sigma are positive, as our residuals are

                    if fig_ax_histogram is not None:
                        
                        import matplotlib.pyplot as plt

                        fig, ax = fig_ax_histogram
                        ax.stairs(counts, bins, color='dodgerblue', lw=3)
                    
                        xgauss = np.linspace(0, round(np.nanmax(bins)), 100)
                        ygauss = yfunc(xgauss, *popt)
                    
                        ax.plot(xgauss, ygauss, lw=3, c='k')
                        ax.axvline(clean_thres*np.abs(popt[1]), lw=4, c='tomato')
                    
                        ax.set_xlabel('Peak residual [km/s]')
                        ax.set_ylabel('Counts')

                    ii = peak_resid < clean_thres*popt[1]
                    print ('Rejecting %d peak velocity residuals above %.3f km/s (%dsigma)'%(np.sum(~ii), clean_thres*popt[1], clean_thres))
                    
                except RuntimeError as e:
                    print('RuntimeError:', e, '*******Rejection of points through histogram method failed. Trying median(peaks)+thres*sigma(peaks) rejection...')
                    got_runtimeerror = True
                    
            if not clean_histogram or got_runtimeerror:
                rej_thresh = np.nanmedian(peak_resid) + clean_thres*np.nanstd(peak_resid)
                ii = peak_resid < rej_thresh
                print ('Rejecting %d peak velocity residuals above %.3f km/s (median+%dsigma)'%(np.sum(~ii), rej_thresh, clean_thres))

            self.lev_list = self.lev_list[ii]
            self.color_list = self.color_list[ii]
            self.coord_list = self.coord_list[ii]
            self.resid_list = self.resid_list[ii]
            peak_resid = peak_resid[ii]
            peak_angle = peak_angle[ii]
            peak_sign = peak_sign[ii]
            
        #peak_sky_coords = get_sky_from_disc_coords(self.lev_list, np.radians(peak_angle))
        peak_error = np.ones(len(self.lev_list)) #np.array([centroid_errors2d(peak_sky_coords[0][i], peak_sky_coords[1][i])[0] for i in range(len_res)])
        #peak_error[peak_error/peak_resid>2] = 0
        peak_weight = np.where(peak_error==0, 0, 1/peak_error)

        #********************
        #FIND GLOBAL PEAK
        #********************
        peak_mean = np.nansum(peak_weight*peak_resid)/np.nansum(peak_weight)
        peak_std = weighted_std(peak_resid, peak_weight, weighted_mean=peak_mean)
        for stdi in np.arange(detect_thres+1)[::-1]: #If it fails to find peaks above the threshold continue and try the next, lower, sigma threshold
            ind_global_peak = peak_resid > peak_mean+stdi*peak_std
            if np.nansum(ind_global_peak) > 0:
                print ('Global peak residual, dv=%.2f, found above %d sigma from mean value...'%(np.max(peak_resid), stdi))                
                break

        """
        self.peak_global_val = np.max(peak_resid[ind_global_peak])
        self.peak_global_angle = np.median(peak_angle[ind_global_peak])
        self.peak_global_radius = np.median(self.lev_list[ind_global_peak])
        """

        #Mean and std from all points
        self.peak_mean = peak_mean
        self.peak_std = peak_std
        
        #Global peak into
        ind_max = np.argmax(peak_resid[ind_global_peak])
        self.peak_global_val = peak_resid[ind_global_peak][ind_max]
        self.peak_global_sign = peak_sign[ind_global_peak][ind_max]
        self.peak_global_angle = peak_angle[ind_global_peak][ind_max]
        self.peak_global_radius = self.lev_list[ind_global_peak][ind_max]
        self.peak_global_sigma = (self.peak_global_val-peak_mean)/peak_std
        
        #Location, amplitude and other properties, for all points
        nonan = ~np.isnan(peak_resid)
        
        self.peak_resid = peak_resid[nonan]
        self.peak_angle = peak_angle[nonan]
        self.peak_sign = peak_sign[nonan]
        self.peak_error = peak_error[nonan]
        self.peak_weight = peak_weight[nonan]

        self.lev_list = self.lev_list[nonan]        
        self.color_list = self.color_list[nonan]
        self.coord_list = self.coord_list[nonan]
        self.resid_list = self.resid_list[nonan]
        

    def _make_clusters(self, n_clusters, axis='phi'): 

        #************************
        #APPLY K-MEANS CLUSTERING
        #************************
        n_peaks = len(self.peak_resid)
        if axis=='phi': peak_both = np.array([self.peak_angle, self.peak_resid]).T
        elif axis=='R': peak_both = np.array([self.lev_list, self.peak_resid]).T
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(peak_both)
        kcenters = kmeans.cluster_centers_
        klabels = kmeans.labels_

        npoints_clust = np.zeros(n_clusters)
        for i in range(n_clusters): npoints_clust[i]=np.sum(klabels==i)

        ind_cluster_peak = np.argmax(kcenters[:,1])
        peak_cluster_nomax = np.sort(kcenters[:,1])[:-1]
        peak_cluster_mean = np.mean(peak_cluster_nomax)
        peak_cluster_std = np.std(peak_cluster_nomax)
        peak_cluster_angle = kcenters[:,0][ind_cluster_peak]

        #**********************
        #FIND CLUSTER VARIANCES
        #**********************
        variance_x, variance_y = [], []
        percentiles = [5, 67] #95%, 68% and 33% contours
        #kde_color = '#FFB000'
        #kde_cmap = get_cmap_from_color(kde_color, lev=len(percentiles)) #'#DC267F' #'#FFB000' #'#648FFF'

        if axis=='phi': xmin_c, xmax_c = -90*1.05, 90*1.05
        elif axis=='R': xmin_c, xmax_c = np.nanmin(self.lev_list), np.nanmax(self.lev_list)
        ymin_c, ymax_c = -0.3*self.peak_global_val, 1.5*self.peak_global_val
        xx, yy = np.mgrid[xmin_c:xmax_c:200j, ymin_c:ymax_c:200j]

        for i in range(n_clusters):
            x = peak_both[:,0][klabels == i]
            y = peak_both[:,1][klabels == i]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y]) #shape (2, ndata)
            try:
                kernel = st.gaussian_kde(values)#, weights=peak_weight[klabels==i])
                perr = np.sqrt(np.diag(kernel.covariance))
                variance_x.append((kernel.covariance[0,0]**0.5/kernel.factor)**2) #True variance: std**2
                variance_y.append(kernel.covariance[1,1])
            except: #when too few points in cluster
                variance_x.append(0)
                variance_y.append(0)
                continue

            f = np.reshape(kernel(positions).T, xx.shape)
            levels = [st.scoreatpercentile(kernel(kernel.resample(1000)), per) for per in percentiles]
            levels.append(np.amax(f))

        #********************
        variance_x = np.asarray(variance_x)
        variance_y = np.asarray(variance_y) * npoints_clust*(n_clusters/n_peaks) #weighting by number of points in clusters
        if axis=='phi': self.klabels_phi = klabels
        if axis=='R': self.klabels_R = klabels
        return kcenters, variance_x, variance_y

    def find_clusters(self, n_phi=8, n_R=8, detect_thres=3):

        kcenters_phi, variance_phi_x, variance_phi_y = self._make_clusters(n_phi, axis='phi')
        kcenters_R, variance_R_x, variance_R_y = self._make_clusters(n_R, axis='R')

        #***************************
        #FIND PEAK AZIMUTHAL CLUSTER VARIANCES
        #***************************
        acc_peaks_phi, var_std_phi, var_width_phi = get_neighbour_peaks(variance_phi_x, kcenters_phi[:,0], variance_phi_y, n_clusters=n_phi, detect_thres=detect_thres)
        print ("accepted variance peaks on PHI:", acc_peaks_phi)

        var_colors_phi = np.array(['1.0']*n_phi).astype('<U9')
        var_colors_phi[acc_peaks_phi] = 'palegreen' #kde_color
        ind_variance_peak_phi = np.argmax(variance_phi_y)
        variance_nomax = np.sort(variance_phi_y)[:-len(acc_peaks_phi)]
        peak_variance_std_phi = var_std_phi #std of variances except those from accepted peaks np.std(variance_nomax)
        self.peak_variance_phi = kcenters_phi[:,0][ind_variance_peak_phi]

        kc_indsort_phi = np.argsort(kcenters_phi[:,0])
        kcent_sort_phi = kcenters_phi[:,0][kc_indsort_phi]
        kcent_sort_vel_phi = kcenters_phi[:,1][kc_indsort_phi]
        var_y_sort_phi = variance_phi_y[kc_indsort_phi]
        self.var_nopeaks_phi = np.delete(var_y_sort_phi, acc_peaks_phi)
        peak_variance_mean_phi = np.mean(self.var_nopeaks_phi) #mean of variances excluding those from accepted peaks

        #peak_variance_sigmas = (self.variance_y[ind_variance_peak]-peak_variance_mean)/peak_variance_std #Considering only peak accepted variance
        self.peak_variance_sigmas_phi_mean = (np.mean(var_y_sort_phi[acc_peaks_phi])-peak_variance_mean_phi)/peak_variance_std_phi #Considering mean std of all accepted peaks
        self.peak_variance_sigmas_phi = (var_y_sort_phi[acc_peaks_phi]-peak_variance_mean_phi)/peak_variance_std_phi #std of each accepted peak        
        #***************************
        #FIND PEAK RADIAL CLUSTER VARIANCES
        #***************************
        acc_peaks_R, var_std_R, var_width_R = get_neighbour_peaks(variance_R_x, kcenters_R[:,0], variance_R_y, n_clusters=n_R, detect_thres=detect_thres)
        print ("accepted variance peaks on R:", acc_peaks_R)

        var_colors_R = np.array(['1.0']*n_R).astype('<U9')
        var_colors_R[acc_peaks_R] = 'palegreen' #kde_color
        ind_variance_peak_R = np.argmax(variance_R_y)
        variance_nomax = np.sort(variance_R_y)[:-len(acc_peaks_R)]
        peak_variance_std_R = var_std_R #std of variances except those from accepted peaks np.std(variance_nomax)
        self.peak_variance_R = kcenters_R[:,0][ind_variance_peak_R]

        kc_indsort_R = np.argsort(kcenters_R[:,0])
        kcent_sort_R = kcenters_R[:,0][kc_indsort_R]
        kcent_sort_vel_R = kcenters_R[:,1][kc_indsort_R]
        var_y_sort_R = variance_R_y[kc_indsort_R]
        self.var_nopeaks_R = np.delete(var_y_sort_R, acc_peaks_R)
        peak_variance_mean_R = np.mean(self.var_nopeaks_R) #mean of variances excluding those from accepted peaks

        #peak_variance_sigmas = (self.variance_y[ind_variance_peak]-peak_variance_mean)/peak_variance_std #Considering only peak accepted variance
        self.peak_variance_sigmas_R_mean = (np.mean(var_y_sort_R[acc_peaks_R])-peak_variance_mean_R)/peak_variance_std_R #Considering mean std of all accepted peaks
        self.peak_variance_sigmas_R = (var_y_sort_R[acc_peaks_R]-peak_variance_mean_R)/peak_variance_std_R #std of each accepted peak        

        #**************************
        self.acc_peaks_phi = acc_peaks_phi
        self.acc_peaks_R = acc_peaks_R
        
        self.kcent_sort_phi = kcent_sort_phi
        self.var_y_sort_phi = var_y_sort_phi

        self.kcent_sort_R = kcent_sort_R
        self.var_y_sort_R = var_y_sort_R

        self.var_colors_phi = var_colors_phi
        self.var_colors_R = var_colors_R

        self.kc_indsort_phi = kc_indsort_phi
        self.kc_indsort_R = kc_indsort_R

        self.kcent_sort_vel_R = kcent_sort_vel_R
        self.kcent_sort_vel_phi = kcent_sort_vel_phi

        #Background properties
        self.peak_variance_std_R = peak_variance_std_R
        self.peak_variance_mean_R = peak_variance_mean_R
        self.peak_variance_std_phi = peak_variance_std_phi
        self.peak_variance_mean_phi = peak_variance_mean_phi
        
        #Weighted mean location of accepted clusters
        self.acc_phi = np.sum(kcent_sort_phi[acc_peaks_phi] * self.peak_variance_sigmas_phi) / np.sum(self.peak_variance_sigmas_phi)                
        self.acc_R = np.sum(kcent_sort_R[acc_peaks_R] * self.peak_variance_sigmas_R) / np.sum(self.peak_variance_sigmas_R)

    def writetxt(self, filename='pick_summary.txt'):

        arr_global_peak = [
            [
                self.peak_global_angle,
                self.peak_global_radius,
                self.peak_global_val*self.peak_global_sign,
                self.peak_global_sigma,
                self.peak_mean,
                self.peak_std,
                'global_peak'
            ]
        ]
        
        arr_clusters_phi = []        
        for n, ind in enumerate(self.acc_peaks_phi):
            arr_clusters_phi.append(
                [
                    self.kcent_sort_phi[ind],
                    np.nan,
                    self.var_y_sort_phi[ind],
                    self.peak_variance_sigmas_phi[n],
                    self.peak_variance_mean_phi,
                    self.peak_variance_std_phi,
                    'cluster_phi_%d'%n
                ]
            )

        arr_clusters_R = []            
        for n, ind in enumerate(self.acc_peaks_R):
            arr_clusters_R.append(
                [                    
                    np.nan,
                    self.kcent_sort_R[ind],
                    self.var_y_sort_R[ind],
                    self.peak_variance_sigmas_R[n],
                    self.peak_variance_mean_R,
                    self.peak_variance_std_R,
                    'cluster_R_%d'%n
                ]
            )
            
        arr_clusters_acc = [
            [
                self.acc_phi,
                self.acc_R,
                np.nan,
                np.nan,
                np.nan,
                np.nan,                
                'weighted_mean_centre_from_accepted_clusters'
            ]
        ]
            
        arr_tot = np.asarray(arr_global_peak + arr_clusters_phi + arr_clusters_R + arr_clusters_acc, dtype=object).squeeze()

        np.savetxt(
            filename,
            arr_tot,
            fmt=('%.2f %.2f %.2e %.2f %.2e %.2e %s').split(),
            header='phi[deg]\tR[au]\tvalue\tsigma\tmean_bckg\tstd_bckg\tcomments',
            delimiter='\t'
        )
        
