from .utils import FrontendUtils, get_tb

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit


_progress_bar = FrontendUtils._progress_bar

def _gauss(x, *p):
    A1, mu1, sigma1 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))

def _doublegauss_sum(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

def _doublegauss_mask(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    gauss1 = A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))
    gauss2 = A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))
    return np.where(gauss1>=gauss2, gauss1, gauss2)

def _bell(x, *p):
    A1, mu1, sigma1, Ls1 = p
    return A1/(1+np.abs((x-mu1)/sigma1)**(2*Ls1))

def _doublebell_sum(x, *p):
    A1, mu1, sigma1, Ls1, A2, mu2, sigma2, Ls2 = p
    return A1/(1+np.abs((x-mu1)/sigma1)**(2*Ls1)) + A2/(1+np.abs((x-mu2)/sigma2)**(2*Ls2))

def _doublebell_mask(x, *p):
    A1, mu1, sigma1, Ls1, A2, mu2, sigma2, Ls2 = p
    bell1 = A1/(1+np.abs((x-mu1)/sigma1)**(2*Ls1))
    bell2 = A2/(1+np.abs((x-mu2)/sigma2)**(2*Ls2))
    return np.where(bell1>=bell2, bell1, bell2)


def fit_twocomponent(cube, model=None, lw_chans=1.0, lower2upper=1.0, sigma_fit=None,
                     method='doublegaussian', kind='sum'):

    data = cube.data
    vchannels = cube.vchannels
    dv = np.mean(vchannels[1:]-vchannels[:-1])
    lw_sign = np.sign(dv)
    
    nchan, nx, ny = np.shape(data)
    n_one, n_two, n_bad = 0, 0, 0
    n_fit = np.zeros((nx, ny))
    
    peak_up, dpeak_up = np.zeros((2, nx, ny))
    centroid_up, dcentroid_up = np.zeros((2, nx, ny))
    linewidth_up, dlinewidth_up = np.zeros((2, nx, ny))
    lineslope_up, dlineslope_up = np.zeros((2, nx, ny))    
    peak_low, dpeak_low = np.zeros((2, nx, ny))
    centroid_low, dcentroid_low = np.zeros((2, nx, ny))
    linewidth_low, dlinewidth_low = np.zeros((2, nx, ny))
    lineslope_low, dlineslope_low = np.zeros((2, nx, ny))
    
    #MODEL AS INITIAL GUESS?
    if model is None:
        print ('Guessing upper surface properties from data to use them as priors for both upper (primary) and lower (secondary) surface components ...')        
        ind_max = np.nanargmax(data, axis=0)
        I_max = np.nanmax(data, axis=0)
        vel_peak = vchannels[ind_max]
        I_max_upper = I_max
        I_max_lower = lower2upper*I_max
        vel_peak_upper = vel_peak_lower = vel_peak
        lw_upper = lw_lower = lw_chans*dv*np.ones_like(vel_peak)
        ls_upper = ls_lower = 1.5
        
    else:
        vel2d, int2d, linew2d, lineb2d = model.props
        R, phi, z = [model.projected_coords[key] for key in ['R', 'phi', 'z']]
        I_upper = int2d['upper']
        I_lower = int2d['lower']
        print ('Using upper and lower surface properties from discminer model as priors...')
        
        if np.any(np.array(['K', 'Kelvin', 'K ', 'Kelvin ']) == cube.header['BUNIT']): #If input unit is K the raw model intensity must be converted    
            restfreq = cube.header['RESTFRQ']*1e-9
            I_upper = get_tb(I_upper*model.beam_area, restfreq, model.beam_info) 
            I_lower = get_tb(I_lower*model.beam_area, restfreq, model.beam_info) 

        #Priors
        ind_max = np.nanargmax(data, axis=0)
        cube_max = np.take_along_axis(data, ind_max[None], axis=0).squeeze()
        I_max_upper = np.where(np.isnan(I_upper), 1.0*cube_max, I_upper)
        I_max_lower = np.where(np.isnan(I_lower), 0.5*cube_max, I_lower)
        vel_peak_upper = np.where(np.isnan(vel2d['upper']), vchannels[ind_max], vel2d['upper'])
        vel_peak_lower = np.where(np.isnan(vel2d['lower']), vchannels[ind_max], vel2d['lower'])
        lw_upper = lw_sign*np.where(np.isnan(linew2d['upper']), 1.5*dv, linew2d['upper'])
        lw_lower = lw_sign*np.where(np.isnan(linew2d['lower']), 1.5*dv, linew2d['lower'])
        ls_upper = np.where(np.isnan(linew2d['upper']), 1.5, lineb2d['upper'])
        ls_lower = np.where(np.isnan(linew2d['lower']), 1.5, lineb2d['lower'])

    if sigma_fit is None: sigma_func = lambda i,j: None
    else: sigma_func = lambda i,j: sigma_fit[:,i,j]

    #******************************
    #KERNELS AND RELEVANT FUNCTIONS
    if method=='doublegaussian':
        fit_func1d = _gauss
        pfunc_two = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j],
                                 I_max_lower[i,j], vel_peak_lower[i,j], lw_lower[i,j]]
        pfunc_one = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j]]
        if kind=='sum':
            fit_func = _doublegauss_sum
        elif kind=='mask':
            fit_func = _doublegauss_mask
        else: #Raise InputError
            pass
        ishift = 0
        
    elif method=='doublebell':
        fit_func1d = _bell_lineslope
        pfunc_two = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j], ls_upper[i,j],
                                 I_max_lower[i,j], vel_peak_lower[i,j], lw_lower[i,j], ls_lower[i,j]]
        pfunc_one = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j], ls_upper[i,j]]
        
        if kind=='sum':
            fit_func = _doublebell_sum
        elif kind=='mask':
            fit_func = _doublebell_mask_lineslope
        else:
            pass
        ishift = 1
        
    else:
        pass

    idlow = np.array([3,4,5])+ishift #ishift accounts for extra index due to lineslope if using bells
    print ('Fitting two-component function to line profiles from cube...')

    #********
    #MAKE FIT
    for i in range(nx):
        for j in range(ny):
            tmp_data = data[:,i,j]
            try: 
                coeff, var_matrix = curve_fit(fit_func,
                                              vchannels, tmp_data,
                                              p0=pfunc_two(i,j),
                                              ftol=1e-10, xtol=1e-10, gtol=1e-10, method='lm')
                                              #bounds = [(0, -np.inf, -np.inf, 0, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)])
                deltas = np.sqrt(np.abs(np.diag(var_matrix)))
                n_two += 1
                n_fit[i,j] = 2

            except RuntimeError:
                try: 
                    coeff, var_matrix = curve_fit(fit_func1d,
                                                  vchannels, tmp_data,
                                                  p0=pfunc_one(i,j)
                    )
                    coeff = np.append(coeff, coeff)
                    deltas = np.sqrt(np.abs(np.diag(var_matrix)))
                    deltas = np.append(deltas, deltas)
                    n_one += 1
                    n_fit[i,j] = 1
                    
                except RuntimeError:
                    n_bad += 1
                    n_fit[i,j] = 0                    
                    continue
                
            if np.abs(coeff[idlow[1]]-vel_peak_upper[i,j]) < np.abs(coeff[1]-vel_peak_upper[i,j]):
                coeff = np.append(coeff[idlow[0]:idlow[-1]+1+ishift], coeff[0:idlow[0]])
                deltas = np.append(deltas[idlow[0]:idlow[-1]+1+ishift], deltas[0:idlow[0]])

            r"""
            #Sometimes the fit can output upper profiles near to zero intensity and in turn the lower profile is attributed to the bulk of the emission.
            #The following seems to improve things due to this in a few pixels. However, the conditional above accounts for this for the most part already.
            if coeff[0]<=1.0 and coeff[3]>1.0:
                coeff = np.append(coeff[idlow[0]:idlow[-1]+1+ishift], coeff[0:idlow[0]])
                deltas = np.append(deltas[idlow[0]:idlow[-1]+1+ishift], deltas[0:idlow[0]])
            #""" 
            peak_up[i,j] = coeff[0]
            centroid_up[i,j] = coeff[1]
            linewidth_up[i,j] = coeff[2]
            peak_low[i,j] = coeff[idlow[0]]
            centroid_low[i,j] = coeff[idlow[1]]
            linewidth_low[i,j] = coeff[idlow[2]]

            dpeak_up[i,j] = deltas[0]
            dcentroid_up[i,j] = deltas[1]
            dlinewidth_up[i,j] = deltas[2]
            dpeak_low[i,j] = deltas[idlow[0]]
            dcentroid_low[i,j] = deltas[idlow[1]]
            dlinewidth_low[i,j] = deltas[idlow[2]]
            
            #dpeak_up[i,j], dcentroid_up[i,j], dlinewidth_up[i,j],_, dpeak_low[i,j], dcentroid_low[i,j], dlinewidth_low[i,j],_ = deltas 
            
        _progress_bar(int(100*i/nx))
    _progress_bar(100)

    upper = [peak_up, centroid_up, linewidth_up]
    dupper = [dpeak_up, dcentroid_up, dlinewidth_up]
    lower = [peak_low, centroid_low, linewidth_low]
    dlower = [dpeak_low, dcentroid_low, dlinewidth_low]

    print ('\nTwo-component fit did not converge for %.2f%s of the pixels'%(100.0*(n_bad)/(nx*ny),'%'))
    print ('\nA single component was fit for %.2f%s of the pixels'%(100.0*(n_one)/(nx*ny),'%'))
    
    return upper, dupper, lower, dlower, n_fit


def fit_gaussian(cube, lw_chans=1.0, sigma_fit=None, peakmethod='gaussian'):
    """
    Fit Gaussian profiles along the velocity axis of the input datacube.
    
    Parameters
    ----------
    lw_chans : int, optional
        Number of channel widths to take as initial guess for line width

    sigma_fit : array_like, optional 
        2-D array of weights computed for each pixel. Shape must be equal to that of the spatial axes of the input data. 

    peakmethod : str, optional
        Amplitude to return is peak of Gaussian fit ('gaussian') or peak of line profile ('max').

    Returns
    -------
    A,B,C,dA,dB,dC : 2-D arrays
        Amplitude, centre, standard deviation, and their corresponding errors, obtained from Gaussians fitted to the input datacube along its velocity axis, on each spatial pixel.
    
    """    
    data = cube.data
    vchannels = cube.vchannels

    nchan, nx, ny = np.shape(data)
    n_fit = np.zeros((nx, ny))                        
    n_bad = 0
    
    peak, dpeak = np.zeros((nx, ny)), np.zeros((nx, ny))
    centroid, dcent = np.zeros((nx, ny)), np.zeros((nx, ny))
    linewidth, dlinew = np.zeros((nx, ny)), np.zeros((nx, ny))

    ind_max = np.nanargmax(data, axis=0)
    I_max = np.nanmax(data, axis=0)
    vel_peak = vchannels[ind_max]
    dv = lw_chans*np.mean(vchannels[1:]-vchannels[:-1])

    if sigma_fit is None: sigma_func = lambda i,j: None
    else: sigma_func = lambda i,j: sigma_fit[:,i,j]

    print ('Fitting Gaussian profile to pixels (along velocity axis)...')

    for i in range(nx):
        for j in range(ny):
            isfin = np.isfinite(data[:,i,j])
            Imax = I_max[i,j]
            try:
                coeff, var_matrix = curve_fit(_gauss, vchannels[isfin], data[:,i,j][isfin],
                                              p0=[Imax, vel_peak[i,j], dv],
                                              sigma=sigma_func(i,j))
                n_fit[i,j] = 1
                    
            except RuntimeError: 
                n_bad+=1
                n_fit[i,j] = 0                
                continue

            if peakmethod=='gaussian': peak[i,j] = coeff[0]
            else: peak[i,j] = Imax
            centroid[i,j] = coeff[1]
            linewidth[i,j] = coeff[2]
            dpeak[i,j], dcent[i,j], dlinew[i,j] = np.sqrt(np.diag(var_matrix))

        _progress_bar(int(100*i/nx))

    _progress_bar(100)

    print ('\nGaussian fit did not converge for %.2f%s of the pixels'%(100.0*n_bad/(nx*ny),'%'))
    
    return [peak, centroid, linewidth], [dpeak, dcent, dlinew], n_fit
