from .utils import FrontendUtils

import numpy as np
from scipy.optimize import curve_fit

_progress_bar = FrontendUtils._progress_bar

def fit_gaussian(data, vchannels, lw_chan=1.0, sigma_fit=None):
    """
    Fit Gaussian profiles along the velocity axis of the input datacube.
    
    Parameters
    ----------
    lw_chan : int, optional
        Number of channel widths to take as initial guess for line width

    sigma_fit : array_like, optional 
        2-D array of weights computed for each pixel. Shape must be equal to that of the spatial axes of the input data. 

    Returns
    -------
    A,B,C,dA,dB,dC : 2-D arrays
        Amplitude, centre, standard deviation, and their corresponding errors, obtained from Gaussians fitted to the input datacube along its velocity axis, on each spatial pixel.
    
    """
    
    gauss = lambda x, *p: p[0]*np.exp(-(x-p[1])**2/(2.*p[2]**2)) #Kernel

    nchan, nx, ny = np.shape(data)
    peak, dpeak = np.zeros((nx, ny)), np.zeros((nx, ny))
    centroid, dcent = np.zeros((nx, ny)), np.zeros((nx, ny))
    linewidth, dlinew = np.zeros((nx, ny)), np.zeros((nx, ny))

    nbad = 0
    ind_max = np.nanargmax(data, axis=0)
    I_max = np.nanmax(data, axis=0)
    vel_peak = vchannels[ind_max]
    dv = lw_chan*np.mean(vchannels[1:]-vchannels[:-1])

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

        _progress_bar(int(100*i/nx))

    _progress_bar(100)

    print ('\nGaussian fit did not converge for %.2f%s of the pixels'%(100.0*nbad/(nx*ny),'%'))
    
    return peak, centroid, linewidth, dpeak, dcent, dlinew
