from .utils import FrontendUtils, InputError, FITSUtils, get_tb

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

import copy

_progress_bar = FrontendUtils._progress_bar

def _gauss(x, *p):
    A1, mu1, sigma1, *rest = p
    A2 = rest[0] if rest else 0.0
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2

def _doublegauss_sum(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

def _doublegauss_mask(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    gauss1 = A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))
    gauss2 = A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))
    return np.where(gauss1>=gauss2, gauss1, gauss2)

def _bell(x, *p):
    A1, mu1, sigma1, Ls1, *rest = p
    A2 = rest[0] if rest else 0.0
    return A1/(1+np.abs((x-mu1)/sigma1)**(2*Ls1)) + A2

def _doublebell_sum(x, *p):
    A1, mu1, sigma1, Ls1, A2, mu2, sigma2, Ls2 = p
    return A1/(1+np.abs((x-mu1)/sigma1)**(2*Ls1)) + A2/(1+np.abs((x-mu2)/sigma2)**(2*Ls2))

def _doublebell_mask(x, *p):
    A1, mu1, sigma1, Ls1, A2, mu2, sigma2, Ls2 = p
    bell1 = A1/(1+np.abs((x-mu1)/sigma1)**(2*Ls1))
    bell2 = A2/(1+np.abs((x-mu2)/sigma2)**(2*Ls2))
    return np.where(bell1>=bell2, bell1, bell2)

def _not_available(method):
    raise InputError(method, "requested moment method is currently unavailable")

def get_channels_from_parcube(parcube_up, parcube_low, vchannels, method='doublebell', kind='mask', n_fit=None):
    
    if parcube_up is None and 'double' in method:
        parcube_up = np.zeros_like(parcube_low)
    
    if parcube_low is None and 'double' in method:
        parcube_low = np.zeros_like(parcube_up)

    nx, ny = parcube_up.shape[1:]
    intensity = np.zeros((len(vchannels), ny, nx))
        
    par_double = lambda i,j: np.append(parcube_up[:,i,j], parcube_low[:,i,j])
    par_single = lambda i,j: parcube_up[:,i,j]
    
    if 'double' in method:
        pars = par_double
    else:
        pars = par_single

    #******************************
    #KERNELS AND RELEVANT FUNCTIONS
    if method=='doublegaussian':
        fit_func1d = _gauss
        if kind=='sum':
            fit_func = _doublegauss_sum
        elif kind=='mask':
            fit_func = _doublegauss_mask
        else:
            raise InputError(kind, "kind must be 'mask' or 'sum'")

    elif method=='doublebell':
        fit_func1d = _bell
        if kind=='sum':
            fit_func = _doublebell_sum
        elif kind=='mask':
            fit_func = _doublebell_mask
        else:
            raise InputError(kind, "kind must be 'mask' or 'sum'")

    elif method=='gaussian':
        fit_func = _gauss

    elif method=='bell':
        fit_func = _bell
        
    else:
        _not_available(method)

    #***********************
    #EVALUATE PARS ON KERNEL
    for i in range(ny):
        for j in range(nx):
            intensity[:,i,j] = fit_func(vchannels, *pars(i,j))
    
    return intensity


def fit_twocomponent(cube, model=None, lw_chans=1.0, lower2upper=1.0,
                     method='doublegaussian', kind='mask', sigma_thres=5,
                     sigma_fit=None, mask=None, planck=False,
                     niter=4, neighs=5, av_func=np.nanmedian,
                     mask_radially=True
):

    data = cube.data    
    vchannels = cube.vchannels
    dv = np.mean(vchannels[1:]-vchannels[:-1])
    lw_sign = np.sign(dv)
    
    nchan, nx, ny = np.shape(data)
    n_one, n_two, n_bad, n_hot, n_mask = 0, 0, 0, 0, 0
    n_fit = np.zeros((nx, ny))
    
    peak_up, dpeak_up = np.zeros((2, nx, ny))
    centroid_up, dcentroid_up = np.zeros((2, nx, ny))
    linewidth_up, dlinewidth_up = np.zeros((2, nx, ny))
    lineslope_up, dlineslope_up = np.zeros((2, nx, ny))    
    peak_low, dpeak_low = np.zeros((2, nx, ny))
    centroid_low, dcentroid_low = np.zeros((2, nx, ny))
    linewidth_low, dlinewidth_low = np.zeros((2, nx, ny))
    lineslope_low, dlineslope_low = np.zeros((2, nx, ny))
    
    is_dbell = method=='doublebell'
    
    #MODEL AS INITIAL GUESS?
    if model is None:
        print ('Guessing upper surface properties from data to use them as seeds for both upper (primary) and lower (secondary) surface components ...')        
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
        I_upper = int2d['upper']*cube.beam_area
        I_lower = int2d['lower']*cube.beam_area
        print ('Using upper and lower surface properties from discminer model as initial guesses...')
        
        if np.any(np.array(['K', 'Kelvin', 'K ', 'Kelvin ']) == cube.header['BUNIT']): #If input unit is K the raw model intensity must be converted    
            r"""
            mheader = copy.copy(cube.header)
            mheader["BUNIT"] = "beam-1 Jy"
            I_upper = FITSUtils._convert_to_tb(I_upper*cube.beam_area, mheader, cube.beam, planck=False, writefits=False)
            #"""
            restfreq = cube.header['RESTFRQ']*1e-9
            I_upper = get_tb(1e3*I_upper, restfreq, cube.beam, full=planck) #mJy/beam-->K
            I_lower = get_tb(1e3*I_lower, restfreq, cube.beam, full=planck) 

        #Initial guesses
        ind_max = np.nanargmax(data, axis=0)
        cube_max = np.take_along_axis(data, ind_max[None], axis=0).squeeze()
        I_max_upper = np.where(np.isnan(I_upper), 1.0*cube_max, I_upper)
        I_max_lower = np.where(np.isnan(I_lower), 0.5*cube_max, I_lower)
        vel_peak_upper = np.where(np.isnan(vel2d['upper']), vchannels[ind_max], vel2d['upper'])
        vel_peak_lower = np.where(np.isnan(vel2d['lower']), vchannels[ind_max], vel2d['lower'])
        lw_upper = lw_sign*np.where(np.isnan(linew2d['upper']), lw_sign*1.5*dv, linew2d['upper'])
        lw_lower = lw_sign*np.where(np.isnan(linew2d['lower']), lw_sign*1.5*dv, linew2d['lower'])
        ls_upper = np.where(np.isnan(linew2d['upper']), 1.5, 1*lineb2d['upper']) #0.5*
        ls_lower = np.where(np.isnan(linew2d['lower']), 1.5, 1*lineb2d['lower']) #0.5*

    if sigma_fit is None: sigma_func = lambda i,j: None
    else: sigma_func = lambda i,j: sigma_fit[:,i,j]

    noise = np.std( np.append(data[:5,:,:], data[-5:,:,:], axis=0), axis=0) #rms intensity from first and last 5 channels

    if mask is None:
        mask = np.zeros((nx,ny)).astype(bool)

    mask = mask | (np.nanmax(data, axis=0) <= sigma_thres*noise)

    if model is not None and mask_radially:
        R, phi, z = [model.projected_coords[key] for key in ['R', 'phi', 'z']]
        mask = mask | np.isnan(R['upper'])
        
    #******************************
    #KERNELS AND RELEVANT FUNCTIONS
    if method=='doublegaussian':
        n_pars = 6
        fit_func1d = _gauss
        pfunc_two = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j],
                                 I_max_lower[i,j], vel_peak_lower[i,j], lw_lower[i,j]]
        pfunc_one = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j]]
        if kind=='sum':
            fit_func = _doublegauss_sum
        elif kind=='mask':
            fit_func = _doublegauss_mask
        else:
            raise InputError(kind, "kind must be 'mask' or 'sum'")
        idlow = np.array([3,4,5])
        bound0 = (0, -np.inf, -np.inf, 0, -np.inf, -np.inf)
        bound1 = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
        bounds = [[0, vchannels[0], 0, 0, vchannels[0], 0], [np.inf, vchannels[-1], np.abs(vchannels[-1]-vchannels[0]), np.inf, vchannels[-1], np.abs(vchannels[-1]-vchannels[0])]]
        
    elif method=='doublebell':
        n_pars = 8
        fit_func1d = _bell
        pfunc_two = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j], ls_upper[i,j],
                                 I_max_lower[i,j], vel_peak_lower[i,j], lw_lower[i,j], ls_lower[i,j]]
        pfunc_one = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j], ls_upper[i,j]]
        
        if kind=='sum':
            fit_func = _doublebell_sum
        elif kind=='mask':
            fit_func = _doublebell_mask
        else:
            raise InputError(kind, "kind must be 'mask' or 'sum'")
        idlow = np.array([4,5,6,7])
        bound0 = (0, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf, 0)
        bound1 = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
        bounds = [[0, vchannels[0], 0, 0, 0, vchannels[0], 0, 0], [np.inf, vchannels[-1], np.abs(vchannels[-1]-vchannels[0]), 10, np.inf, vchannels[-1], np.abs(vchannels[-1]-vchannels[0]), 10]]

    else:
        _not_available(method)

    def fill_props_ij(i, j, coeff, deltas):
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

        if is_dbell:
            lineslope_up[i,j] = coeff[3]
            lineslope_low[i,j] = coeff[idlow[3]]
            dlineslope_up[i,j] = deltas[3]
            dlineslope_low[i,j] = deltas[idlow[3]]
            
    def fill_props(coeff, deltas):
        peak_up[:] = coeff[:,:,0]
        centroid_up[:] = coeff[:,:,1]
        linewidth_up[:] = coeff[:,:,2]
        
        peak_low[:] = coeff[:,:,idlow[0]]
        centroid_low[:] = coeff[:,:,idlow[1]]
        linewidth_low[:] = coeff[:,:,idlow[2]]

        dpeak_up[:] = deltas[:,:,0]
        dcentroid_up[:] = deltas[:,:,1]
        dlinewidth_up[:] = deltas[:,:,2]

        dpeak_low[:] = deltas[:,:,idlow[0]]
        dcentroid_low[:] = deltas[:,:,idlow[1]]
        dlinewidth_low[:] = deltas[:,:,idlow[2]]

        if is_dbell:
            lineslope_up[:] = coeff[:,:,3]
            lineslope_low[:] = coeff[:,:,idlow[3]]
            dlineslope_up[:] = deltas[:,:,3]
            dlineslope_low[:] = deltas[:,:,idlow[3]]
            
    print ('Fitting two-component function along velocity axis of the input cube...')

    #********
    #MAKE FIT    
    #********
    def process_xi(i, datai, maski, n_pars=8):

        empty = np.zeros(n_pars)
        n_one, n_two, n_bad, n_hot, n_mask = 0, 0, 0, 0, 0
        coeff, deltas, n_fit = [], [], []

        for j in range(ny): #must change by actual list of j's
            tmp_data = datai[:,j]
            peak = 2*np.nanmax(tmp_data)
            bounds[1][0] = peak
            bounds[1][idlow[0]] = peak             
            
            if maski[j]:
                n_mask += 1
                n_fit.append(-10)
                coeff.append(empty)
                deltas.append(empty)
                continue
            
            try:
                p0 = pfunc_two(i,j) #change by neighbour_guess call

                #if n_pars==8:
                #    bounds[1][3] = p0[3]*3
                #    bounds[1][idlow[3]] = p0[idlow[3]]*3
                    
                coeffj, var_matrix = curve_fit(fit_func,
                                               vchannels, tmp_data,
                                               p0=p0,
                                               #bounds=bounds, method='trf')
                                               ftol=1e-10, xtol=1e-10, gtol=1e-10, method='lm')

                deltasj = np.sqrt(np.abs(np.diag(var_matrix)))
                n_two += 1
                n_fit.append(2)
                
            except RuntimeError:
                try: 
                    coeffj, var_matrix = curve_fit(fit_func1d,
                                                   vchannels, tmp_data,
                                                   p0=pfunc_one(i,j)
                    )
                    coeffj = np.append(coeffj, coeffj)
                    deltasj = np.sqrt(np.abs(np.diag(var_matrix)))
                    deltasj = np.append(deltasj, deltasj)
                    n_one += 1
                    n_fit.append(1)
                    
                except RuntimeError:
                    n_bad += 1
                    n_fit.append(0)
                    coeff.append(empty)
                    deltas.append(empty)
                    continue

            coeff.append(coeffj) 
            deltas.append(deltasj)
            
            #fill_props(i,j,coeff,deltas)
        return coeff, deltas, n_fit, [n_one, n_two, n_bad, n_hot, n_mask]

    
    def parallel_processing(data, mask, n_pars=8):

        with Pool(ncpus=None) as pool:
            results = list(tqdm(pool.imap(
                lambda i: process_xi(i, data[:,i,:], mask[i,:], n_pars=n_pars),
                range(nx)
            ), total=nx, desc="Processing two-component fit"))

        coeff, deltas, n_fit, n_diag = [], [], [], []
        #phii, residi = [], []
        for coeff_local, deltas_local, n_fit_local, n_diag_local in results:
            coeff.append(coeff_local)
            deltas.append(deltas_local)
            n_fit.append(n_fit_local)
            n_diag.append(n_diag_local)

        coeff = np.asarray(coeff)
        deltas = np.asarray(deltas)
        n_fit = np.asarray(n_fit)
        n_diag = np.asarray(n_diag)

        return coeff, deltas, n_fit, n_diag
    
    coeff, deltas, n_fit, n_diag = parallel_processing(data, mask, n_pars=n_pars)
    n_one, n_two, n_bad, n_hot, n_mask = np.sum(n_diag, axis=0)

    fill_props(coeff, deltas) #If ny is not uniform, as is the case for hot pixel fit, call fill_props_ij instead, so that one can loop over ij
    
    #*****************************
    #KEEP TRACK OF 'HOT' PIXELS
    #*****************************
    #Hot pixels will be tagged as -1

    def clean_nfit():
        mm = n_fit == -10 #noise
        ii = ((peak_up < 0.0) | (peak_low < 0.0)) & (~mm) #negative intensities
        peak_thres = 2*np.nanmax(cube_max)
        jj = ((peak_up > peak_thres) | (peak_low > peak_thres)) & (~mm) #too large intensities
        cc = (centroid_up == centroid_low) & (n_fit != 1) & (~mm) #up==low
        ww = ((np.abs(linewidth_up) <= 0.5*np.abs(dv)) | (np.abs(linewidth_low) <= 0.5*np.abs(dv))) & (~mm) #component too narrow (<=half a chan width)
        dd = ((np.abs(linewidth_up) > 5.0) | (np.abs(linewidth_low) > 5.0)) & (~mm) #Unrealistically broad component

        bb = False #dcentroid_up > 1*np.abs(dv)

        w3 = False #((np.abs(linewidth_up) <= 1*np.abs(dv)) | (np.abs(linewidth_low) <= 1*np.abs(dv))) & (~mm) #narrow component
        w4 = False #(np.abs(linewidth_low) >= 2*np.abs(linewidth_up)) & (~mm) #lower line width much larger than upper

        w5 = False #dcentroid_up > 1
        
        n_fit[ii+jj+cc+ww+dd+w5] = -1 #bad pixel
        n_hot = np.sum(ii+jj+cc+ww+dd + w3+w4+w5)

        n_fit[w3] = 3 #1chan narrow component
        n_fit[w4] = 4 #1chan narrow component
        n_fit[w5] = 5 #Bad delta
        
        return n_hot
        
    #**********
    #PACK PROPS
    #**********
    upper = [peak_up, centroid_up, linewidth_up]
    dupper = [dpeak_up, dcentroid_up, dlinewidth_up]
    lower = [peak_low, centroid_low, linewidth_low]
    dlower = [dpeak_low, dcentroid_low, dlinewidth_low]

    if is_dbell:
        upper += [lineslope_up]
        dupper += [dlineslope_up]
        lower += [lineslope_low]
        dlower += [dlineslope_low]

    n_hot = clean_nfit()
    
    print ('\nTwo-component fit did not converge for %.2f%s of the pixels'%(100.0*(n_bad)/(nx*ny),'%'))
    print ('A single component was fit for %.2f%s of the pixels'%(100.0*(n_one)/(nx*ny),'%'))
    print ('Masked pixels below intensity threshold: %.2f%s'%(100.0*(n_mask)/(nx*ny),'%'))
    print ('Hot pixels: %.2f%s'%(100.0*(n_hot)/(nx*ny),'%'))        

    #******************
    #HANDLE HOT PIXELS
    #******************    
    if niter>0:
        print ('\nRe-doing fit for  %d hot pixels and %d single-component pixels'%(n_hot, n_one))

        def neighbour_guess(i, j, n_fit, neighs=3, av_func=np.nanmean):

            if i<neighs or j<neighs:
                return None
            
            neigh_arr = np.arange(-neighs, neighs+1)

            ileft, iright = i-neighs, i+neighs+1
            jleft, jright = j-neighs, j+neighs+1
            window = n_fit[ileft:iright, jleft:jright]

            masked = window==-10
            n_mask = np.sum(masked) #n masked pixels

            hot = window==-1
            n_hot = np.sum(hot) #n hot pixels
            
            one = window==1
            n_one = np.sum(one) #single-component pixels

            
            three = window==3
            n_three = np.sum(three) #narrow components

            four = window==4
            n_four = np.sum(four) #broad lower linewidth
            
            
            n_bad = n_mask+n_hot+n_one + n_three+n_four
            tot = (2*neighs+1)**2

            if tot-n_bad < n_bad:
                return None

            ic, jc = (~masked & ~hot & ~one  &  ~three & ~four).nonzero() #get clean pixels where double fit worked

            up_guess = [av_func(up_prop[ileft:iright, jleft:jright][ic, jc]) for up_prop in upper]
            low_guess = [av_func(low_prop[ileft:iright, jleft:jright][ic, jc]) for low_prop in lower]
            return up_guess+low_guess
                         
        for ni in range(niter):
            print ('Iteration #%d...'%(ni+1))

            #Select hot pixels (-1 flag) and single-component pixels (+1 flag)
            m, n = np.where((n_fit==-1) | (n_fit==1) | (n_fit>2))
            #m, n = np.where(n_fit==-1)
                
            for k in range(len(m)):
                i, j = m[k], n[k]                                
                uplow = neighbour_guess(i, j, n_fit, neighs=neighs, av_func=av_func)
                
                if uplow is None:
                    continue

                tmp_data = data[:,i,j]

                try:                    
                    coeff, var_matrix = curve_fit(fit_func,
                                                  vchannels, tmp_data,
                                                  p0 = uplow,
                                                  #bounds=bounds, method='trf')                                                  
                                                  ftol=1e-10, xtol=1e-10, gtol=1e-10, method='lm')
                    deltas = np.sqrt(np.abs(np.diag(var_matrix)))
                    n_two += 1
                    n_fit[i,j] = 2
                    
                    r"""
                    if (deltas[:3]==np.inf).any(): # or deltas[1] > 0.5*np.abs(dv):
                        coeff, var_matrix = curve_fit(fit_func1d,
                                                      vchannels, tmp_data,
                                                      p0=pfunc_one(i,j)
                        )
                        coeff = np.append(coeff, coeff)
                        deltas = np.sqrt(np.abs(np.diag(var_matrix)))
                        deltas = np.append(deltas, deltas)
                        n_one += 1
                        n_fit[i,j] = 1
                    else:                
                        n_two += 1
                        n_fit[i,j] = 2
                    #"""
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

                fill_props_ij(i,j,coeff,deltas)
                
            n_hot = clean_nfit()
            print ('Resulting hot pixels:', n_hot)

        #***************************************************
        #FIT ONE COMPONENT FUNCTION TO REMAINING HOT PIXELS
        #***************************************************
        m, n = np.where(n_fit==-1)

        for k in range(len(m)):
            i, j = m[k], n[k]                                
            tmp_data = data[:,i,j]

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

            fill_props_ij(i,j,coeff,deltas)

    n_hot = clean_nfit()
    print ('Resulting hot pixels after fitting 1D component:', n_hot)
    
    return upper, dupper, lower, dlower, n_fit
    
    """
    #****************
    #TREAT HOT PIXELS
    #****************    
    def neighbour_guess(i, j, n_fit, neighs=3, av_func=np.nanmean):

        if i<neighs or j<neighs:
            return None
        
        neigh_arr = np.arange(-neighs, neighs+1)

        ileft, iright = i-neighs, i+neighs+1
        jleft, jright = j-neighs, j+neighs+1
        window = n_fit[ileft:iright, jleft:jright]
        
        masked = window==-10
        n_mask = np.sum(masked) #n masked pixels

        hot = window==-1
        n_hot = np.sum(hot) #n hot pixels
            
        one = window==1
        n_one = np.sum(one) #single-component pixels
            
        three = window==3
        n_three = np.sum(three) #narrow components

        four = window==4
        n_four = np.sum(four) #broad lower linewidth
        
        n_bad = n_mask+n_hot+n_one + n_three+n_four
        tot = (2*neighs+1)**2

        if tot-n_bad < n_bad:
            return None

        ic, jc = (~masked & ~hot & ~one  &  ~three & ~four).nonzero() #get clean pixels where double fit worked

        up_guess = [av_func(up_prop[ileft:iright, jleft:jright][ic, jc]) for up_prop in upper]
        low_guess = [av_func(low_prop[ileft:iright, jleft:jright][ic, jc]) for low_prop in lower]
        return up_guess+low_guess

    #Process rows in parallel    
    def process_ij(i, datai, bad_j_indices, n_fit, n_pars=8):
        empty = np.zeros(n_pars)
        coeff, deltas, n_fit_tmp = [], [], []
    
        for j in bad_j_indices:

            uplow = neighbour_guess(i, j, n_fit, neighs=neighs, av_func=av_func)
            
            if uplow is None:
                n_fit_tmp.append(0)
                coeff.append(empty)
                deltas.append(empty)
                continue

            tmp_data = datai[:,j]
            
            try:                    
                coeffj, var_matrix = curve_fit(fit_func,
                                               vchannels, tmp_data,
                                               p0 = uplow,
                                               ftol=1e-10, xtol=1e-10, gtol=1e-10, method='lm')
                deltasj = np.sqrt(np.abs(np.diag(var_matrix)))
                n_fit_tmp.append(2)
                    
            except RuntimeError:
                try: 
                    coeffj, var_matrix = curve_fit(fit_func1d,
                                                  vchannels, tmp_data,
                                                  p0=pfunc_one(i,j)
                    )
                    coeffj = np.append(coeffj, coeffj)
                    deltasj = np.sqrt(np.abs(np.diag(var_matrix)))
                    deltasj = np.append(deltasj, deltasj)
                    n_fit_tmp.append(1)
                    
                except RuntimeError:
                    n_fit_tmp.append(0)
                    coeff.append(empty)
                    deltas.append(empty)
                    continue            

            coeff.append(coeffj) 
            deltas.append(deltasj)
                
        return coeff, deltas, n_fit_tmp

    def parallel_processing_ij(data, bad_pixel_indices, n_fit, n_pars=8):
        rows_with_bad_fits = {}
        for i, j in bad_pixel_indices:
            if i not in rows_with_bad_fits:
                rows_with_bad_fits[i] = []
            rows_with_bad_fits[i].append(j)

        rows2fit = list(rows_with_bad_fits.keys())
        
        with Pool(ncpus=None) as pool:
            results = list(tqdm(pool.imap(
                lambda i: process_ij(i, data[:, i, :], rows_with_bad_fits[i], n_fit, n_pars=n_pars),
                rows2fit
            ), total=len(rows2fit), desc="Re-fitting hot pixels"))
            
        coeff, deltas, n_fit_tmp = [], [], []
    
        # Collect results from each row
        for coeff_local, deltas_local, n_fit_local in results:
            coeff.append(coeff_local)
            deltas.append(deltas_local)
            n_fit_tmp.append(n_fit_local)

        return coeff, deltas, n_fit_tmp
    
    if niter>0:
        print ('\nRe-doing fit for  %d hot pixels and %d single-component pixels'%(n_hot, n_one))
        
        for n in range(niter):
            print ('Iteration #%d...'%(n+1))

            #Select hot pixels (-1 flag) and single-component pixels (+1 flag)
            ib,jb = np.where((n_fit==-1) | (n_fit==1) | (n_fit>2))
            bad_pixel_indices = zip(ib,jb)
            
            coeffk, deltask, n_fitk = parallel_processing_ij(data, bad_pixel_indices, n_fit)
            
            #Update coeff, deltas, and n_fit matrices per ij
            k=0
            for i in range(len(coeffk)):
                for j in range(len(coeffk[i])):
                    fill_props_ij(ib[k], jb[k], coeffk[i][j], deltask[i][j])
                    n_fit[ib[k],jb[k]] = n_fitk[i][j]
                    k+=1
                    
            n_hot = clean_nfit()
            print ('Resulting hot pixels:', n_hot)
    
    return upper, dupper, lower, dlower, n_fit
    """

def fit_twocomponent_(cube, model=None, lw_chans=1.0, lower2upper=1.0,
                     method='doublegaussian', kind='mask', sigma_thres=5,
                     sigma_fit=None, mask=None, planck=False,
                     niter=4, neighs=5, av_func=np.nanmedian,
                     mask_radially=True
):

    data = cube.data    
    vchannels = cube.vchannels
    dv = np.mean(vchannels[1:]-vchannels[:-1])
    lw_sign = np.sign(dv)
    
    nchan, nx, ny = np.shape(data)
    n_one, n_two, n_bad, n_hot, n_mask = 0, 0, 0, 0, 0
    n_fit = np.zeros((nx, ny))
    
    peak_up, dpeak_up = np.zeros((2, nx, ny))
    centroid_up, dcentroid_up = np.zeros((2, nx, ny))
    linewidth_up, dlinewidth_up = np.zeros((2, nx, ny))
    lineslope_up, dlineslope_up = np.zeros((2, nx, ny))    
    peak_low, dpeak_low = np.zeros((2, nx, ny))
    centroid_low, dcentroid_low = np.zeros((2, nx, ny))
    linewidth_low, dlinewidth_low = np.zeros((2, nx, ny))
    lineslope_low, dlineslope_low = np.zeros((2, nx, ny))
    
    is_dbell = method=='doublebell'
    
    #MODEL AS INITIAL GUESS?
    if model is None:
        print ('Guessing upper surface properties from data to use them as seeds for both upper (primary) and lower (secondary) surface components ...')        
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
        I_upper = int2d['upper']*cube.beam_area
        I_lower = int2d['lower']*cube.beam_area
        print ('Using upper and lower surface properties from discminer model as initial guesses...')
        
        if np.any(np.array(['K', 'Kelvin', 'K ', 'Kelvin ']) == cube.header['BUNIT']): #If input unit is K the raw model intensity must be converted    
            r"""
            mheader = copy.copy(cube.header)
            mheader["BUNIT"] = "beam-1 Jy"
            I_upper = FITSUtils._convert_to_tb(I_upper*cube.beam_area, mheader, cube.beam, planck=False, writefits=False)
            #"""
            restfreq = cube.header['RESTFRQ']*1e-9
            I_upper = get_tb(1e3*I_upper, restfreq, cube.beam, full=planck) #mJy/beam-->K
            I_lower = get_tb(1e3*I_lower, restfreq, cube.beam, full=planck) 

        #Initial guesses
        ind_max = np.nanargmax(data, axis=0)
        cube_max = np.take_along_axis(data, ind_max[None], axis=0).squeeze()
        I_max_upper = np.where(np.isnan(I_upper), 1.0*cube_max, I_upper)
        I_max_lower = np.where(np.isnan(I_lower), 0.5*cube_max, I_lower)
        vel_peak_upper = np.where(np.isnan(vel2d['upper']), vchannels[ind_max], vel2d['upper'])
        vel_peak_lower = np.where(np.isnan(vel2d['lower']), vchannels[ind_max], vel2d['lower'])
        lw_upper = lw_sign*np.where(np.isnan(linew2d['upper']), lw_sign*1.5*dv, linew2d['upper'])
        lw_lower = lw_sign*np.where(np.isnan(linew2d['lower']), lw_sign*1.5*dv, linew2d['lower'])
        ls_upper = np.where(np.isnan(linew2d['upper']), 1.5, 1*lineb2d['upper']) #0.5*
        ls_lower = np.where(np.isnan(linew2d['lower']), 1.5, 1*lineb2d['lower']) #0.5*

    if sigma_fit is None: sigma_func = lambda i,j: None
    else: sigma_func = lambda i,j: sigma_fit[:,i,j]

    noise = np.std( np.append(data[:5,:,:], data[-5:,:,:], axis=0), axis=0) #rms intensity from first and last 5 channels

    if mask is None:
        mask = np.zeros((nx,ny)).astype(bool)

    mask = mask | (np.nanmax(data, axis=0) <= sigma_thres*noise)

    if model is not None and mask_radially:
        R, phi, z = [model.projected_coords[key] for key in ['R', 'phi', 'z']]
        mask = mask | np.isnan(R['upper'])
        
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
        else:
            raise InputError(kind, "kind must be 'mask' or 'sum'")
        idlow = np.array([3,4,5])
        bound0 = (0, -np.inf, -np.inf, 0, -np.inf, -np.inf)
        bound1 = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)

    elif method=='doublebell':
        fit_func1d = _bell
        pfunc_two = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j], ls_upper[i,j],
                                 I_max_lower[i,j], vel_peak_lower[i,j], lw_lower[i,j], ls_lower[i,j]]
        pfunc_one = lambda i,j: [I_max_upper[i,j], vel_peak_upper[i,j], lw_upper[i,j], ls_upper[i,j]]
        
        if kind=='sum':
            fit_func = _doublebell_sum
        elif kind=='mask':
            fit_func = _doublebell_mask
        else:
            raise InputError(kind, "kind must be 'mask' or 'sum'")
        idlow = np.array([4,5,6,7])
        bound0 = (0, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf, 0)
        bound1 = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)

    else:
        _not_available(method)

    def fill_props(i, j, coeff):
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

        if is_dbell:
            lineslope_up[i,j] = coeff[3]
            lineslope_low[i,j] = coeff[idlow[3]]
            dlineslope_up[i,j] = deltas[3]
            dlineslope_low[i,j] = deltas[idlow[3]]
        
    print ('Fitting two-component function along velocity axis of the input cube...')

    #********
    #MAKE FIT
    for i in range(nx):
        for j in range(ny):
            tmp_data = data[:,i,j]

            if mask[i,j]:
                n_mask += 1
                n_fit[i,j] = -10                                    
                continue
            
            try: 
                coeff, var_matrix = curve_fit(fit_func,
                                              vchannels, tmp_data,
                                              p0=pfunc_two(i,j),
                                              ftol=1e-10, xtol=1e-10, gtol=1e-10, method='lm')
                                              #bounds = [bound0, bound1])
                deltas = np.sqrt(np.abs(np.diag(var_matrix)))
                n_two += 1
                n_fit[i,j] = 2

                r"""
                if (deltas[:3]==np.inf).any():# or deltas[1] > 0.5*np.abs(dv):
                    coeff, var_matrix = curve_fit(fit_func1d,
                                                  vchannels, tmp_data,
                                                  p0=pfunc_one(i,j)
                    )
                    coeff = np.append(coeff, coeff)
                    deltas = np.sqrt(np.abs(np.diag(var_matrix)))
                    deltas = np.append(deltas, deltas)
                    n_one += 1
                    n_fit[i,j] = 1
                else:                
                    n_two += 1
                    n_fit[i,j] = 2
                #"""
                
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

            r"""
            if coeff[0] < coeff[idlow[0]]:
                coeff = np.append(coeff[idlow[0]:idlow[-1]+1], coeff[0:idlow[0]])
                deltas = np.append(deltas[idlow[0]:idlow[-1]+1], deltas[0:idlow[0]])
            #"""
            
            r"""
            #Can induce biases when upper and lower velocities are too close
            if np.abs(coeff[idlow[1]]-vel_peak_upper[i,j]) < np.abs(coeff[1]-vel_peak_upper[i,j]):
                coeff = np.append(coeff[idlow[0]:idlow[-1]+1], coeff[0:idlow[0]])
                deltas = np.append(deltas[idlow[0]:idlow[-1]+1], deltas[0:idlow[0]])
            #"""
            r"""
            #Sometimes the fit can output upper profiles near to zero intensity and in turn the lower profile is attributed to the bulk of the emission.
            #The following seems to improve things due to this in a few pixels. However, the conditional above accounts for this for the most part already.
            if coeff[0]<=1.0 and coeff[3]>1.0:
                coeff = np.append(coeff[idlow[0]:idlow[-1]+1], coeff[0:idlow[0]])
                deltas = np.append(deltas[idlow[0]:idlow[-1]+1], deltas[0:idlow[0]])
            #""" 

            fill_props(i,j,coeff)
                        
        _progress_bar(int(100*i/nx))
    _progress_bar(100)

    #*****************************
    #KEEP TRACK OF 'HOT' PIXELS
    #*****************************
    #Hot pixels will be tagged as -1

    def clean_nfit():
        mm = n_fit == -10 #noise
        ii = ((peak_up < 0.0) | (peak_low < 0.0)) & (~mm) #negative intensities
        peak_thres = 2*np.nanmax(cube_max)
        jj = ((peak_up > peak_thres) | (peak_low > peak_thres)) & (~mm) #too large intensities
        cc = (centroid_up == centroid_low) & (n_fit != 1) & (~mm) #up==low
        ww = ((np.abs(linewidth_up) <= 0.5*np.abs(dv)) | (np.abs(linewidth_low) <= 0.5*np.abs(dv))) & (~mm) #narrow component
        dd = ((np.abs(linewidth_up) > 5.0) | (np.abs(linewidth_low) > 5.0)) & (~mm) #Unrealistically broad component

        bb = False #dcentroid_up > 1*np.abs(dv)

        w3 = False #((np.abs(linewidth_up) <= 1*np.abs(dv)) | (np.abs(linewidth_low) <= 1*np.abs(dv))) & (~mm) #narrow component
        w4 = False #(np.abs(linewidth_low) >= 2*np.abs(linewidth_up)) & (~mm) #lower line width much larger than upper
        
        n_fit[ii+jj+cc+ww+dd] = -1 #bad pixel
        n_hot = np.sum(ii+jj+cc+ww+dd + w3+w4)

        n_fit[w3] = 3 #1chan narrow component
        n_fit[w4] = 4 #1chan narrow component
        
        return n_hot
        
    #**********
    #PACK PROPS
    #**********
    upper = [peak_up, centroid_up, linewidth_up]
    dupper = [dpeak_up, dcentroid_up, dlinewidth_up]
    lower = [peak_low, centroid_low, linewidth_low]
    dlower = [dpeak_low, dcentroid_low, dlinewidth_low]

    if is_dbell:
        upper += [lineslope_up]
        dupper += [dlineslope_up]
        lower += [lineslope_low]
        dlower += [dlineslope_low]

    n_hot = clean_nfit()
    
    print ('\nTwo-component fit did not converge for %.2f%s of the pixels'%(100.0*(n_bad)/(nx*ny),'%'))
    print ('A single component was fit for %.2f%s of the pixels'%(100.0*(n_one)/(nx*ny),'%'))
    print ('Masked pixels below intensity threshold: %.2f%s'%(100.0*(n_mask)/(nx*ny),'%'))
    print ('Hot pixels: %.2f%s'%(100.0*(n_hot)/(nx*ny),'%'))        

    if niter>0:
        print ('\nRe-doing fit for  %d hot pixels and %d single-component pixels'%(n_hot, n_one))

        def neighbour_guess(i, j, n_fit, neighs=3, av_func=np.nanmean):

            if i<neighs or j<neighs:
                return None
            
            neigh_arr = np.arange(-neighs, neighs+1)

            ileft, iright = i-neighs, i+neighs+1
            jleft, jright = j-neighs, j+neighs+1
            window = n_fit[ileft:iright, jleft:jright]

            masked = window==-10
            n_mask = np.sum(masked) #n masked pixels

            hot = window==-1
            n_hot = np.sum(hot) #n hot pixels
            
            one = window==1
            n_one = np.sum(one) #single-component pixels

            
            three = window==3
            n_three = np.sum(three) #narrow components

            four = window==4
            n_four = np.sum(four) #broad lower linewidth
            
            
            n_bad = n_mask+n_hot+n_one + n_three+n_four
            tot = (2*neighs+1)**2

            if tot-n_bad < n_bad:
                return None

            ic, jc = (~masked & ~hot & ~one  &  ~three & ~four).nonzero() #get clean pixels where double fit worked

            up_guess = [av_func(up_prop[ileft:iright, jleft:jright][ic, jc]) for up_prop in upper]
            low_guess = [av_func(low_prop[ileft:iright, jleft:jright][ic, jc]) for low_prop in lower]
            return up_guess+low_guess
                         
        for ni in range(niter):
            print ('Iteration #%d...'%(ni+1))

            #Select hot pixels (-1 flag) and single-component pixels (+1 flag)
            m, n = np.where((n_fit==-1) | (n_fit==1) | (n_fit>2))
            #m, n = np.where(n_fit==-1)
                
            for k in range(len(m)):
                i, j = m[k], n[k]                                
                uplow = neighbour_guess(i, j, n_fit, neighs=neighs, av_func=av_func)
                
                if uplow is None:
                    continue

                tmp_data = data[:,i,j]

                try:                    
                    coeff, var_matrix = curve_fit(fit_func,
                                                  vchannels, tmp_data,
                                                  p0 = uplow,
                                                  ftol=1e-10, xtol=1e-10, gtol=1e-10, method='lm')
                    deltas = np.sqrt(np.abs(np.diag(var_matrix)))
                    n_two += 1
                    n_fit[i,j] = 2
                    
                    r"""
                    if (deltas[:3]==np.inf).any(): # or deltas[1] > 0.5*np.abs(dv):
                        coeff, var_matrix = curve_fit(fit_func1d,
                                                      vchannels, tmp_data,
                                                      p0=pfunc_one(i,j)
                        )
                        coeff = np.append(coeff, coeff)
                        deltas = np.sqrt(np.abs(np.diag(var_matrix)))
                        deltas = np.append(deltas, deltas)
                        n_one += 1
                        n_fit[i,j] = 1
                    else:                
                        n_two += 1
                        n_fit[i,j] = 2
                    #"""
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

                fill_props(i,j,coeff)
                
            n_hot = clean_nfit()
            print ('Resulting hot pixels:', n_hot)
    
    return upper, dupper, lower, dlower, n_fit


def fit_gaussian(*args, **kwargs): #Backcompat
    return fit_onecomponent(*args, **kwargs)

def fit_onecomponent(
        cube, method='gaussian', lw_chans=1.0, peak_kernel=True, sigma_fit=None, sigma_thres=4, fit_continuum=False
):
    """
    Fit 'gaussian' or 'bell' profiles along the velocity axis of the input cube.
    
    Parameters
    ----------
    lw_chans : int, optional
        Number of channel widths to take as initial guess for line width

    sigma_fit : array_like, optional 
        2-D array of weights computed for each pixel. Shape must be equal to that of the spatial axes of the input data. 

    peak_kernel : bool, optional
        If True (default) the returned amplitude is the peak of the kernel fitted to the line. Otherwise, the actual peak of the line profile is returned.

    Returns
    -------
    A,B,C,dA,dB,dC : 2-D arrays
        Amplitude, centre, standard deviation, and their corresponding errors, obtained from Gaussians fitted to the input datacube along its velocity axis, on each spatial pixel.
    
    """    
    data = cube.data
    vchannels = cube.vchannels

    nchan, nx, ny = np.shape(data)
    n_fit = np.zeros((nx, ny))                        
    n_bad, n_mask = 0, 0
    
    peak, dpeak = np.zeros((nx, ny)), np.zeros((nx, ny))
    centroid, dcent = np.zeros((nx, ny)), np.zeros((nx, ny))
    linewidth, dlinew = np.zeros((nx, ny)), np.zeros((nx, ny))
    lineslope, dlineslope = np.zeros((nx, ny)), np.zeros((nx, ny))    
    if fit_continuum:
        continuum, dcontinuum = np.zeros((nx, ny)), np.zeros((nx, ny))
        
    is_bell = method=='bell'
    
    ind_max = np.nanargmax(data, axis=0)
    I_max = np.nanmax(data, axis=0)
    vel_peak = vchannels[ind_max]
    dv = lw_chans*np.mean(vchannels[1:]-vchannels[:-1])
    Ls = 2.0 #p0 Line slope
    
    if sigma_fit is None:
        sigma_func = lambda i,j: None
    else:
        sigma_func = lambda i,j: sigma_fit[:,i,j]

    #******************************
    #KERNELS AND RELEVANT FUNCTIONS
    if method in ['gaussian', 'gauss']:
        fit_func1d = _gauss
        if fit_continuum:
            pfunc_one = lambda i,j: [I_max[i,j], vel_peak[i,j], dv, 0.0]
        else:
            pfunc_one = lambda i,j: [I_max[i,j], vel_peak[i,j], dv]

    elif method=='bell':
        fit_func1d = _bell
        if fit_continuum:
            pfunc_one = lambda i,j: [I_max[i,j], vel_peak[i,j], dv, Ls, 0.0]
        else:
            pfunc_one = lambda i,j: [I_max[i,j], vel_peak[i,j], dv, Ls]
    else:
        _not_available(method)

    noise = np.std( np.append(data[:5,:,:], data[-5:,:,:], axis=0), axis=0) #rms intensity from first and last 5 channels
    mask = np.nanmax(data, axis=0) <= sigma_thres*noise
        
    print ('Fitting one-component function along velocity axis of the input cube...')

    for i in range(nx):
        for j in range(ny):
            tmp_data = data[:,i,j]
            
            if mask[i,j]:
                n_mask += 1
                n_fit[i,j] = -10                                    
                continue
            
            try:
                coeff, var_matrix = curve_fit(fit_func1d, vchannels, tmp_data, p0=pfunc_one(i,j), sigma=sigma_func(i,j))
                n_fit[i,j] = 1
                    
            except RuntimeError: 
                n_bad+=1
                n_fit[i,j] = 0                
                continue

            deltas = np.sqrt(np.diag(var_matrix))
            
            if peak_kernel:
                peak[i,j] = coeff[0]
            else:
                peak[i,j] = I_max[i,j]
                
            centroid[i,j] = coeff[1]
            linewidth[i,j] = coeff[2]
            dpeak[i,j], dcent[i,j], dlinew[i,j] = deltas[:3]

            if is_bell:
                lineslope[i,j] = coeff[3]
                dlineslope[i,j] = deltas[3]

            if fit_continuum:
                continuum[i,j] = coeff[-1]
                dcontinuum[i,j] = deltas[-1]
                
        _progress_bar(int(100*i/nx))

    _progress_bar(100)

    #***************
    #PACK AND RETURN
    #***************    
    upper = [peak, centroid, np.abs(linewidth)]
    dupper = [dpeak, dcent, dlinew]

    if is_bell:
        upper += [lineslope]
        dupper += [dlineslope]

    if fit_continuum:
        upper += [continuum]
        dupper += [dcontinuum]
        
    print ('\nOne-component fit did not converge for %.2f%s of the pixels'%(100.0*n_bad/(nx*ny),'%'))
    print ('Masked pixels below intensity threshold: %.2f%s'%(100.0*(n_mask)/(nx*ny),'%'))
    
    return upper, dupper, n_fit
