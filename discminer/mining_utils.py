"""
Useful repeating code
"""
import json
import decimal
import argparse
import warnings
import numbers
import runpy
import copy
import sys
import os

from discminer.plottools import get_discminer_cmap, make_1d_legend
from discminer.tools.utils import FrontendUtils
from discminer.grid import GridTools
from discminer.core import Data
from discminer.rail import Rail
from discminer.disc2d import Model
import discminer.cart as cart

from astropy.io import fits
import astropy.units as u

import numpy as np
import matplotlib


SMALL_SIZE = 10
MEDIUM_SIZE = 15

disctex = {
    'dmtau': r'$\rm DM\,Tau$',
    'aatau': r'$\rm AA\,Tau$',
    'lkca15': r'$\rm LkCa15$',
    'hd34282': r'$\rm HD\,34282$',
    'mwc758': r'$\rm MWC\,758$', 
    'cqtau': r'$\rm CQ\,Tau$',
    'sycha': r'$\rm SY\,Cha$',
    'pds66': r'$\rm PDS\,66$',
    'hd135344': r'$\rm HD\,135344B$',
    'j1604': r'$\rm J1604$',
    'j1615': r'$\rm J1615$',
    'v4046': r'$\rm V4046\,Sgr$',
    'j1842': r'$\rm J1842$',
    'j1852': r'$\rm J1852$',
    'hd143006': r'$\rm HD\,143006$',
    'hd163296': r'$\rm HD\,163296$'
}

moltex = {
    '12co': r'$^{\rm 12}$CO',
    '13co': r'$^{\rm 13}$CO',
    'cs': r'CS',
    'hcn': r'HCN',
    'c2h': r'C$_{\rm 2}$H'
}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def init_data_and_model(parfile='parfile.json', Rmin=0, Rmax=1.1, twodim=False, init_model=True, write_extent=True, verbose=True):
    #Rmin: If dimensionless, fraction of beam_size
    #Rmax: If dimensionless, fraction of Rout    
    with open(parfile) as jf:
        pars = json.load(jf)

    meta = pars['metadata']
    best = pars['best_fit']
    custom = pars['custom']

    file_data = meta['file_data']
    Rout = best['intensity']['Rout']
    dpc = meta['dpc']*u.pc

    datacube = Data(file_data, dpc, twodim=twodim, disc=meta['disc'], mol=meta['mol']) # Read data and convert to Cube object

    func_defaults = {
        'velocity_func': cart.keplerian_vertical,
        'z_upper_func': cart.z_upper_exp_tapered,
        'z_lower_func': cart.z_lower_exp_tapered,
        'intensity_func': cart.intensity_powerlaw_rout,
        'linewidth_func': cart.linewidth_powerlaw,
        'lineslope_func': cart.lineslope_powerlaw,
        'line_profile': cart.line_profile_bell,
        'line_uplow': cart.line_uplow_mask
    }
    
    def _check_customtags(model, func):

        if func == 'z_upper_func':            

            if 'surf2pwl' in meta['kind']:
                model.z_upper_func = cart.z_upper_powerlaw
            elif 'surfirregular' in meta['kind']:
                model.z_upper_func = cart.z_upper_irregular                
            else:
                model.z_upper_func = func_defaults[func]
            
        elif func == 'z_lower_func':            

            if 'surf2pwl' in meta['kind']:
                model.z_lower_func = cart.z_lower_powerlaw
            else:
                model.z_lower_func = func_defaults[func]

        elif func == 'intensity_func':
            
            if 'I2pwl' in meta['kind']:
                model.intensity_func = cart.intensity_powerlaw_rbreak
            elif 'I2pwlnosurf' in meta['kind']:
                model.intensity_func = cart.intensity_powerlaw_rbreak_nosurf    
            else:
                model.intensity_func = func_defaults[func]

        elif func == 'line_uplow':
            
            if 'sum' in meta['kind']:
                model.line_uplow = cart.line_uplow_sum
            else:
                model.line_uplow = func_defaults[func]
                
        else:
            setattr(model, func, func_defaults[func])
            
    def set_model_funcs(model):
        func_list = list(func_defaults.keys())        
        func_custom = dict(f for f in zip(func_list, [False]*len(func_list)))
            
        try: #Check if local customcart module exists
            cwd = os.getcwd()
            sys.path.append(cwd)
            import customcart
            from inspect import getmembers, isfunction
            funcs = getmembers(customcart, isfunction)

            for f in funcs:
                if f[0] in func_list:
                    setattr(model, f[0], f[1])
                    func_custom[f[0]] = True
                    
        except ImportError:
            print ('Using default and/or tag-based parametric forms for model...')

        for func in func_list:
            if not func_custom[func]:
                _check_customtags(model, func)

    if init_model:

        if isinstance(Rmax, numbers.Real):
            Rmax = Rmax*Rout*u.au
            
        model = Model(datacube, Rmax=Rmax, Rmin=Rmin, write_extent=write_extent, prototype=True, verbose=verbose)

        set_model_funcs(model)

        #****************
        #PROTOTYPE PARAMS
        #****************
        model.params = copy.copy(best)
        model.params['intensity']['I0'] /= meta['downsamp_factor']

        return datacube, model

    else:
        return datacube

    
def read_json(decimals=True): #Read json file again but keep track of (if) decimals
    if decimals:
        parse_func = lambda x: decimal.Decimal(str(x))
    else:
        parse_func = float

    with open('parfile.json') as json_file: 
        pars = json.load(json_file, parse_float=parse_func)
    return pars
            
def get_2d_plot_decorators(moment, unit_simple=False, fmt_vertical=False):    
    I_res2abs = 3.0
    v_res2abs = 20.0
    L_res2abs = 5.0
    Ls_res2abs = 3.0

    pars = read_json()
    vsys = float(pars['best_fit']['velocity']['vsys'])
    custom = pars['custom']
    meta = pars['metadata']

    try:
        ctitle = disctex[meta['disc']]+' '+moltex[meta['mol']]
        
    except KeyError:
        ctitle = meta['disc']+' '+meta['mol']

    if moment=='delta_velocity':
        clim = custom['vlim']
        fclim = float(clim)*1000
        unit = '[m s$^{-1}$]'
        if unit_simple: unit = 'm/s'        
        clabel = r'$\Delta$ Velocity %s'%unit
        cmap_mom = matplotlib.pyplot.get_cmap('magma')
        cmap_res = cmap_mom 
        levels_im = np.linspace(0.2*fclim, fclim, 64)
        levels_cc = np.linspace(0.2*fclim, fclim, 4)

    if moment=='delta_linewidth':
        clim = custom['Llim']
        fclim = float(clim)*1000
        unit = '[m s$^{-1}$]'
        if unit_simple: unit = 'm/s'        
        clabel = r'$\Delta$ Line width %s'%unit
        cmap_mom = 'magma' #get_discminer_cmap('linewidth')
        cmap_res = cmap_mom #get_discminer_cmap('linewidth', kind='residuals')
        levels_im = np.linspace(0.0, fclim, 64)
        levels_cc = np.linspace(0.2*fclim, fclim, 4)

    if moment=='delta_peakintensity':
        clim = custom['Ilim']
        fclim = float(clim)
        unit = '[K]'
        if unit_simple: unit = 'K'        
        clabel = r'$\Delta$ Peak Intensity %s'%unit
        cmap_mom = 'magma' #get_discminer_cmap('intensity_2')
        cmap_res = cmap_mom #get_discminer_cmap('intensity_2', kind='residuals')
        levels_im = np.linspace(0.0, fclim, 64)
        levels_cc = np.linspace(0.2*fclim, fclim, 4)
        
    if moment=='reducedchi2':
        clim = custom['Ilim']
        fclim = float(clim)
        unit = '[R-Chi2]'
        if unit_simple: unit = 'R-Chi2'        
        clabel = r'Reduced $X^2$'
        cmap_mom = 'magma'
        cmap_res = cmap_mom 
        levels_im = np.linspace(0.0, fclim, 64)
        levels_cc = np.linspace(0.2*fclim, fclim, 4)
        
    if moment in ['velocity', 'v0phi', 'v0r', 'v0z', 'vr_leftover']:
        clim = custom['vlim']
        fclim = float(clim)        
        unit = '[km s$^{-1}$]'
        if unit_simple: unit = 'km/s'
        clabel = r'Centroid Velocity %s'%unit
        cmap_mom = cmap_res = get_discminer_cmap('velocity')
        levels_im = np.linspace(-v_res2abs*fclim, v_res2abs*fclim, 64)+vsys
        levels_cc = np.linspace(-2, 2, 9)+vsys
    
    if moment=='linewidth':
        clim = custom['Llim']
        fclim = float(clim)
        unit = '[km s$^{-1}$]'
        if unit_simple: unit = 'km/s'        
        clabel = r'Line width %s'%unit
        cmap_mom = get_discminer_cmap('linewidth')
        cmap_res = get_discminer_cmap('linewidth', kind='residuals')
        levels_im = np.linspace(0.0, L_res2abs*fclim, 64)
        levels_cc = np.linspace(0.2, 0.8, 4)

    if moment=='lineslope':
        clim = custom['Lslim']
        fclim = float(clim)        
        unit = ''
        clabel = r'Line slope'
        cmap_mom = get_discminer_cmap('linewidth')
        cmap_res = get_discminer_cmap('linewidth', kind='residuals')
        levels_im = np.linspace(0.0, Ls_res2abs*fclim, 48)
        levels_cc = np.linspace(1.0, Ls_res2abs*fclim, 5)
    
    if moment in ['peakintensity', 'peakint']:
        clim = custom['Ilim']
        fclim = float(clim)        
        unit = '[K]'
        if unit_simple: unit = 'K'        
        clabel = r'Peak Intensity %s'%unit
        cmap_mom = get_discminer_cmap('intensity_2')    
        cmap_res = get_discminer_cmap('intensity_2', kind='residuals')
        levels_im = np.linspace(0.0, I_res2abs*fclim, 64)
        levels_cc = np.linspace(10, 50, 5)

    sp_dec = str(clim).split('.')
    nsign = 0
    if fmt_vertical:
        nsign = 1
    
    if len(sp_dec)==2:
        ndigs = len(sp_dec[0]) + len(sp_dec[-1])        
        cfmt = '%'+'%d.%df'%(ndigs+1+nsign, len(sp_dec[-1]))
    else:
        ndigs = len(sp_dec[0])
        cfmt = '%'+'%d.1f'%(ndigs+1+1+nsign) #ndigs + . + 1f + nsign

    if 'delta' in moment:
        cfmt = '%d'
        
    return ctitle, clabel, fclim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit

def get_1d_plot_decorators(moment, tag=''):
    I_res2abs = 5.0
    v_res2abs = 20.0
    L_res2abs = 7.0
    Ls_res2abs = 3.0
        
    pars = read_json(decimals=False)
    vsys = float(pars['best_fit']['velocity']['vsys'])
    custom = pars['custom']
    meta = pars['metadata']

    if len(tag)>0: tag = tag+' '
        
    if moment in ['velocity', 'v0phi', 'v0r', 'v0z', 'vr_leftover']:
        unit = '[km/s]' #'[km s$^{-1}$]'
        clim0_res = -custom['vlim']
        clim1_res = custom['vlim']    
        clim0, clim1 = 0.0, v_res2abs*clim1_res + vsys
        clabel = clabel_res = None #Modified within each script
        
    if moment=='linewidth':
        clim0_res = -custom['Llim']
        clim1_res = custom['Llim']
        clim0, clim1 = 0, L_res2abs*clim1_res
        unit = '[km/s]' #'[km s$^{-1}$]'
        clabel = r'Line width %s%s'%(tag, unit)
        clabel_res = r'L. width residuals %s'%unit    

    if moment=='lineslope':
        clim0_res = -custom['Lslim']
        clim1_res = custom['Lslim']
        clim0, clim1 = 0, Ls_res2abs*clim1_res
        unit = ''
        clabel = r'Line slope'
        clabel_res = r'L. slope residuals'
        
    if moment in ['peakintensity', 'peakint']:
        clim0_res = -custom['Ilim']
        clim1_res = custom['Ilim']
        clim0, clim1 = 0.0, I_res2abs*clim1_res
        unit = '[K]'
        clabel = r'Peak Intensity %s%s'%(tag, unit)
        clabel_res = r'Peak Int. residuals %s'%unit    
    
    return clabel, clabel_res, clim0, clim0_res, clim1, clim1_res, unit

def _get_mask_tuples(lims):
    mask_tup = []
    for i in range(len(lims[::2])):
        mask_tup.append(lims[i*2:i*2+2])
    return mask_tup

def _merge_R_phi_mask(mask_R, mask_phi):
    #Mask and merge azimuthal+radial sections

    mask_phi['lims_tup'] = _get_mask_tuples(mask_phi['lims'])
    mask_R['lims_tup'] = _get_mask_tuples(mask_R['lims'])

    mask_sec = np.zeros_like(mask_R['map2d']).astype(bool)

    if len(mask_phi['lims_tup'])==0:
        mask_phi['lims_tup'] = [None]*len(mask_R['lims_tup'])

    if len(mask_R['lims_tup'])==0:
        mask_R['lims_tup'] = [None]*len(mask_phi['lims_tup'])

    cross_sign = lambda x, y: x | y
    same_sign = lambda x, y: x & y
    
    for i in range(len(mask_phi['lims_tup'])):
        lim_phi = mask_phi['lims_tup'][i]
        lim_R = mask_R['lims_tup'][i]

        if (lim_phi is None) and (lim_R is None):
            continue

        elif (lim_phi is not None) and (lim_R is None):
            limdown, limup = lim_phi
            dcond = mask_phi['map2d'] >= limdown
            ucond = mask_phi['map2d'] <= limup            

            if limdown*limup < 0 and limdown > limup:
                mask_func = cross_sign
            else:
                mask_func = same_sign

            mask_sec = mask_sec | mask_func(dcond, ucond)

        elif (lim_phi is None) and (lim_R is not None):
            limdown, limup = lim_R
            mask_sec = mask_sec | ((mask_R['map2d'] >= limdown) & (mask_R['map2d'] <= limup))

        else: #Both are not None
            limdown_R, limup_R = lim_R
            limdown_phi, limup_phi = lim_phi
            dcond = mask_phi['map2d'] >= limdown_phi
            ucond = mask_phi['map2d'] <= limup_phi            

            if limdown_phi*limup_phi < 0 and limdown_phi > limup_phi:
                mask_func = cross_sign
            else:
                mask_func = same_sign
                
            mask_sec = mask_sec | (mask_func(dcond, ucond) & (mask_R['map2d'] >= limdown_R) & (mask_R['map2d'] <= limup_R))

    return mask_sec

def make_masks(ax, mask_R, mask_phi, Rmax=1000, **kwargs_mask):

    kw_mask = dict(edgecolor='none', facecolor='w', alpha=1)
    kw_mask.update(kwargs_mask)

    mask_phi_tup = _get_mask_tuples(mask_phi)
    if len(mask_phi_tup)==0:
        mask_phi_tup = [[0, 360]]

    mask_R_tup = _get_mask_tuples(mask_R)
    if len(mask_R_tup)==0:
        mask_R_tup = [[0, Rmax]]

    def add_mask(mask_phi_i, mask_R_i):
        # Adapted from https://stackoverflow.com/questions/22789356/plot-a-donut-with-fill-or-fill-between
        n, radii = 50, list(mask_R_i)
        theta = np.radians(np.linspace(*mask_phi_i, n, endpoint=True))
        xs = np.outer(radii, np.cos(theta))
        ys = np.outer(radii, np.sin(theta))

        # x,y coords should be traversed in opposite directions
        xs[0,:] = xs[0,::-1]
        ys[0,:] = ys[0,::-1]
        
        ax.fill(np.ravel(xs), np.ravel(ys), **kw_mask)
        ax.fill(np.ravel(xs), np.ravel(ys), edgecolor='k', facecolor='none', lw=1.5)    
        
    for i, maski in enumerate(mask_phi_tup):
        add_mask(mask_phi_tup[i], mask_R_tup[i])
            
def get_noise_mask(
        datacube,
        thres=4, return_mean=True,
        mask_phi={'map2d': None,
                  'lims': [] #List of limits
        },
        mask_R={'map2d': None,
                'lims': [] #List of limits
        }        
):    
    #std from line-free channels, per pixel
    data = datacube.data    
    noise = np.std( np.append(data[:5,:,:], data[-5:,:,:], axis=0), axis=0)

    if return_mean:
        noise = np.nanmean(noise)

    mask = np.nanmax(data, axis=0) < thres*noise
    mask_sec = _merge_R_phi_mask(mask_R, mask_phi)
    mask = mask | mask_sec
    
    return noise, mask


def load_moments(
        args, moment=None, kernel=None, mask=[],
        clip_Rgrid=None, clip_Rmin=0*u.au, clip_Rmax=np.inf*u.au,
        deltas=False
):    
    if moment is None:
        moment = args.moment

    if kernel is None:
        kernel = args.kernel
        
    if kernel in ['gauss', 'gaussian', 'quadratic', 'bell']:
        tag_surf = 'both'
        ref_surf = 'upper' #For contour plots
    else:
        if args.surface in ['up', 'upper']:
            tag_surf = 'up'
            ref_surf = 'upper'
        elif args.surface in ['low', 'lower']:
            tag_surf = 'low'
            ref_surf = 'lower'
            
    if kernel in ['gauss', 'gaussian']:
        tag_base = f'{moment}_gaussian'
    elif kernel in ['quadratic']:
        tag_base = f'{moment}_quadratic'        
    elif kernel in ['bell']:
        tag_base = f'{moment}_bell'
    elif kernel in ['dgauss', 'doublegaussian']:
        tag_base = f'{moment}_{tag_surf}_doublegaussian_{args.kind}'
    elif kernel in ['dbell', 'doublebell']:
        tag_base = f'{moment}_{tag_surf}_doublebell_{args.kind}'

    if deltas:
        tag_base = 'delta_'+tag_base
        
    #Read and mask moment maps, and compute residuals
    moment_data = fits.getdata(tag_base+'_data.fits').squeeze()
    moment_model = fits.getdata(tag_base+'_model.fits').squeeze()

    try:
        if args.smooth>0.0:
            from scipy.ndimage import gaussian_filter 
            moment_data = gaussian_filter(moment_data, args.smooth)
            moment_model = gaussian_filter(moment_model, args.smooth)
            
    except AttributeError:
        pass

    moment_data_unma = moment_data
    moment_model_unma = moment_model
    moment_data[mask] = np.nan
    moment_model[mask] = np.nan
    
    def clip_radially(prop2d): #should be within masking function
        Rmin = clip_Rmin.to('au').value
        Rgrid = np.nan_to_num(clip_Rgrid, nan=np.inf).to('au').value
        Rmax = clip_Rmax.to('au').value
        return np.where((Rgrid<Rmin) | (Rgrid>Rmax), np.nan, prop2d)    

    if isinstance(clip_Rgrid, u.Quantity):
        moment_data = clip_radially(moment_data)
        moment_model = clip_radially(moment_model)

    if moment in ['delta_velocity', 'delta_linewidth']:
        moment_data *= 1000
        moment_model *= 1000
        
    residuals = moment_data - moment_model

    if args.surface in ['low', 'lower']:
        pass
        #from scipy.ndimage import gaussian_filter
        #moment_data = gaussian_filter(moment_data, 0.5)
        #moment_data[moment_data > 50.0] = np.nan

    return moment_data, moment_model, residuals, dict(surf = tag_surf,
                                                      ref_surf = ref_surf,
                                                      base = tag_base,
                                                      mask = np.isnan(moment_data),
                                                      data_unmasked = moment_data_unma,
                                                      model_unmasked = moment_model_unma)


def load_disc_grid():
    R = dict(
        upper=np.load('upper_R.npy'),
        lower=np.load('lower_R.npy')
    )

    phi = dict(
        upper=np.load('upper_phi.npy'),
        lower=np.load('lower_phi.npy')
    )

    z = dict(
        upper=np.load('upper_z.npy'),
        lower=np.load('lower_z.npy')
    )
    return R, phi, z

def make_and_save_filaments(map2d,
                            model=None,
                            tag='',                            
                            writefits=True,
                            return_all=False,
                            cmap='jet',
                            surface='upper'
):

    if model is None:
        _, model = init_data_and_model()
        model.make_model()

    mrail = Rail(model, map2d)
    Rgrid = model.projected_coords['R'][surface]
    
    fil_pos, fil_neg = mrail.make_filaments(
        smooth_size=0.1*model.beam_size,
        adapt_thresh=0.5*model.beam_size,
        size_thresh=100*u.pix**2
    )

    """
    masked_pos = np.ma.masked_where(fil_pos.skeleton!=1, fil_pos.skeleton)
    masked_neg = np.ma.masked_where(fil_neg.skeleton!=1, fil_neg.skeleton)    
    ax.imshow(masked_pos, cmap='Reds_r', alpha=1, zorder=3, extent = [mm.X.min(),mm.X.max(), mm.Y.min(),mm.Y.max()], origin='lower')
    ax.imshow(masked_neg, cmap='Blues_r', alpha=1, zorder=3, extent = [mm.X.min(),mm.X.max(), mm.Y.min(),mm.Y.max()], origin='lower')
    """

    fil_pos_list = [fil_pos.skeleton, fil_pos.skeleton_longpath]
    fil_neg_list = [fil_neg.skeleton, fil_neg.skeleton_longpath]
    fil_pos_obj, fil_neg_obj = [], []

    check_outer_nans = lambda fil: np.isnan(Rgrid[fil.pixel_coords]).any()
        
    for i,fil in enumerate(fil_pos.filaments):
        
        if check_outer_nans(fil):
            continue
        else:
            fil_pos_obj.append(fil)
        
        fil_i = np.zeros_like(fil_pos.skeleton)
        fil_i[fil.pixel_coords] = 1 
        fil_pos_list.append(fil_i)

        
    for i,fil in enumerate(fil_neg.filaments):
        
        if check_outer_nans(fil):
            continue
        else:
            fil_neg_obj.append(fil)
        
        fil_i = np.zeros_like(fil_neg.skeleton)
        fil_i[fil.pixel_coords] = 1 
        fil_neg_list.append(fil_i)

    fil_pos.filaments = fil_pos_obj
    fil_neg.filaments = fil_neg_obj
    
    if writefits:
        if len(tag)>0:
            if tag[0]!='_':
                tag='_'+tag
                
        fits.writeto('filaments_pos%s.fits'%tag, np.asarray(fil_pos_list), header=model.header, overwrite=True)
        fits.writeto('filaments_neg%s.fits'%tag, np.asarray(fil_neg_list), overwrite=True)

    cmap = matplotlib.cm.get_cmap(cmap)
    npos = len(fil_pos.filaments)
    nneg = len(fil_neg.filaments)
    cpos = np.linspace(0.6, 1.0, npos)        
    cneg = np.linspace(0.4, 0.0, nneg)
    colors_dict = {}
    colors_dict.update({i+1: matplotlib.colors.to_hex(cmap(cpos[i])) for i in range(npos)})
    colors_dict.update({-i-1: matplotlib.colors.to_hex(cmap(cneg[i])) for i in range(nneg)})
        
    if return_all:
        return fil_pos, fil_neg, fil_pos_list, fil_neg_list, colors_dict       

    else:
        return fil_pos_list, fil_neg_list, colors_dict


def mark_planet_location(ax, args, r=[], phi=[], labels=[], coords='disc', model=None, zfunc=None, zpars=None, incl=None, PA=None, xc=None, yc=None, dpc=None, midplane=True, kwargs_text={}, **kwargs_scatter):

    kwargs_sc = dict(edgecolors='gold', facecolors='none', marker='o', s=450, lw=4.5, alpha=1.0, label=None, zorder=22)
    kwargs_sc.update(kwargs_scatter)

    kwargs_tx = dict(fontsize=MEDIUM_SIZE-2, color='k', ha='center', va='center', weight='bold', rotation=0, zorder=23)
    kwargs_tx.update(kwargs_text)

    if len(r)==0:
        rs = args.rp
        phis = args.phip
    else:
        rs = r
        phis = phi

    npoints = len(rs)    

    if len(labels)==0:
        labels = args.labelp

    if len(labels)==1:
        labels = labels*npoints

    #Fetch orientation and surface parameters from model obj if passed
    if (coords=='sky' or args.input_coords=='sky') and model is not None:

        dpc = model.dpc
        orientation = model.params['orientation']
        incl = orientation['incl']
        PA = orientation['PA']
        xc = orientation['xc']
        yc = orientation['yc']

        if args.surface=='upper':
            zpars = model.params['height_upper']
            zfunc = model.z_upper_func
        else:
            zpars = model.params['height_lower']
            zfunc = model.z_lower_func
            
    xcoords, ycoords = [], []
    for i in range(npoints):

        if args.input_coords in ['disc', 'disk']:
            if coords=='disc': #Coordinate system for the planet marker
                if args.projection=='cartesian':
                    phii = np.radians(phis[i])
                    xi, yi = rs[i]*np.cos(phii), rs[i]*np.sin(phii)
            
                elif args.projection=='polar':
                    xi, yi = phis[i], rs[i]

            elif coords=='sky':
                phii = np.radians(phis[i])
                zp = zfunc({'R': rs[i]*u.au.to('m')}, **zpars)*u.m.to('au')
                if midplane:
                    zp*=0
                xi,yi,zi = GridTools.get_sky_from_disc_coords(rs[i], phii, zp, incl, PA, xc, yc) 

        elif args.input_coords=='sky':
            rsky = rs[i]*(dpc.to('pc')).value #Projected distance in au
            phisky = np.radians(phis[i]) #Position angle, measured from the North
            xsky = -rsky*np.sin(phisky)
            ysky = rsky*np.cos(phisky)
            
            if coords=='disc':
                xdisc, ydisc = GridTools.get_disc_from_sky_coords(xsky, ysky, zfunc, zpars, incl, PA, xc=xc, yc=yc, midplane=midplane)
                
                if args.projection=='cartesian':
                    xi, yi = xdisc, ydisc
                    ri = np.hypot(xdisc, ydisc)
                    print (ri)
                    
                elif args.projection=='polar':
                    phii = np.arctan2(ydisc, xdisc)
                    ri = np.hypot(xdisc, ydisc)
                    xi, yi = np.degrees(phii), ri

            elif coords=='sky':
                xi, yi = xsky, ysky
                
        ax.scatter(xi, yi, **kwargs_sc)
        
        if len(labels)>0:
            ax.text(xi, yi, labels[i], **kwargs_tx)

        xcoords.append(xi)
        ycoords.append(yi)
        
    return xcoords, ycoords

def overlay_filaments(ax, x, y, prop, projection='cartesian', **kwargs):

    fil_pos_obj, fil_neg_obj, *_ = make_and_save_filaments(prop, return_all=True, **kwargs)

    if projection=='cartesian':
        ax.contour(x, y, fil_pos_obj.skeleton, linewidths=0.2, colors='darkred', alpha=1, zorder=11)
        ax.contour(x, y, fil_neg_obj.skeleton, linewidths=0.2, colors='navy', alpha=1, zorder=11)

    if projection=='polar':
        proj_fil = lambda fil_obj: (x[fil_obj.skeleton.astype(bool)], y[fil_obj.skeleton.astype(bool)])
        ax.scatter(*proj_fil(fil_pos_obj), marker='o', lw=0.9, s=20, ec='k', fc='tomato', alpha=1, zorder=11)
        ax.scatter(*proj_fil(fil_neg_obj), marker='o', lw=0.9, s=20, ec='k', fc='dodgerblue', alpha=1, zorder=11)

    
    
def overlay_continuum(ax, parfile='parfile_scattered.json', comp=-1, convert_to_tb=False, vmin=0.1, vmax=1.0, lev=32, contours=False, coords='disc', surface='upper', data_cont=None, model_cont=None, model_proj=None, **kwargs_contourf):

    if data_cont is None or model_cont is None:
        data_cont, model_cont = init_data_and_model(parfile=parfile, twodim=True, write_extent=False)

    if convert_to_tb:
        try:
            data_cont.convert_to_tb(writefits=False)
        except KeyError:
            warnings.warn('\nUnable to convert continuum intensity to brightness temperature units...\n', Warning)
    
    if model_proj is None:
        R_cont, phi_cont, z_cont, R_nonan, phi_nonan, z_nonan = model_cont.get_projected_coords(writebinaries=False)
    else:
        R_cont, phi_cont, z_cont, R_nonan, phi_nonan, z_nonan = model_proj
    

    img = data_cont.data[comp]
    peak_img = np.nanmax(img)
    kwargs = dict(levels=np.linspace(vmin*peak_img, vmax*peak_img, lev))
    kwargs.update(kwargs_contourf)

    Xproj_cont = R_cont[surface]*np.cos(phi_cont[surface])*u.m.to('au')
    Yproj_cont = R_cont[surface]*np.sin(phi_cont[surface])*u.m.to('au')

    if contours:
        levels=np.linspace(vmin*peak_img, vmax*peak_img, lev)                        
        kwargs.update(levels=levels)
        linewidths = 2.0*levels/np.max(levels)

        if coords=='sky':        
            im = ax.contour(img, extent=model_cont.extent, levels=levels, colors='k', linewidths=linewidths, zorder=30) 
        if coords in ['disc', 'disk']:
            im = ax.contour(Xproj_cont, Yproj_cont, img, levels=levels, colors='k', linewidths=linewidths, zorder=30) 

    if coords=='sky':
        im = ax.contourf(img, extent=model_cont.extent, **kwargs)
    if coords in ['disc', 'disk']:
        im = ax.contourf(Xproj_cont, Yproj_cont, img, **kwargs)
            
    return im, data_cont, model_cont


def overlay_spirals(ax, args, mtags, Rmin=0, Rmax=np.inf, phi_extent=2*np.linspace(-360, 360, 500)):

    #SPIRAL PRESCRIPTIONS
    sp_lin = lambda x, a, b: a + b*x
    sp_log = lambda x, a, b: a*np.exp(b*x)
    
    arr_read = np.loadtxt(
        'spirals_fit_parameters_%s_%s.txt'%(mtags['base'].replace(args.moment, args.spiral_moment), args.spiral_type),
        dtype = {
            'names': ('a', 'b', 'color', 'sign', 'id'),
            'formats': (float, float, 'U10', 'U10', int)
        },
        skiprows = 1,
        comments = None,
    )

    arr_read = np.atleast_1d(arr_read)
    phi_ext_rad = np.radians(phi_extent)

    if args.spiral_type == 'linear':
        sp_func = sp_lin
    else:
        sp_func = sp_log
            
    for arr in arr_read:
        if (
                (arr[3]=='pos' and arr[4] in args.spiral_ids)
                or
                (arr[3]=='neg' and arr[4] in args.spiral_ids)
        ):
            #Can this be translated back to sky coordinates?
            R_ext = sp_func(phi_ext_rad, *tuple(arr)[:2])
            ind = (R_ext > Rmin) & (R_ext < Rmax)
            xplot = R_ext[ind]*np.cos(phi_ext_rad[ind])
            yplot = R_ext[ind]*np.sin(phi_ext_rad[ind])            
            ax.plot(xplot, yplot, lw=3.5,  color=arr[2], zorder=20)
            ax.plot(xplot, yplot, lw=4.5, color='k', zorder=19)

def save_fits(data, header, overwrite=True, momenttype='velocity', filename='map.fits'):

    hdr = copy.copy(header)
    if momenttype=='velocity':
        hdr["BUNIT"] = "km/s"
        hdr["BTYPE"] = "Velocity"

    kwargs = dict(overwrite=overwrite, header=hdr)

    fits.writeto(filename, data, **kwargs)

def show_output(args):
    if args.show_output:
        matplotlib.pyplot.show(block=args.show_block)
    else:
        matplotlib.pyplot.close()
