"""
Useful repeating code
"""
import json
import decimal
import warnings
import copy

from discminer.plottools import get_discminer_cmap
from discminer.core import Data
from discminer.rail import Rail
from discminer.disc2d import Model
import discminer.cart as cart

from astropy.io import fits
import astropy.units as u

import numpy as np
import matplotlib

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
    'j1852': r'$\rm J1852$'
}

moltex = {
    '12co': r'$^{\rm 12}$CO',
    '13co': r'$^{\rm 13}$CO',
    'cs': r'CS'
}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SMALL_SIZE = 10
MEDIUM_SIZE = 15

    
def add_parser_args(parser,
                    moment=False, kernel=False, kind=False, surface=False, projection=False,
                    mask_minor=False, mask_major=False, Rinner=False, Router=False,
                    fold=False, writetxt=False, sigma=False, gradient=False, smooth=False
):

    def set_default(val, default):
        if isinstance(val, bool):
            return default
        else:
            return val
        
    if moment:
        d0 = set_default(moment, 'velocity')
        parser.add_argument('-m', '--moment', default=d0, type=str,
                            choices=['velocity', 'linewidth', 'lineslope', 'peakint', 'peakintensity'],
                            help="Type of moment map to be analysed. DEFAULTS to '%s'"%d0)
    if mask_minor:
        d0 = set_default(mask_minor, 30.0)
        parser.add_argument('-b', '--mask_minor', default=d0, type=float,
                            help="+- azimuthal mask around disc minor axis for calculation of vphi and vz velocity components. DEFAULTS to %.1f deg"%d0)
    if mask_major:
        d0 = set_default(mask_major, 30.0)
        parser.add_argument('-a', '--mask_major', default=d0, type=float,
                            help="+- azimuthal mask around disc major axis for calculation of vR velocity component. DEFAULTS to %.1f deg"%d0)
    if kernel:
        d0 = set_default(kernel, 'gaussian')
        parser.add_argument('-k', '--kernel', default=d0, type=str,
                            choices=['gauss', 'gaussian', 'bell', 'quadratic', 'dgauss', 'doublegaussian', 'dbell', 'doublebell'],
                            help="Kernel utilised for line profile fit and computation of moment maps. DEFAULTS to '%s'"%d0)
    if kind:
        d0 = set_default(kind, 'mask')
        parser.add_argument('-ki', '--kind', default=d0, type=str, choices=['mask', 'sum'],
                            help="How the upper and lower surface kernel profiles must be merged. DEFAULTS to '%s'"%d0)
    if surface:
        d0 = set_default(surface, 'upper')
        parser.add_argument('-s', '--surface', default=d0, type=str,
                            choices=['up', 'upper', 'low', 'lower'],
                            help="Use upper or lower surface moment map. DEFAULTS to '%s'"%d0)
    if fold:
        d0 = set_default(fold, 'absolute')        
        parser.add_argument('-f', '--fold', default=d0, type=str,
                            choices=['absolute', 'standard'],
                            help="if moment=velocity, fold absolute or standard velocity residuals. DEFAULTS to '%s'"%d0)
    if projection:
        d0 = set_default(fold, 'cartesian')                
        parser.add_argument('-p', '--projection', default=d0, type=str,
                            choices=['cartesian', 'polar'],
                            help="Project residuals onto a cartesian or a polar map. DEFAULTS to '%s'"%d0)
    if writetxt:
        d0 = set_default(writetxt, 1)                
        parser.add_argument('-w', '--writetxt', default=d0, type=int,
                            choices=[0, 1],
                            help="write output into txt file(s). DEFAULTS to %d"%d0)
    if Rinner:
        d0 = set_default(Rinner, 1.0)                        
        parser.add_argument('-i', '--Rinner', default=d0, type=float,
                            help="Number of beams to mask out from disc inner region. DEFAULTS to %.2f"%d0)
    if Router:
        d0 = set_default(Router, 0.98)                
        parser.add_argument('-o', '--Router', default=d0, type=float,
                            help="Fraction of Rout to consider as the disc outer radius for the analysis. DEFAULTS to %.2f"%d0)
    if sigma:
        d0 = set_default(sigma, 5)                
        parser.add_argument('-si', '--sigma', default=d0, type=float,
                            help="Mask out pixels with values below sigma threshold. DEFAULTS to %.1f"%d0)
    if gradient:
        d0 = set_default(gradient, 'r')
        parser.add_argument('-g', '--gradient', default=d0, type=str, choices=['peak', 'r', 'phi'],
                            help="Coordinate along which the gradient will be computed. If 'peak', the maximum gradient is computed. DEFAULTS to '%s'"%d0)

    if smooth:
        d0 = set_default(smooth, 0.0)
        parser.add_argument('-sm', '--smooth', default=d0, type=float,
                            help="Smooth moment_data using skimage.restoration library. DEFAULTS to %.1f"%d0)        

    args = parser.parse_args()

    try:
        if args.moment=='peakint':
            args.moment = 'peakintensity'
    except AttributeError:
        pass

    try:
        if args.surface=='up':
            args.surface = 'upper'
    except AttributeError:
        args.surface = 'upper'

    try:
        if args.surface=='low':
            args.surface = 'lower'
    except AttributeError:
        args.surface = 'upper'

    try:
        if args.kernel=='gauss':
            args.kernel = 'gaussian'
    except AttributeError:
        args.kernel = 'gaussian'
    
    try:
        if args.kernel=='dgauss':
            args.kernel = 'doublegaussian'
    except AttributeError:
        args.kernel = 'gaussian'

    try:
        if args.kernel=='dbell':
            args.kernel = 'doublebell'
    except AttributeError:
        args.kernel = 'gaussian'
    
    return args


def init_data_and_model(parfile='parfile.json', Rmin=0, Rmax=1.1, init_model=True):
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

    datacube = Data(file_data, dpc) # Read data and convert to Cube object

    if init_model:
        Rmax = Rmax*Rout*u.au
        model = Model(datacube, Rmax=Rmax, Rmin=Rmin, prototype=True)
        
        model.velocity_func = model.keplerian_vertical # vrot = sqrt(GM/r**3)*R
        model.line_profile = model.line_profile_bell

        if 'sum' in meta['kind']:
            model.line_uplow = model.line_uplow_sum
        else:
            model.line_uplow = model.line_uplow_mask

        if 'I2pwl' in meta['kind']:
            model.intensity_func = cart.intensity_powerlaw_rbreak
        elif 'I2pwlnosurf' in meta['kind']:
            model.intensity_func = cart.intensity_powerlaw_rbreak_nosurf    
        else:
            model.intensity_func = cart.intensity_powerlaw_rout

        if 'surf2pwl' in meta['kind']:
            model.z_upper_func = cart.z_upper_powerlaw
            model.z_lower_func = cart.z_lower_powerlaw
        else:
            model.z_upper_func = cart.z_upper_exp_tapered
            model.z_lower_func = cart.z_lower_exp_tapered
            
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
        
    if moment=='velocity':
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
        levels_im = np.linspace(0.0, L_res2abs*fclim, 48)
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
        cmap_mom = get_discminer_cmap('intensity')    
        cmap_res = get_discminer_cmap('intensity', kind='residuals')
        levels_im = np.linspace(0.0, I_res2abs*fclim, 48)
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
        
    if moment=='velocity':
        unit = '[km s$^{-1}$]'
        clim0_res = -custom['vlim']
        clim1_res = custom['vlim']    
        clim0, clim1 = 0.0, v_res2abs*clim1_res + vsys
        clabel = clabel_res = None #Modified within each script
        
    if moment=='linewidth':
        clim0_res = -custom['Llim']
        clim1_res = custom['Llim']
        clim0, clim1 = 0, L_res2abs*clim1_res
        unit = '[km s$^{-1}$]'
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

def get_noise_mask(datacube, thres=4, return_mean=True):
    #std from line-free channels, per pixel
    data = datacube.data
    noise = np.std( np.append(data[:5,:,:], data[-5:,:,:], axis=0), axis=0) 
    if return_mean:
        noise_mean = np.mean(noise)
        mask = np.nanmax(data, axis=0) < thres*noise_mean
        return noise_mean, mask
    else:
        mask = np.nanmax(data, axis=0) < thres*noise
        return noise, mask

def load_moments(
        args, moment=None, kernel=None, mask=[],
        clip_Rgrid=None, clip_Rmin=0*u.au, clip_Rmax=np.inf*u.au
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

    #Read and mask moment maps, and compute residuals
    moment_data = fits.getdata(tag_base+'_data.fits')
    moment_model = fits.getdata(tag_base+'_model.fits')

    moment_data[mask] = np.nan
    moment_model[mask] = np.nan

    try:
        if args.smooth>0.0:
            from scipy.ndimage import gaussian_filter 
            moment_data = gaussian_filter(moment_data, args.smooth)
            moment_model = gaussian_filter(moment_model, args.smooth)                        
    except AttributeError:
        pass
    
    def clip_radially(prop2d): #should be within masking function in discminer         
        Rmin = clip_Rmin.to('au').value
        Rgrid = np.nan_to_num(clip_Rgrid).to('au').value
        Rmax = clip_Rmax.to('au').value
        return np.where((Rgrid<Rmin) | (Rgrid>Rmax), np.nan, prop2d)    

    if isinstance(clip_Rgrid, u.Quantity):
        moment_data = clip_radially(moment_data)
        moment_model = clip_radially(moment_model)

    residuals = moment_data - moment_model

    if args.surface in ['low', 'lower']:
        pass
        #from scipy.ndimage import gaussian_filter
        #moment_data = gaussian_filter(moment_data, 0.5)
        #moment_data[moment_data > 50.0] = np.nan

    return moment_data, moment_model, residuals, dict(surf=tag_surf, ref_surf=ref_surf, base=tag_base)
    
    
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

def make_1d_legend(ax, **kwargs):
    kwargs_def = dict(
        frameon=False,
        fontsize=MEDIUM_SIZE,
        ncol=3,
        handlelength=2.0,
        handletextpad=0.5,
        borderpad=0.0,
        columnspacing=1.5,
        loc='lower right',
        bbox_to_anchor=(1.0, 1.0)
    )
    kwargs_def.update(kwargs)
    return ax.legend(**kwargs_def)

def make_and_save_filaments(model, map2d,
                            writefits=True, tag='',
                            return_all=False, cmap='jet',
                            surface='upper'
):
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

    check_outer_nans = lambda fil: np.isnan(Rgrid[fil.pixel_coords]).any()
        
    for i,fil in enumerate(fil_pos.filaments):
        
        if check_outer_nans(fil):
            del fil_pos.filaments[i]
            continue
        
        fil_i = np.zeros_like(fil_pos.skeleton)
        fil_i[fil.pixel_coords] = 1 
        fil_pos_list.append(fil_i)

        
    for i,fil in enumerate(fil_neg.filaments):
        
        if check_outer_nans(fil):
            del fil_neg.filaments[i]
            continue
        
        fil_i = np.zeros_like(fil_neg.skeleton)
        fil_i[fil.pixel_coords] = 1 
        fil_neg_list.append(fil_i)

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

