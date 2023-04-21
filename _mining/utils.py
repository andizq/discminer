"""
Useful repeating code
"""
import json
import decimal
import warnings
import copy

from discminer.plottools import get_discminer_cmap
from discminer.core import Data
from discminer.disc2d import Model
import discminer.cart as cart

from astropy.io import fits
import astropy.units as u

import numpy as np

molplot = {
    '12co': r'$^{\rm 12}$CO',
    '13co': r'$^{\rm 13}$CO',
    'cs': r'CS'
}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SMALL_SIZE = 10
MEDIUM_SIZE = 15

    
def add_parser_args(parser,
                    moment=False, kind=False, surface=False, projection=False,
                    mask_minor=False, mask_major=False, Rinner=False, Router=False,
                    fold=False, writetxt=False, 
):

    if moment:
        parser.add_argument('-m', '--moment', default='velocity', type=str, choices=['velocity', 'linewidth', 'lineslope', 'peakint', 'peakintensity'], help="velocity, linewidth or peakintensity")

    if mask_minor:
        parser.add_argument('-b', '--mask_minor', default=60.0, type=float, help="+- azimuthal mask around disc minor axis for computation of vphi and vz velocity components")
    if mask_major:
        parser.add_argument('-a', '--mask_major', default=30.0, type=float, help="+- azimuthal mask around disc major axis for computation of vR velocity component")
    if kind:
        parser.add_argument('-k', '--kind', default='gaussian', type=str, choices=['gauss', 'gaussian', 'bell', 'dgauss', 'doublegaussian', 'dbell', 'doublebell'], help="gauss(or gaussian), dbell(or doublebell)")
    if surface:
        parser.add_argument('-s', '--surface', default='upper', type=str, choices=['up', 'upper', 'low', 'lower'], help="upper or lower surface moment map")
    if fold:
        parser.add_argument('-f', '--fold', default='absolute', type=str, choices=['absolute', 'standard'], help="if moment=velocity, fold absolute or standard velocity residuals")
    if projection:
        parser.add_argument('-p', '--projection', default='cartesian', type=str, choices=['cartesian', 'polar'], help="Project residuals onto a cartesian or a polar map")
    if writetxt:
        parser.add_argument('-w', '--writetxt', default=False, type=bool, help="write txt files")
    if Rinner:
        parser.add_argument('-i', '--Rinner', default=1.0, type=float, help="Number of beams to mask out from inner region")
    if Router:
        parser.add_argument('-o', '--Router', default=0.9, type=float, help="Fraction of Rout to consider as the outer radius for the analysis")

    args = parser.parse_args()

    try:
        if args.moment=='peakint':
            args.moment = 'peakintensity'
    except AttributeError:
        pass

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
        
        model.z_upper_func = cart.z_upper_exp_tapered
        model.z_lower_func = cart.z_lower_exp_tapered
        model.velocity_func = model.keplerian_vertical # vrot = sqrt(GM/r**3)*R
        model.line_profile = model.line_profile_bell
        model.line_uplow = model.line_uplow_mask
    
        if 'I2pwl' in meta['kind']:
            model.intensity_func = cart.intensity_powerlaw_rbreak
        elif 'I2pwlnosurf' in meta['kind']:
            model.intensity_func = cart.intensity_powerlaw_rbreak_nosurf    
        else:
            model.intensity_func = cart.intensity_powerlaw_rout

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
    
    ctitle = meta['disc'].upper()+' '+molplot[meta['mol']]
        
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
    
def load_moments(args, moment=None):
    if moment is None: moment = args.moment
    if moment=='peakint': moment='peakintensity'
    
    if args.kind in ['gauss', 'gaussian', 'bell']:
        tag_surf = 'both' 
    else:
        if args.surface in ['up', 'upper']:
            tag_surf = 'up' 
        elif args.surface in ['low', 'lower']:
            tag_surf = 'low'

    if args.kind in ['gauss', 'gaussian']:
        tag_base = f'{moment}_gaussian'
    elif args.kind in ['bell']:
        tag_base = f'{moment}_bell'
    elif args.kind in ['dgauss', 'doublegaussian']:
        tag_base = f'{moment}_{tag_surf}_doublegaussian_mask'
    elif args.kind in ['dbell', 'doublebell']:
        tag_base = f'{moment}_{tag_surf}_doublebell_mask'

    moment_data = fits.getdata(tag_base+'_data.fits')
    moment_model = fits.getdata(tag_base+'_model.fits')
    
    return moment_data, moment_model, dict(surf=tag_surf, base=tag_base)

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
    kwargs_def = dict(frameon=False, fontsize=MEDIUM_SIZE-2, ncol=3, loc='lower right', bbox_to_anchor=(1.0, 1.0))
    kwargs_def.update(kwargs)
    return ax.legend(**kwargs_def)
