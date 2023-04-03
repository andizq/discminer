"""
Useful repeating code
"""
import json
import decimal
import numpy as np
from astropy.io import fits
import warnings

from discminer.plottools import get_discminer_cmap

molplot = {
    '12co': r'$^{\rm 12}$CO',
    '13co': r'$^{\rm 13}$CO',
    'cs': r'CS'
}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def add_parser_args(parser,
                    moment=False, kind=False, surface=False, projection=False,
                    mask_ang=False, Rinner=False, Router=False,
                    fold=False, writetxt=False, 
):

    if moment:
        parser.add_argument('-m', '--moment', default='velocity', type=str, choices=['velocity', 'linewidth', 'lineslope', 'peakint', 'peakintensity'], help="velocity, linewidth or peakintensity")
    if mask_ang:
        parser.add_argument('-a', '--mask_ang', default=60.0, type=float, help="+- azimuthal mask around disc minor axis for computation of velocity components")
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
        
    return parser.parse_args()
        
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

def get_noise_mask(data, thres=4, return_mean=True):
    #std from line-free channels, per pixel
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
