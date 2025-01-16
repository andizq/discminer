from discminer.mining_control import _mining_gradient
from discminer.mining_utils import (load_disc_grid,
                                    init_data_and_model,
                                    get_noise_mask,
                                    get_2d_plot_decorators,
                                    mark_planet_location,
                                    show_output,
                                    load_moments)

import discminer.cart as cart
from discminer.plottools import (make_polar_map,
                                 make_substructures,
                                 make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 make_1d_legend,
                                 use_discminer_style)

import json
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_gradient(None)
    args = parser.parse_args()

args.projection = 'polar'

#**************************
#JSON AND SOME DEFINITIONS
#**************************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

vsys = best['velocity']['vsys']
Rout = best['intensity']['Rout']
incl = best['orientation']['incl']
PA = best['orientation']['PA']
xc = best['orientation']['xc']
yc = best['orientation']['yc']

gaps = custom['gaps']
rings = custom['rings']

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, unit_simple=True, fmt_vertical=True)

levels_resid = np.linspace(-clim, clim, 48)

clim_grad = 0.04*clim #For gradient: dv/dR [unit/au]
cmap_grad = plt.get_cmap('nipy_spectral')
cmap_grad_r = plt.get_cmap('nipy_spectral_r')

levels_grad = np.linspace(0.05*clim_grad, clim_grad, 48)

file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.1*Rout*u.au #Max model radius, 10% larger than disc Rout

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model()

noise_mean, mask = get_noise_mask(datacube, thres=2)
vchannels = datacube.vchannels
model.make_model()

R, phi, z = load_disc_grid()

#*****************
#LOAD MOMENT MAPS    
#*****************
if args.absolute_Rinner>=0:
    Rin_plt = args.absolute_Rinner
else:
    Rin_plt = args.Rinner*datacube.beam_size

if args.absolute_Router>=0:
    Rout_plt = args.absolute_Router*u.au
else:
    Rout_plt = args.Router*Rout*u.au

moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask,
    clip_Rmin=0.0*u.au,
    clip_Rmax=Rout*u.au,
    clip_Rgrid=R[args.surface]*u.m
)

#*******************
#SURFACE PROPERTIES
#*******************
if args.surface in ['up', 'upper']:
    if 'surf2pwl' in meta['kind']:
        z_func = cart.z_upper_powerlaw
    elif 'surfirregular' in meta['kind']:
        z_func = cart.z_upper_irregular
    else:
        z_func = cart.z_upper_exp_tapered
    z_pars = best['height_upper']

elif args.surface in ['low', 'lower']:
    if 'surf2pwl' in meta['kind']:
        z_func = cart.z_lower_powerlaw
    elif 'surfirregular' in meta['kind']:
        z_func = cart.z_lower_irregular            
    else:
        z_func = cart.z_lower_exp_tapered
    z_pars = best['height_lower']

#**************
#FIGURE LABELS
#**************
if args.moment in ['velocity', 'v0phi', 'v0r', 'v0z', 'vr_leftover', 'linewidth']:
    unit = 'm/s'
    ufac = u.Unit('km/s').to(unit)
    dfmt = '%3d'
elif args.moment=='peakintensity':
    unit = 'K'
    ufac = 1
    dfmt = '%4.1f'
else:
    unit = ''
    ufac = 1
    dfmt = '%4.1f'
    
clabels = {
    'linewidth': r'$\Delta$ Line width [m s$^{-1}$]',
    'lineslope': r'$\Delta$ Line slope',
    'velocity': r'$\Delta$ Centroid [m s$^{-1}$]',
    'v0phi': r'$\Delta$ Centroid [m s$^{-1}$]',
    'v0r': r'$\Delta$ Centroid [m s$^{-1}$]',
    'v0z': r'$\Delta$ Centroid [m s$^{-1}$]',
    'vr_leftover': r'$\Delta$ Centroid [m s$^{-1}$]',    
    'peakintensity': r'$\Delta$ Peak Int. [K]'
}

clabel_ = clabels[args.moment].split(' [')
clabel_unit = unit #clabel_[1].split(']')[0]
clabel_prop = clabel_[0]#.split('$')[-1]
cu = (clabel_prop, clabel_unit)

clabels_grad = {
    'r': r'$\delta/\delta R$ (%s) [%s/au]'%cu,
    #'phi': r'$\delta/R\delta \phi$ (%s) [%s/au]'%cu,
    'phi': r'$\nabla_\phi$ (%s) [%s/au]'%cu,
    'peak': r'$\delta/\delta_{\rm peak}$ (%s) [%s/au]'%cu,
}

#***********
#MAKE PLOTS
#***********
if args.gradient=='peak':
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12, 6.5))
else:
    fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(12, 10))

#Positive gradient    
_, _, cbar = make_polar_map(ufac*residuals, ufac*levels_grad,
                            R[args.surface]*u.m, phi[args.surface]*u.rad, Rout_plt,
                            fig=fig, ax=ax[0],
                            Rin=Rin_plt, gradient=args.gradient,
                            kwargs_gradient_peaks={'threshold': args.threshold},
                            findpeaks='pos', filepeaks='phi_gradient_peaks_%s_positive.txt'%mtags['base'],
                            cmap=cmap_grad, fmt=dfmt, clabel=clabels_grad[args.gradient])
if args.gradient!='peak':
    #Negative gradient
    _, _, cbar = make_polar_map(ufac*residuals, ufac*np.sort(-levels_grad),
                                R[args.surface]*u.m, phi[args.surface]*u.rad, Rout_plt,
                                fig=fig, ax=ax[1],
                                Rin=Rin_plt, gradient=args.gradient,
                                kwargs_gradient_peaks={'threshold': args.threshold},                                
                                findpeaks='neg', filepeaks='phi_gradient_peaks_%s_negative.txt'%mtags['base'],
                                cmap=cmap_grad_r, fmt=dfmt, clabel=clabels_grad[args.gradient])

#REFERENCE MAP
_, _, cbar = make_polar_map(ufac*residuals, ufac*levels_resid,
                            R[args.surface]*u.m, phi[args.surface]*u.rad, Rout_plt,
                            fig=fig, ax=ax[-1],
                            Rin=Rin_plt, gradient=0,
                            cmap=cmap_res, fmt='%4d', clabel=clabels[args.moment])


if args.moment in ['velocity', 'linewidth']:
    try:
        peaks_pos = np.loadtxt('phi_gradient_peaks_%s_positive.txt'%mtags['base']).T    
        peaks_neg = np.loadtxt('phi_gradient_peaks_%s_negative.txt'%mtags['base']).T

        kwargs_peaks = dict(color='indigo', lw=4, alpha=1.0, s=250)    
        ax[-1].scatter(peaks_pos[0], peaks_pos[1], marker='+', **kwargs_peaks)
        ax[-1].scatter(peaks_neg[0], peaks_neg[1], marker='_', **kwargs_peaks, label='Negative peak gradient')
        ax[0].scatter(None, None, label='Positive peak gradient', marker='+', **kwargs_peaks)
        ax[0].scatter(None, None, label='Negative peak gradient', marker='_', **kwargs_peaks)        
            
    except FileNotFoundError:
        pass

if args.show_legend:
    make_1d_legend(ax[0], handlelength=1.5, loc='lower center', bbox_to_anchor=(0.5, 1.2))
    
for axi in ax:
    if axi!=ax[-1]:
        axi.set_xlabel(None)
        kc = {'color': '1.0'}
    else:
        kc = {}

    if axi==ax[-1]:
        label_rings = True
    elif axi==ax[-2]:
        axi.set_ylabel(None)        
        label_rings = True
    else:
        axi.set_ylabel(None)
        label_rings = False

    make_substructures(axi, gaps=gaps, rings=rings, twodim=True, polar=True, label_rings=label_rings, kwargs_gaps=kc, kwargs_rings=kc)
    
mark_planet_location(ax[-1], args, edgecolor='k', lw=3.5, s=550, coords='disc', zfunc=z_func, zpars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc, dpc=dpc)    
ax[0].set_title(ctitle, fontsize=16, color='k')

plt.savefig('gradient_residuals_deproj_%s_d%s_polar.png'%(mtags['base'], args.gradient), bbox_inches='tight', dpi=200)
show_output(args)
