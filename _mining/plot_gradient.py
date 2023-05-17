from discminer.plottools import (make_round_map,
                                 make_polar_map,
                                 make_substructures,
                                 make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 use_discminer_style)

from utils import (load_disc_grid,
                   init_data_and_model,
                   get_noise_mask,
                   add_parser_args,
                   get_2d_plot_decorators,
                   load_moments)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import json

from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='plot gradient residual maps', description='Plot gradient residual maps')
parser.add_argument('-g', '--gradient', default='r', type=str, choices=['peak', 'r', 'phi'], help="Coordinate along which the gradient will be computed. If 'peak', the maximum gradient is computed.")

args = add_parser_args(parser, moment=True, kind=True, surface=True, Rinner=True, Router=True)

#**********************
#JSON AND PARSER STUFF
#**********************
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

clim_grad = 0.05*clim #For gradient: dv/dR [unit/au]
cmap_grad = plt.get_cmap('nipy_spectral')
cmap_grad_r = plt.get_cmap('nipy_spectral_r')

levels_grad = np.linspace(0, clim_grad, 48)

#****************
#SOME DEFINITIONS
#****************
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

#*************************
#LOAD MOMENT MAPS    
Rin_plt = args.Rinner*datacube.beam_size
Rout_plt = args.Router*Rout*u.au

moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask,
    clip_Rmin=0.0*u.au,
    clip_Rmax=Rout*u.au,
    clip_Rgrid=R[args.surface]*u.m
)

#*******************************
#DECORATIVE STUFF

if args.moment in ['velocity', 'linewidth']:
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
    'peakintensity': r'$\Delta$ Peak Int. [K]'
}

clabel_ = clabels[args.moment].split(' [')
clabel_unit = unit #clabel_[1].split(']')[0]
clabel_prop = clabel_[0]#.split('$')[-1]
cu = (clabel_prop, clabel_unit)

clabels_grad = {
    'r': r'$\delta/\delta R$ (%s) [%s/au]'%cu,
    'phi': r'$\delta/R\delta \phi$ (%s) [%s/au]'%cu,
    'peak': r'$\delta/\delta_{\rm peak}$ (%s) [%s/au]'%cu,
}

#**********
#MAKE PLOTS
#**********

if args.gradient=='peak':
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12, 6.5))
else:
    fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(12, 10))

#Positive gradient    
_, _, cbar = make_polar_map(ufac*residuals, ufac*levels_grad,
                            R[args.surface]*u.m, phi[args.surface]*u.rad, Rout_plt,
                            fig=fig, ax=ax[0],
                            Rin=Rin_plt, gradient=args.gradient,
                            cmap=cmap_grad, fmt=dfmt, clabel=clabels_grad[args.gradient])
if args.gradient!='peak':
    #Negative gradient
    _, _, cbar = make_polar_map(ufac*residuals, ufac*np.sort(-levels_grad),
                                R[args.surface]*u.m, phi[args.surface]*u.rad, Rout_plt,
                                fig=fig, ax=ax[1],
                                Rin=Rin_plt, gradient=args.gradient,
                                cmap=cmap_grad_r, fmt=dfmt, clabel=clabels_grad[args.gradient])


#REFERENCE MAP
_, _, cbar = make_polar_map(ufac*residuals, ufac*levels_resid,
                            R[args.surface]*u.m, phi[args.surface]*u.rad, Rout_plt,
                            fig=fig, ax=ax[-1],
                            Rin=Rin_plt, gradient=0,
                            cmap=cmap_res, fmt='%4d', clabel=clabels[args.moment])

for axi in ax:
    if axi!=ax[-1]:
        axi.set_xlabel(None)
        kc = {'color': '1.0'}
    else:
        kc = {}
        
    make_substructures(axi, gaps=gaps, rings=rings, twodim=True, polar=True, label_rings=True,
                       kwargs_gaps=kc, kwargs_rings=kc)

ax[0].set_title(ctitle, fontsize=16, color='k')

plt.savefig('gradient_residuals_deproj_%s_d%s_polar.png'%(mtags['base'], args.gradient), bbox_inches='tight', dpi=200)
plt.show()
plt.close()
