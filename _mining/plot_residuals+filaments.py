from discminer.core import Data
import discminer.cart as cart
from discminer.plottools import (make_round_map,
                                 make_polar_map,
                                 make_substructures,
                                 make_up_ax,
                                 mod_minor_ticks,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 use_discminer_style)

from utils import (make_and_save_filaments,
                   init_data_and_model,
                   get_2d_plot_decorators,
                   get_noise_mask,
                   load_moments,
                   load_disc_grid,
                   add_parser_args)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

import json
from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='plot residual maps', description='Plot residual maps')
parser.add_argument('-f', '--filaments', default=1, type=int, choices=[0, 1], help="Make filaments")
args = add_parser_args(parser, moment=True, kind=True, surface=True, projection=True, Rinner=True, Router=True)

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

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.1*Rout*u.au #Max model radius, 10% larger than disc Rout

#********************
#LOAD DATA AND GRID
#********************
datacube = Data(file_data, dpc) # Read data and convert to Cube object
noise_mean, mask = get_noise_mask(datacube, thres=2)

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.15*np.min([xmax, Rmax.value])
extent= np.array([-xmax, xmax, -xmax, xmax])

#*************************
#LOAD DISC GEOMETRY
R, phi, z = load_disc_grid()
RR = R[args.surface]/au_to_m
PP = phi[args.surface]

Xproj = RR*np.cos(phi[args.surface])
Yproj = RR*np.sin(phi[args.surface])

#*************************
#LOAD AND CLIP MOMENT MAPS    
moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask,
    clip_Rmin=0.0*u.au,
    clip_Rmax=args.Router*Rout*u.au,
    clip_Rgrid=R[args.surface]*u.m
)

#******************************
#INIT MODEL AND MAKE FILAMENTS
#******************************
_, model = init_data_and_model()
model.make_model()
fil_pos_list, fil_neg_list, colors_dict = make_and_save_filaments(model, residuals, tag=mtags['base']+'_'+args.projection, return_all=True)
fil_pos = fil_pos_list[0]
fil_neg = fil_neg_list[0]

#***********
#MAKE PLOTS
clabels = {
    'linewidth': r'$\Delta$ Line width [km s$^{-1}$]',
    'lineslope': r'$\Delta$ Line slope',
    'velocity': r'$\Delta$ Centroid [km s$^{-1}$]',
    'peakintensity': r'$\Delta$ Peak Int. [K]'
}


fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(14, 7))

axr = fig.add_axes([0.55, 0.1, 0.4, 0.8])
ax[0,1].set_visible(False)
ax[1,1].set_visible(False)

axf = ax[0,0] #fit in polar coords
axp = ax[1,0] #pitch angle

#FIT SPIRALS
sp_lin = lambda x, a, b: a + b*x
sp_log = lambda x, a, b: a*np.exp(b*x)

def make_savgol(prof):
    tmp = len(prof)
    wl = tmp if tmp%2 else tmp-1

    try:
        ysav = savgol_filter(prof, wl, 2)
        ysav_deriv = savgol_filter(prof, wl, 2, deriv=1)
    except np.linalg.LinAlgError:
        ysav = prof
        ysav_deriv = None

    return ysav, ysav_deriv

def clean_filament_xy(xp, yp):
    ind = np.argsort(xp)
    xp = xp[ind]
    yp = yp[ind]
    dx = np.abs(xp[1:] - xp[:-1])
    ind2 = np.append([True], [dx > 0.5*np.median(dx)])
    xp = xp[ind2]
    yp = yp[ind2]
    dx = np.abs(xp[1:] - xp[:-1])
    dy = np.abs(yp[1:] - yp[:-1])
    
    #spiral possibly split across phi axis boundary
    # Fails if spiral crosses axis boundary multiple times:
    #  See e.g. mwc758 12co peakint. 
    indc = dx > 10*np.median(dx)
    indj = dy > 10*np.median(dy)

    def shift(xp, ind):
        indp = np.arange(1, len(xp))[ind]
        for i in indp:
            left = xp[:i]
            right = xp[i:]
            #print(len(left),len(right))
            if len(left)<=len(right):
                xp[:i] += 360
            else:
                xp[i:] -= 360
        return xp
    
    if np.sum(indc):
        xp = shift(xp, indc)
        
    # Attempt to do the same using dy
    #elif np.sum(indj):
    #    xp = shift(xp, indj)        
        
    ind3 = np.argsort(xp)
    xp = xp[ind3]
    yp = yp[ind3]
    
    return xp, yp


def fit_and_plot(ax, xp, yp, color=None, kind='positive'):
    if kind in ['pos', 'positive']:
        if color is None:
            color = 'r'
        lin_c = 'magenta'
        log_c = 'k'
        sav_c = 'tomato'
    elif kind in ['neg', 'negative']:
        if color is None:
            color = 'b'
        lin_c = 'cyan'
        log_c = 'k'
        sav_c = 'dodgerblue'
    else:
        raise ValueError
   
    ax.scatter(xp, yp, s=20, color=color)

    popt, pcov = curve_fit(sp_lin, xp, yp) #, sigma=20*np.ones_like(yn))
    #ax.scatter(xp, sp_lin(xp, *popt), fc=lin_c, marker='s', lw=1, s=30)

    xlin = np.linspace(xp.min(), xp.max(), 50)
    ylin = sp_lin(xp, *popt)

    popt, pcov = curve_fit(sp_log, xp, yp, p0=[100, 0]) #, p0=[100, -0.1])
    ax.scatter(xp, sp_log(xp, *popt), c=log_c, marker='+', lw=1, s=30)

    ysav, dy = make_savgol(yp)
    ax.plot(xp, ysav, c=sav_c, ls='-', lw=2)

    dylin = np.gradient(ylin, np.radians(xp))

    return ylin, dylin


for i,fil in enumerate(fil_pos_list[2:]):    
    fp = fil.astype(bool)
    xp = np.degrees(PP[fp])
    yp = RR[fp]
    xp, yp = clean_filament_xy(xp, yp)
    yfp, dyp = fit_and_plot(axf, xp, yp, color=colors_dict[i+1])
    pitchp = np.degrees(np.arctan(np.abs(dyp)/yfp))
    axp.plot(yfp, pitchp, lw=3, c=colors_dict[i+1])

for i,fil in enumerate(fil_neg_list[2:]):    
    fn = fil.astype(bool)
    xn = np.degrees(PP[fn])
    yn = RR[fn]
    xn, yn = clean_filament_xy(xn, yn)
    yfn, dyn = fit_and_plot(axf, xn, yn, color=colors_dict[-i-1], kind='neg')
    pitchn = np.degrees(np.arctan(np.abs(dyn)/yfn))
    axp.plot(yfn, pitchn, lw=3, c=colors_dict[-i-1])
    
axf.set_xlim(-270, 270)
axf.set_ylim(0, None)
axf.set_ylim(args.Rinner*datacube.beam_size.value, args.Router*Rout)
axf.set_xlabel('Azimuth [deg]', labelpad=5)
axf.set_ylabel('Radius [au]')
axf.set_xticks(np.arange(-270,270+1,90))
axf.axvline(-90, ls=':', lw=2.5, color='0.3', dash_capstyle='round')
axf.axvline(90, ls=':', lw=2.5, color='0.3', dash_capstyle='round')
mod_major_ticks(axp, axis='y', nbins=5)

axp.set_xlim(args.Rinner*datacube.beam_size.value, args.Router*Rout)
axp.set_ylim(0, None)
axp.set_xlabel('Radius [au]', labelpad=5)
axp.set_ylabel('Pitch angle [deg]')
make_substructures(axp, gaps=gaps, rings=rings, label_gaps=True, label_rings=True)  
mod_major_ticks(axp, axis='x', nbins=10)
mod_major_ticks(axp, axis='y', nbins=5)

#RIGHT PANEL
   
if args.projection=='cartesian':
    levels_resid = np.linspace(-clim, clim, 32)
    
    if args.surface in ['up', 'upper']:
        z_func = cart.z_upper_exp_tapered
        z_pars = best['height_upper']

    elif args.surface in ['low', 'lower']:
        z_func = cart.z_lower_exp_tapered
        z_pars = best['height_lower']
    
    fig, axr = make_round_map(residuals, levels_resid, Xproj*u.au, Yproj*u.au, args.Router*Rout*u.au,
                              fig=fig, ax=axr, make_cbar=False,
                              z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                              cmap=cmap_res, clabel=unit, fmt=cfmt, 
                              gaps=gaps, rings=rings,
                              mask_inner=args.Router*Rout*u.au, kwargs_mask={'facecolor': '0.9'})
    
    make_substructures(axr, gaps=gaps, rings=rings, twodim=True, label_rings=True)

    for i,fil in enumerate(fil_pos_list[2:]):
        axr.contour(Xproj, Yproj, fil, linewidths=1.0, colors=colors_dict[i+1], alpha=1, zorder=13)
    for i,fil in enumerate(fil_neg_list[2:]):        
        axr.contour(Xproj, Yproj, fil, linewidths=1.0, colors=colors_dict[-i-1], alpha=1, zorder=13)

            
axr.set_title(ctitle, fontsize=16, color='k')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

plt.savefig('residuals_filaments_%s_%s.png'%(mtags['base'], args.projection), bbox_inches='tight', dpi=200)
plt.show()
plt.close()
