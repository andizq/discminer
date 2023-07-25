from discminer.plottools import (get_discminer_cmap,
                                 get_continuous_cmap,
                                 make_pie_map,
                                 make_substructures,
                                 use_discminer_style)

from utils import (init_data_and_model,
                   get_2d_plot_decorators,
                   get_noise_mask,
                   load_moments,
                   add_parser_args)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from astropy import units as u

import json
import copy
from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='plot residual maps', description='Plot residual maps')
args = add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, projection=True, smooth=True)
     
#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

Rout = best['intensity']['Rout']

gaps = custom['gaps']
rings = custom['rings']

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, unit_simple=True, fmt_vertical=True)

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model()

noise_mean, mask = get_noise_mask(datacube)
vchannels = datacube.vchannels
model.make_model()

#****************
#LOAD MOMENT MAPS
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)

#******************
#LOAD DISC GEOMETRY
ng = model.grid['nx']
R_s = model.grid['R'].reshape(ng, ng)*u.m.to('au')
phi_s = model.grid['phi'].reshape(ng, ng)

X = R_s*np.cos(phi_s)
Y = R_s*np.sin(phi_s)

#****************
#USEFUL FUNCTIONS
def readtxt(tag):
    if tag!='': tag = '_'+tag
    return np.loadtxt('radial_profile_%s%s.dat'%(args.moment, tag)).T

def make_savgol(prof):
    try:
        ysav = savgol_filter(prof, 5, 3)
        ysav_deriv = savgol_filter(prof, 5, 3, deriv=1)
    except np.linalg.LinAlgError:
        ysav = prof
        ysav_deriv = None
    return ysav, ysav_deriv 

#************************
#LOAD VELOCITY COMPONENTS
Rp, vp, dvp = readtxt('vphi')
Rr, vr, dvr = readtxt('vr')
Rz, vz, dvz = readtxt('vz')    

f_vp = interp1d(Rp, 1e3*vp)
f_vr = interp1d(Rr, 1e3*vr)
f_vz = interp1d(Rz, 1e3*vz)

try:
    vp_sav, dvp_sav = make_savgol(1e3*vp)
    f_dvp = interp1d(Rp, dvp_sav)
except np.linalg.LinAlgError:
    f_dvp = interp1d(Rp, np.zeros_like(Rp)) 

#************************
#DEFINE PIE MAP ARGUMENTS

#Maps to plot per quadrant
mask = (R_s>np.max(Rp)) | (R_s<np.min(Rp)) 
R_s[mask] = np.nan

quadrant_map2d = {
    1: f_vz(R_s),
    2: f_vr(R_s),
    3: f_vp(R_s),
    4: f_dvp(R_s)
}

#Contourf levels per quadrant
nlev = 128
vlim_u = 1e3*clim
dvlim_u = 0.25*vlim_u

levels_vel = np.linspace(-vlim_u, vlim_u, nlev)
levels_div = np.linspace(-dvlim_u, dvlim_u, nlev)    
    
quadrant_levels = {
    1: levels_vel,
    2: levels_vel,
    3: levels_vel,
    4: levels_div,
}

#cmap per quadrant
cmap_vel = get_discminer_cmap('velocity')
hex_list = ['#%s'%tmp for tmp in ["901b22","c2242d","fdffff","a3bf50","778b3b"]]
cmap_div = get_continuous_cmap(hex_list)

quadrant_cmap = {
    1: cmap_vel,
    2: cmap_vel,
    3: cmap_vel,
    4: cmap_div,
}

#colorbar label per quadrant
quadrant_clabel = {
    1: r'$\upsilon_{\rm z}$ [m/s]',
    2: r'$\upsilon_{\rm R}$ [m/s]',
    3: r'$\Delta$$\upsilon_{\phi}$ [m/s]',
    4: r'${\delta{\Delta \upsilon_{\phi}}}/{\delta{R}}$'
}

#Call function
fig, ax = make_pie_map(
    X*u.au, Y*u.au, np.max(Rp)*u.au,
    quadrant_map2d=quadrant_map2d,
    quadrant_levels=quadrant_levels,
    quadrant_cmap=quadrant_cmap,
    quadrant_clabel=quadrant_clabel,    
    gaps=gaps, rings=rings
)
make_substructures(ax, gaps=gaps, rings=rings, label_rings=True, twodim=True)

ax.set_title(ctitle, fontsize=16, color='k')
fig.savefig('pie_velocity_components_%s.png'%mtags['base'], bbox_inches='tight', dpi=200)
plt.show()
    
#MAKE 2D PLOT OF RESIDUALS-VPHI-VZ to illustrate uncertainty on vR
#SHOW 2D VELOCITY UNCERTAINTIES
