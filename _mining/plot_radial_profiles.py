from discminer.plottools import (get_discminer_cmap, make_substructures,
                                 make_up_ax, mod_minor_ticks, mod_major_ticks,
                                 use_discminer_style, mod_nticks_cbars)
from discminer.rail import Rail, Contours

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from astropy import units as u
from astropy.io import fits
import json
import copy

from utils import (init_data_and_model,
                   get_noise_mask,
                   get_1d_plot_decorators,
                   load_moments,
                   load_disc_grid,
                   add_parser_args,
                   make_1d_legend,
                   MEDIUM_SIZE)

from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='plot radial profiles', description='Plot radial profiles from moments and residuals [velocity, linewidth, [peakintensity, peakint]?')
args = add_parser_args(parser, moment=True, kind=True, surface=True, writetxt=True, mask_minor=True, mask_major=True, Rinner=True, Router=True)

#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

Mstar = best['velocity']['Mstar']
vsys = best['velocity']['vsys']
vel_sign = best['velocity']['vel_sign']
Rout = best['intensity']['Rout']
incl = best['orientation']['incl']

gaps = custom['gaps']
rings = custom['rings']
Rmax = 1.1*Rout*u.au #Max model radius, 10% larger than disc Rout

clabel, clabel_res, clim0, clim0_res, clim1, clim1_res, unit = get_1d_plot_decorators(args.moment)

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model()

noise_mean, mask = get_noise_mask(datacube, thres=2)
vchannels = datacube.vchannels
model.make_model()

#*************************
#LOAD DISC GEOMETRY
R, phi, z = load_disc_grid()

R_s = R[args.surface]*u.m.to('au')
phi_s = phi[args.surface]

#*************************
#LOAD MOMENT MAPS
moment_data, moment_model, mtags = load_moments(args)
if mtags['surf']=='both':
    surf_ref = 'upper'
else:
    surf_ref = args.surface
tag_base = mtags['base']

#**************************
#MASK AND COMPUTE RESIDUALS
moment_data = np.where(mask, np.nan, moment_data)
moment_model = np.where(mask, np.nan, moment_model)
moment_residuals = moment_data - moment_model
    
if args.moment=='velocity':
    moment_residuals_abs = np.abs(moment_data-vsys) - np.abs(moment_model-vsys)

#**************************
#MAKE PLOTS
beam_au = datacube.beam_size.to('au').value
R_prof = np.arange(args.Rinner*beam_au, args.Router*Rout, beam_au/5)
xlim0, xlim1 = 0.5*R_prof[0], 1.05*R_prof[-1]

def writetxt(arr, tag=''):
    if tag!='': tag = '_'+tag
    arr = np.asarray(arr).T
    np.savetxt('radial_profile_%s%s.dat'%(args.moment, tag), arr, fmt='%.6f', header='R[au] attribute stddev')

def get_normalisation(mask_ang, component='z'):
    """
    mask_ang : +- mask around disc minor axis
    component : velocity component to get normalisation for    
    """
    if component=='phi':
        div_factor = 4*np.sin(np.pi/2 - np.radians(mask_ang)) * np.abs(np.sin(incl))/(2*np.pi - 4*np.radians(mask_ang))
    elif component=='z':
        div_factor = -np.cos(np.abs(incl))        
    else: return comp
    
    return div_factor

def make_savgol(prof):
    try:
        ysav = savgol_filter(prof, 5, 3)
        ysav_deriv = savgol_filter(prof, 5, 3, deriv=1)
    except np.linalg.LinAlgError:
        ysav = prof
        ysav_deriv = None
    return ysav, ysav_deriv 


if args.moment=='velocity':
    mask_ang = args.mask_minor #+-mask around disc minor axis
    #*******************
    #VELOCITY COMPONENTS
    #*******************    

    #VZ
    rail_vz = Rail(model, moment_residuals, R_prof)
    vel_z, vel_z_error = rail_vz.get_average(mask_ang=mask_ang, surface=surf_ref)
    div_factor_z = get_normalisation(mask_ang, component='z')

    vel_z /= div_factor_z
    vel_z_error /= div_factor_z 

    #DVPHI
    rail_phi = Rail(model, moment_residuals_abs, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(mask_ang=mask_ang, surface=surf_ref)
    div_factor_phi = get_normalisation(mask_ang, component='phi')

    vel_phi /= div_factor_phi
    vel_phi_error /= div_factor_phi 

    #VPHI
    rail_phi = Rail(model, np.abs(moment_data-vsys), R_prof)
    vel_rot, _ = rail_phi.get_average(mask_ang=mask_ang, surface=surf_ref)
    vel_rot /= div_factor_phi
    vel_rot_error = vel_phi_error

    #VR
    mask_r = mask | (np.abs(phi_s) < np.radians(args.mask_major)) | (np.abs(phi_s) > np.radians(180-args.mask_major))
    moment_data_r = np.where(mask_r, np.nan, moment_data) 

    f_vp = interp1d(R_prof, vel_rot)
    f_vz = interp1d(R_prof, vel_z)
    R_interp = np.where((R_s>R_prof[0]) & (R_s<R_prof[-1]), R_s, R_prof[0])

    vr = -1/(np.sin(phi_s)*np.sin(incl)) * (moment_data_r - vsys - vel_sign*f_vp(R_interp)*np.cos(phi_s)*np.sin(incl) + f_vz(R_interp)*np.cos(incl))

    rail_vr = Rail(model, vr, R_prof)
    vel_r, vel_r_error = rail_vr.get_average(mask_ang=0.0, surface=surf_ref, av_func=np.nanmedian)

    #WRITE?
    if args.writetxt: writetxt([R_prof, vel_z, vel_z_error], tag='vz')
    if args.writetxt: writetxt([R_prof, vel_r, vel_r_error], tag='vr')
    if args.writetxt: writetxt([R_prof, vel_phi, vel_phi_error], tag='vphi')
    if args.writetxt: writetxt([R_prof, vel_rot, vel_phi_error], tag='rotationcurve')
    
    #PLOT 3D VELOCITIES
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,4))
        
    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)
    ax.plot(R_prof, ysav_phi, c='dodgerblue', lw=3, label=r'$\Delta\upsilon_\phi$', zorder=12)
    ax.fill_between(R_prof, vel_phi+vel_phi_error, vel_phi-vel_phi_error, color='dodgerblue', alpha=0.15, zorder=9)

    ysav_z, ysav_z_deriv = make_savgol(vel_z)    
    ax.plot(R_prof, ysav_z, c='k', lw=3, label=r'$\upsilon_{\rm z}$', zorder=8)
    ax.fill_between(R_prof, vel_z+vel_z_error, vel_z-vel_z_error, color='k', alpha=0.15, zorder=8)
        
    ysav_r, ysav_r_deriv = make_savgol(vel_r)
    ax.plot(R_prof, ysav_r, c='#FFB000', lw=3, label=r'$\upsilon_{\rm R}$', zorder=7)
    ax.fill_between(R_prof, vel_r+vel_r_error, vel_r-vel_r_error, color='#FFB000', alpha=0.15, zorder=7)

    ax.axhline(0, lw=2, ls='--', color='0.7')
    
    ax.set_xlabel('Radius [au]')
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylabel(r'$\delta\upsilon$ [km/s]')
    ax.set_ylim(clim0_res, clim1_res)
    mod_major_ticks(ax, axis='x', nbins=10)
    mod_minor_ticks(ax)
    make_1d_legend(ax, fontsize=MEDIUM_SIZE+1)
    
    make_substructures(ax, gaps=gaps, rings=rings, label_gaps=True, label_rings=True)
    
    plt.savefig('velocity_components_%s.png'%tag_base, bbox_inches='tight', dpi=200)
    plt.show()

    #*******************
    #ROTATION CURVE
    #*******************
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,4))        

    #DATA CURVE
    ysav_rot, ysav_rot_deriv = make_savgol(vel_rot)
    ax.plot(R_prof, ysav_rot, c='tomato', lw=3.5, label=r'Data', zorder=12)
    ax.fill_between(R_prof, vel_rot+vel_rot_error, vel_rot-vel_rot_error, color='tomato', alpha=0.15, zorder=9)
    
    #r"""
    #MODEL CURVE
    rail_phi = Rail(model, np.abs(moment_model-vsys), R_prof)
    vel_phi, _ = rail_phi.get_average(mask_ang=mask_ang, surface=surf_ref)
    vel_phi /= div_factor_phi
    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)
    ax.plot(R_prof, ysav_phi, c='0.7', lw=4.0, label=r'Model', zorder=11)
    #"""
    
    #PERFECT KEPLERIAN
    coords = {'R': R_prof*u.au.to('m')}
    velocity_upper = model.get_attribute_map(coords, 'velocity', surface=surf_ref) * vel_sign
    ax.plot(R_prof, velocity_upper, c='k', lw=2.5, ls='--', label=r'Keplerian (%.2f Msun)'%Mstar, zorder=13)

    #DECORATIONS
    ax.axhline(0, lw=2, ls='--', color='0.7')

    ax.set_xlabel('Radius [au]')
    ax.set_xlim(xlim0, xlim1)    
    ax.set_ylabel(r'Rotation velocity [km/s]')
    ax.set_ylim(clim0, 1.2*np.nanmax(vel_phi))
    mod_major_ticks(ax, axis='x', nbins=10)
    mod_minor_ticks(ax)    
    make_1d_legend(ax)

    make_substructures(ax, gaps=gaps, rings=rings, label_gaps=True, label_rings=True)  
    
    plt.savefig('rotation_curve_%s.png'%tag_base, bbox_inches='tight', dpi=200)
    plt.show()

    """
    #Data rotation curve - perfect_Keplerian
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,4)) 
    ax.plot(R_prof, vel_rot-velocity_upper, c='k', lw=2.5, ls='--', label=r'Keplerian (%.2f Msun)'%Mstar, zorder=13)
    plt.show()
    """
    
else:
    mask_ang = 0.0
    #*****************
    #ABSOLUTE PROFILES
    #*****************
    kw_avg = dict(surface=surf_ref, av_func=np.nanmedian, mask_ang=mask_ang)
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,4))

    #DATA CURVE
    rail_phi = Rail(model, moment_data, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(**kw_avg)

    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)    
    ax.plot(R_prof, ysav_phi, c='tomato', lw=4.0, label=r'Data', zorder=12)
    ax.fill_between(R_prof, vel_phi+vel_phi_error, vel_phi-vel_phi_error, color='tomato', alpha=0.15, zorder=9)

    if args.writetxt: writetxt([R_prof, vel_phi, vel_phi_error], tag='data')

    #MODEL CURVE
    rail_phi = Rail(model, moment_model, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(**kw_avg)

    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)    
    ax.plot(R_prof, ysav_phi, c='dodgerblue', lw=3.5, label=r'Model', zorder=11)

    if args.writetxt: writetxt([R_prof, vel_phi, vel_phi_error], tag='model')
    
    #DECORATIONS
    ax.axhline(0, lw=2, ls='--', color='0.7')

    ax.set_xlabel('Radius [au]')
    ax.set_xlim(xlim0, xlim1)    
    ax.set_ylabel(clabel)
    ax.set_ylim(clim0, clim1)
    mod_major_ticks(ax, axis='x', nbins=10)
    mod_minor_ticks(ax)
    make_1d_legend(ax)

    make_substructures(ax, gaps=gaps, rings=rings, label_gaps=True, label_rings=True)  
    
    plt.savefig('radial_profile_%s.png'%tag_base, bbox_inches='tight', dpi=200)
    plt.show()

    #****************
    #RESIDUALS
    #****************    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,4))

    rail_phi = Rail(model, moment_residuals, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(**kw_avg)

    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)    
    ax.plot(R_prof, ysav_phi, c='tomato', lw=4.0, label=r'Residuals', zorder=12)
    ax.fill_between(R_prof, vel_phi+vel_phi_error, vel_phi-vel_phi_error, color='tomato', alpha=0.15, zorder=9)

    if args.writetxt: writetxt([R_prof, vel_phi, vel_phi_error], tag='residuals')    

    #DECORATIONS
    ax.axhline(0, lw=2, ls='--', color='0.7')

    ax.set_xlabel('Radius [au]')
    ax.set_xlim(xlim0, xlim1)    
    ax.set_ylabel(clabel_res)
    ax.set_ylim(clim0_res, clim1_res)
    mod_major_ticks(ax, axis='x', nbins=10)
    mod_minor_ticks(ax)

    make_substructures(ax, gaps=gaps, rings=rings, label_gaps=True, label_rings=True)
        
    plt.savefig('radial_profile_residuals_%s.png'%tag_base, bbox_inches='tight', dpi=200)
    plt.show()
    

