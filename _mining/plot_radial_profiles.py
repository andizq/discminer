from discminer.core import Data
from discminer.plottools import (get_discminer_cmap,
                                 make_up_ax, mod_minor_ticks, mod_major_ticks,
                                 use_discminer_style, mod_nticks_cbars)
from discminer.rail import Rail, Contours
from discminer.disc2d import General2d
import discminer.cart as cart

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from astropy import units as u
from astropy.io import fits
import json
import copy

from utils import get_1d_plot_decorators, load_moments, add_parser_args
from argparse import ArgumentParser

use_discminer_style()

parser = ArgumentParser(prog='plot radial profiles', description='Plot radial profiles from moments and residuals [velocity, linewidth, [peakintensity, peakint]?')
args = add_parser_args(parser, moment=True, kind=True, surface=True, writetxt=True, mask_ang=True)

if args.moment=='peakint':
     args.moment = 'peakintensity'

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
incl = best['orientation']['incl']

gaps = custom['gaps']
rings = custom['rings']
kinks = custom['kinks']

clabel, clabel_res, clim0, clim0_res, clim1, clim1_res, unit = get_1d_plot_decorators(args.moment)

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.1*best['intensity']['Rout']*u.au #Max model radius, 10% larger than disc Rout

#********
#LOAD DATA
#********
datacube = Data(file_data, dpc) # Read data and convert to Cube object
noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0)
mask = np.max(datacube.data, axis=0) < 4*np.mean(noise)
vchannels = datacube.vchannels

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = General2d(datacube, Rmax, Rmin=0, prototype=True)

model.z_upper_func = cart.z_upper_exp_tapered
model.z_lower_func = cart.z_lower_exp_tapered
model.velocity_func = model.keplerian_vertical # vrot = sqrt(GM/r**3)*R
model.line_profile = model.line_profile_bell

if 'I2pwl' in meta['kind']:
    model.intensity_func = cart.intensity_powerlaw_rbreak
elif 'I2pwlnosurf' in meta['kind']:
    model.intensity_func = cart.intensity_powerlaw_rbreak_nosurf    
else:
    model.intensity_func = cart.intensity_powerlaw_rout

#**************
#PROTOTYPE PARS
#**************
model.params = copy.copy(best)
model.params['intensity']['I0'] /= meta['downsamp_factor']

#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
model.make_model()

#*************************
#LOAD MOMENT MAPS
moment_data, moment_model, mtags = load_moments(args)
if mtags['surf']=='both':
    surf_ref = 'upper'
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
R_prof = np.arange(1*beam_au, 0.9*best['intensity']['Rout'], beam_au/4)
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
    mask_ang = args.mask_ang #+-mask around disc minor axis
    #*******************
    #VELOCITY COMPONENTS
    #*******************    
    rail_vz = Rail(model, moment_residuals, R_prof)
    vel_z, vel_z_error = rail_vz.get_average(mask_ang=mask_ang, surface=surf_ref)
    div_factor_z = get_normalisation(mask_ang, component='z')

    vel_z /= div_factor_z
    vel_z_error /= div_factor_z 
    
    rail_phi = Rail(model, moment_residuals_abs, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(mask_ang=mask_ang, surface=surf_ref)
    div_factor_phi = get_normalisation(mask_ang, component='phi')

    vel_phi /= div_factor_phi
    vel_phi_error /= div_factor_phi 

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,4))
    ysav_z, ysav_z_deriv = make_savgol(vel_z)
        
    ax.plot(R_prof, ysav_z, c='k', lw=3, label=r'$z$-component', zorder=12)
    ax.fill_between(R_prof, vel_z+vel_z_error, vel_z-vel_z_error, color='k', alpha=0.15, zorder=9)

    if args.writetxt: writetxt([R_prof, vel_z, vel_z_error], tag='vz')
    if args.writetxt: writetxt([R_prof, vel_phi, vel_phi_error], tag='vphi')
    
    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)
    ax.plot(R_prof, ysav_phi, c='dodgerblue', lw=3, label=r'$\phi$-component', zorder=12)
    ax.fill_between(R_prof, vel_phi+vel_phi_error, vel_phi-vel_phi_error, color='dodgerblue', alpha=0.15, zorder=9)

    ax.axhline(0, lw=2, ls='--', color='0.7')
    
    ax.set_xlabel('Radius [au]')
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylabel(r'$\delta\upsilon$ [km/s]')
    ax.set_ylim(clim0_res, clim1_res)
    mod_major_ticks(ax, axis='x', nbins=10)
    mod_minor_ticks(ax)
    ax.legend(frameon=False, fontsize=12)

    Contours.make_substructures(ax, gaps=gaps, rings=rings, kinks=kinks)
    
    plt.savefig('velocity_components_%s.png'%tag_base, bbox_inches='tight', dpi=200)
    plt.show()

    #*******************
    #ROTATION CURVE
    #*******************
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,4))
    
    #DATA CURVE
    rail_phi = Rail(model, np.abs(moment_data-vsys), R_prof)
    vel_rot, _ = rail_phi.get_average(mask_ang=mask_ang, surface=surf_ref)
    vel_rot /= div_factor_phi

    ysav_rot, ysav_rot_deriv = make_savgol(vel_rot)
    ax.plot(R_prof, ysav_rot, c='tomato', lw=3.5, label=r'Data curve', zorder=12)
    ax.fill_between(R_prof, vel_rot+vel_phi_error, vel_rot-vel_phi_error, color='tomato', alpha=0.15, zorder=9)

    if args.writetxt: writetxt([R_prof, vel_rot, vel_phi_error], tag='rotationcurve')    

    #r"""
    #MODEL CURVE
    rail_phi = Rail(model, np.abs(moment_model-vsys), R_prof)
    vel_phi, _ = rail_phi.get_average(mask_ang=mask_ang, surface=surf_ref)
    vel_phi /= div_factor_phi
    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)
    ax.plot(R_prof, ysav_phi, c='0.7', lw=4.0, label=r'Model curve', zorder=11)
    #"""
    
    #PERFECT KEPLERIAN
    coords = {'R': R_prof*au_to_m}
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
    ax.legend(frameon=False, fontsize=12)

    Contours.make_substructures(ax, gaps=gaps, rings=rings, kinks=kinks)
    
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
    ax.plot(R_prof, ysav_phi, c='tomato', lw=4.0, label=r'Data curve', zorder=12)
    ax.fill_between(R_prof, vel_phi+vel_phi_error, vel_phi-vel_phi_error, color='tomato', alpha=0.15, zorder=9)

    if args.writetxt: writetxt([R_prof, vel_phi, vel_phi_error], tag='data')

    #MODEL CURVE
    rail_phi = Rail(model, moment_model, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(**kw_avg)

    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)    
    ax.plot(R_prof, ysav_phi, c='dodgerblue', lw=3.5, label=r'Model curve', zorder=11)

    if args.writetxt: writetxt([R_prof, vel_phi, vel_phi_error], tag='model')
    
    #DECORATIONS
    ax.axhline(0, lw=2, ls='--', color='0.7')

    ax.set_xlabel('Radius [au]')
    ax.set_xlim(xlim0, xlim1)    
    ax.set_ylabel(clabel)
    ax.set_ylim(clim0, clim1)
    mod_major_ticks(ax, axis='x', nbins=10)
    mod_minor_ticks(ax)
    ax.legend(frameon=False, fontsize=12)

    Contours.make_substructures(ax, gaps=gaps, rings=rings, kinks=kinks)
    
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

    Contours.make_substructures(ax, gaps=gaps, rings=rings, kinks=kinks)
        
    plt.savefig('radial_profile_residuals_%s.png'%tag_base, bbox_inches='tight', dpi=200)
    plt.show()
    

