from discminer.plottools import (get_discminer_cmap, make_substructures,
                                 make_up_ax, mod_minor_ticks, mod_major_ticks,
                                 use_discminer_style, mod_nticks_cbars)
from discminer.rail import Rail, Contours

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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

parser = ArgumentParser(prog='plot radial profiles', description='Compute radial profiles from moment maps and residuals.')
args = add_parser_args(parser, moment=True, kernel=True, kind=True, surface=True, writetxt=True, mask_minor=True, mask_major=True, Rinner=True, Router=True, sigma=True, smooth=True)

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
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)
ref_surf = mtags['ref_surf']
tag_base = mtags['base']

#**************************
#ABSOLUTE RESIDUALS    
if args.moment=='velocity':
    residuals_abs = np.abs(moment_data-vsys) - np.abs(moment_model-vsys)

#**************************
#MAKE PLOTS
beam_au = datacube.beam_size.to('au').value
R_prof = np.arange(args.Rinner*beam_au, args.Router*Rout, beam_au/4.0) #changed to beam/4 from beam/5 before
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
        ysav = savgol_filter(prof, 9, 3)
        ysav_deriv = savgol_filter(prof, 9, 3, deriv=1)
    except np.linalg.LinAlgError:
        ysav = prof
        ysav_deriv = None
    return ysav, ysav_deriv 

def make_basic_layout(ax):
    ax.set_xlim(xlim0, xlim1)
    ax.set_xlabel('Radius [au]', fontsize=MEDIUM_SIZE-1)
    mod_major_ticks(ax, axis='x', nbins=10)
    mod_minor_ticks(ax)
    ax.axhline(0, lw=2, ls='--', color='0.7')

def make_profile(ax, R_prof, ysav, y, yerr, kind='data', perr='bar', **kwargs):
    if kind=='data':
        label = 'Data'
        color = 'tomato'
        zorder = 9
        lw = 3.5        
    elif kind=='residuals':
        label = 'Residuals'
        color = '0.1'
        zorder = 9
        lw = 3.5
    elif kind=='model':
        label = 'Model'
        color = 'dodgerblue'
        zorder = 8
        lw = 3.5
    elif kind=='vphi':
        label = r'$\Delta\upsilon_\phi$'
        color = 'dodgerblue'
        zorder = 9
        lw = 3.0
    elif kind=='vz':
        label = r'$\upsilon_{\rm z}$'
        color = 'k'
        zorder = 8
        lw = 3.0
    elif kind=='vr':
        label = r'$\upsilon_{\rm R}$'
        color = '#FFB000'
        zorder = 7
        lw = 3.0
    else:
        raise ValueError(kind)

    ax.plot(R_prof, ysav, c=color, lw=lw, label=label, zorder=zorder)

    if kind!='model':
        if perr=='bar':
            ax.errorbar(R_prof, y, yerr=yerr, c=color,
                        elinewidth=1.2, capsize=1.8, linestyle='none',
                        marker='o', ms=6.5, markeredgewidth=1.7, markerfacecolor='0.8',
                        alpha=0.7, zorder=zorder)
        elif perr=='fill':
            ax.fill_between(R_prof, y+yerr, y-yerr, color=color, alpha=0.15, zorder=zorder)
        else:
            raise ValueError(perr)
        
    if args.writetxt: writetxt([R_prof, y, yerr], tag=kind)

    
#*************
#MAIN BODY
#*************
kw_ylabel = dict(fontsize=MEDIUM_SIZE-2, labelpad=0)

if args.moment=='velocity':

    mask_ang = args.mask_minor #+-mask around disc minor axis

    #*******************
    #VELOCITY COMPONENTS
    #*******************    

    """    
    if args.surface in ['low', 'lower']:
        mask_r = mask | (np.abs(phi_s) < np.radians(args.mask_major)) | (np.abs(phi_s) > np.radians(180-args.mask_major))
        moment_data = np.where(mask_r, np.nan, moment_data)
        if args.moment=='velocity':
            pass
            moment_data = np.where(np.abs(moment_data-vsys)>5, np.nan, moment_data)    
    """
    if args.kernel in ['doublebell', 'doublegaussian']:
        mask_map_ref = np.abs((moment_data-vsys)/(np.cos(model.projected_coords['phi'][args.surface])*np.sin(incl)))
    else:
        mask_map_ref = None
        
    kw_avg = dict(surface=ref_surf, av_func=np.nanmean, mask_ang=mask_ang, sigma_thres=args.sigma, mask_from_map=mask_map_ref)
    
    #VZ
    rail_vz = Rail(model, residuals, R_prof)
    vel_z, vel_z_error = rail_vz.get_average(tag='vz', **kw_avg)
    div_factor_z = get_normalisation(mask_ang, component='z')

    vel_z /= div_factor_z
    vel_z_error /= div_factor_z 

    #DVPHI
    rail_phi = Rail(model, residuals_abs, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(**kw_avg)
    div_factor_phi = get_normalisation(mask_ang, component='phi')

    vel_phi /= div_factor_phi
    vel_phi_error /= div_factor_phi 

    #VPHI
    rail_phi = Rail(model, np.abs(moment_data-vsys), R_prof)
    vel_rot, vel_rot_error = rail_phi.get_average(tag=tag_base+'_vphi_data', plot_diagnostics=True, forward_error=True, **kw_avg)
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
    vel_r, vel_r_error = rail_vr.get_average(mask_ang=0.0, surface=ref_surf, av_func=np.nanmedian)
    
    #SAVE 3D VELOCITIES IN ONE FILE
    header = 'radius[au] v_rot[m/s] v_phi[m/s] dv_phi_rot[m/s] v_r[m/s] dv_r[m/s] v_z[m/s] dv_z[m/s]'
    vel_arr = [R_prof, vel_rot, vel_phi, vel_phi_error, vel_r, vel_r_error, vel_z, vel_z_error]
    vel_arr_t = np.asarray(vel_arr).T

    np.savetxt('%s_%s_%s_radial_profile_3D_velocities.txt'%(meta['disc'], meta['mol'], args.kernel), vel_arr_t, fmt='%.6f', 
               header=header, footer=str(meta['log_file']) + ' --mask_minor ' + str(args.mask_minor) + ' --mask_major ' + str(args.mask_major) + ' --Rinner ' + str(args.Rinner) + ' --Router ' + str(args.Router))

    #PLOT 3D VELOCITIES
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11,4))
        
    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)
    make_profile(ax, R_prof, ysav_phi, vel_phi, vel_phi_error, kind='vphi', perr='fill')
    ax.scatter(R_prof, vel_phi, ec='dodgerblue', fc='none', s=25, zorder=12)
    
    ysav_z, ysav_z_deriv = make_savgol(vel_z)    
    make_profile(ax, R_prof, ysav_z, vel_z, vel_z_error, kind='vz', perr='fill')
    ax.scatter(R_prof, vel_z, ec='k', fc='none', s=25, zorder=11)
    
    ysav_r, ysav_r_deriv = make_savgol(vel_r)
    make_profile(ax, R_prof, ysav_r, vel_r, vel_r_error, kind='vr', perr='fill')    
    ax.scatter(R_prof, vel_r, ec='#FFB000', fc='none', s=25, zorder=10)
    
    #DECORATIONS
    make_basic_layout(ax)
    ax.set_ylabel(r'$\delta\upsilon$ [km/s]', fontsize=MEDIUM_SIZE, labelpad=10)
    ax.set_ylim(clim0_res, clim1_res)
    make_1d_legend(ax, fontsize=MEDIUM_SIZE+1)    
    make_substructures(ax, gaps=gaps, rings=rings, label_gaps=True, label_rings=True)
    
    plt.savefig('velocity_components_%s.png'%tag_base, bbox_inches='tight', dpi=200)
    plt.show()

    #*******************
    #ROTATION CURVE
    #*******************
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11,4))        

    #DATA CURVE
    ysav_rot, ysav_rot_deriv = make_savgol(vel_rot)
    make_profile(ax, R_prof, ysav_rot, vel_rot, vel_rot_error, kind='data')

    #MODEL CURVE
    rail_phi = Rail(model, np.abs(moment_model-vsys), R_prof)
    #vel_phi, _ = rail_phi.get_average(mask_ang=mask_ang, surface=ref_surf)
    vel_phi, _ = rail_phi.get_average(surface=ref_surf, av_func=np.nanmean, mask_ang=mask_ang, sigma_thres=np.inf,
                                      mask_from_map=mask_map_ref, tag=tag_base+'_vphi_model', plot_diagnostics=True)
    div_factor_model = get_normalisation(mask_ang, component='phi')
    vel_phi /= div_factor_model
    ysav_phi, ysav_phi_deriv_mod = make_savgol(vel_phi)
    make_profile(ax, R_prof, ysav_phi, vel_phi, _, kind='model')
    
    #PERFECT KEPLERIAN
    coords = {'R': R_prof*u.au.to('m')}
    velocity_upper = model.get_attribute_map(coords, 'velocity', surface=ref_surf) * vel_sign
    ax.plot(R_prof, velocity_upper, c='k', lw=2.5, ls='--', label=r'Keplerian (%.2f Msun)'%Mstar, zorder=13)

    #DECORATIONS
    make_basic_layout(ax)
    ax.set_ylabel(r'Rotation velocity [km/s]', fontsize=MEDIUM_SIZE, labelpad=10)
    ax.set_ylim(clim0, 1.2*np.nanmax(vel_phi))
    make_1d_legend(ax, fontsize=MEDIUM_SIZE-3)
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

    mask_ang = args.mask_minor
    
    if args.surface in ['low', 'lower']:
        mask_r = mask | (np.abs(phi_s) < np.radians(args.mask_major)) | (np.abs(phi_s) > np.radians(180-args.mask_major)) | (moment_data>40)
        moment_data = np.where(mask_r, np.nan, moment_data)

    if args.kernel in ['doublebell', 'doublegaussian']:
        mask_map_ref = moment_data
    else:
        mask_map_ref = None
        
    #*****************
    #ABSOLUTE PROFILES
    #*****************
    kw_avg = dict(surface=ref_surf, av_func=np.nanmedian, mask_ang=mask_ang, sigma_thres=args.sigma, mask_from_map=mask_map_ref, plot_diagnostics=True)
        
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(11,6))

    #DATA CURVE
    rail_phi = Rail(model, moment_data, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(**kw_avg)

    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)    
    make_profile(ax[0], R_prof, ysav_phi, vel_phi, vel_phi_error, kind='data')    

    #MODEL CURVE
    rail_phi = Rail(model, moment_model, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(**kw_avg)

    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)
    make_profile(ax[0], R_prof, ysav_phi, vel_phi, vel_phi_error, kind='model')        

    #RESIDUALS
    rail_phi = Rail(model, residuals, R_prof)
    vel_phi, vel_phi_error = rail_phi.get_average(**kw_avg)

    ysav_phi, ysav_phi_deriv = make_savgol(vel_phi)
    make_profile(ax[1], R_prof, ysav_phi, vel_phi, vel_phi_error, kind='residuals')

    #***********
    #DECORATIONS
    #***********
    ax[0].set_ylim(clim0, clim1)
    ax[1].set_ylim(clim0_res, clim1_res)

    #fmt (Homogenise fmt for both panels)
    #******    
    ticks = ax[1].get_yticks()

    makeint = False
    for tick in ticks:
        if str(tick)[-2:] != '.0':
            break
        else:
            makeint = True

    #ftick = round(ticks[0], 2)
    ftick = round(clim0_res, 2)
    ftick = int(ftick) if makeint else ftick
    ftick_res = str(ftick)
    
    isfloat = isinstance(ftick, float)
    #lfmt = len(str(ftick))
    lfmt = 5
    
    if isfloat:
        if abs(ftick)<0.2 and lfmt<5: #e.g. -0.1
            ftick_res += '0'
            lfmt+=1
        
        ndec = len(str(ftick).split('.')[-1])
        ndec_res = len(str(ftick_res).split('.')[-1])
        
        cfmt = '%'+'%d.%df'%(lfmt+ndec, ndec)
        cfmt_res = '%'+'%d.%df'%(lfmt+ndec_res, ndec_res)
        print (ndec_res, ftick_res)
    else:
        cfmt = cfmt_res = '%'+'%dd'%lfmt
    
    #******
    #Axes
    #****
    for axi in ax:
        make_basic_layout(axi)
        axi.tick_params(which='both', labelsize=MEDIUM_SIZE-3)

    ax[0].yaxis.set_major_formatter(FormatStrFormatter(cfmt))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter(cfmt_res))            
    ax[0].set_xlabel(None)
    ax[1].set_xlabel('Radius [au]', fontsize=MEDIUM_SIZE-2)
    ax[0].set_ylabel(clabel, **kw_ylabel)
    ax[1].set_ylabel('Residuals', **kw_ylabel) #+clabel_res.split('residuals')[-1] #--> unit

    make_1d_legend(ax[0], ncol=2)

    make_substructures(ax[0], gaps=gaps, rings=rings, label_gaps=True, label_rings=True)  
    make_substructures(ax[1], gaps=gaps, rings=rings)
    
    plt.savefig('radial_profile_%s.png'%tag_base, bbox_inches='tight', dpi=200)
    plt.show()
