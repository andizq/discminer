from discminer.mining_control import _mining_spectra
from discminer.mining_utils import (get_noise_mask,
                                    get_2d_plot_decorators,
                                    init_data_and_model,
                                    load_moments,
                                    show_output)

from discminer.core import Data
from discminer.cube import Cube
from discminer.grid import GridTools
from discminer.tools import fit_kernel
from discminer.plottools import (make_up_ax,
                                 mod_minor_ticks,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 use_discminer_style)


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy import units as u
from astropy.io import fits

import json
import copy
from argparse import ArgumentParser

use_discminer_style()
matplotlib.rcParams['hatch.linewidth'] = 1.3

if __name__ == '__main__':
    parser = _mining_spectra(None)
    args = parser.parse_args()

#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']
Rout = best['intensity']['Rout']

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment)

"""
chot = '0.5'
cmap_mom.set_under(chot)
cmap_mom.set_over(chot)

cmap_res.set_under(chot)
cmap_res.set_over(chot)
"""

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc

mol_tex = {
    '12co': r'$^{12}$CO',
    '13co': r'$^{13}$CO',
    'cs': r'CS'
}

R_lev = np.arange(25, 0.98*Rout, 50)*au_to_m

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model(Rmin=0, Rmax=1.0)
noise_mean, mask = get_noise_mask(datacube, thres=0)

pix_downsamp = model.grid['step']*meta['downsamp_fit']/au_to_m
step = model.grid['step']/au_to_m

#Useful definitions for plots
xmax = model.skygrid['xmax'] 
xlim = 1.1*Rout
extent= np.array([-xmax, xmax, -xmax, xmax])/au_to_m
plot_title = ctitle + r' $-$ ' + clabel.split('[')[0]

#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
model.make_model()
modelcube = Data('cube_model_%s.fits'%tag, dpc)
#modelcube = model.make_model()
modelcube.convert_to_tb(writefits=False, planck=args.planck)
datacube.convert_to_tb(writefits=False, planck=args.planck)

#*************************
#LOAD MOMENT MAPS
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)

#Show Data or Model?
if args.type=='data':
    map2d = moment_data
    cube = datacube
elif args.type=='model':
    map2d = moment_model
    cube = modelcube
    
vchannels = cube.vchannels

#Show Moment map or Channel map?
if args.channel_id > -1:
    map2d = cube.data[args.channel_id]
    map2d[mask] = np.nan
    plot_title = 'Channel: %.1f km/s'%vchannels[args.channel_id] + r' $-$ ' + clabel.split('[')[0]
    
#**************
#MAKE PLT AXES
#**************
figx = figy = 9

ncols = 5
nrows = 5

dw = 0.9/ncols
dh = dw*figx/figy

fig = plt.figure(figsize=(figx,figy))

ax = [
    [
        fig.add_axes([0.05+i*dw, (0.95-dh)-j*dh, dw, dh]) for i in range(ncols) if (j in [0, nrows-1] or i in [0, ncols-1])
    ] for j in range(nrows)
]

ax_corner = [ax[0][0], ax[0][ncols-1], ax[nrows-1][ncols-1], ax[nrows-1][0]]
for axi in ax_corner:
    axi.axis('off')

ax_order = [
    ax[2][1], ax[1][1],
    ax[0][3], ax[0][2], ax[0][1],
    ax[1][0], ax[2][0], ax[3][0],
    ax[4][1], ax[4][2], ax[4][3],
    ax[3][1]
]

#********************
#MAKE CENTRAL PLOT
#********************
dm = 0.0
axm = fig.add_axes([0.05+dw+dm, 0.05+dh+dm, 3*dw-2*dm, 3*dh-2*dm])

kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=levels_im)
kwargs_cc = dict(colors='k', linestyles='-', extent=extent, levels=levels_cc, linewidths=0.4)
kwargs_cbar = dict(orientation='horizontal', pad=0.03, shrink=0.95, aspect=15)

im0 = axm.contourf(map2d, extend='both', **kwargs_im)
if args.show_contours:
    cc0 = axm.contour(map2d, **kwargs_cc)

make_up_ax(axm, xlims=(-xlim, xlim), ylims=(-xlim, xlim), labelsize=11)
axm.axis('off')

model.make_emission_surface(
    axm,
    R_lev=R_lev,
    #which='upper',
    kwargs_R={'colors': '0.1', 'linewidths': 0.5},
    kwargs_phi={'colors': '0.1', 'linewidths': 0.5}
)
model.make_disc_axes(axm)

#**************************
#GET AND PLOT LINE PROFILE
#**************************
get_sky_from_disc_coords = GridTools.get_sky_from_disc_coords

incl = model.params['orientation']['incl']
PA = model.params['orientation']['PA']
xc = model.params['orientation']['xc']
yc = model.params['orientation']['yc']
vsys = model.params['velocity']['vsys']
orient = (incl, PA, xc, yc)

get_z = lambda r: model.z_upper_func({'R': np.asarray(r)*au_to_m}, **model.params['height_upper'])/au_to_m

def get_lineprofile(data, r, az, npix=0, avg_func=np.nanmean):
    z = get_z(r)
    x, y, _ = get_sky_from_disc_coords(r, az, z, *orient)
    j = np.argmin(np.abs(model.skygrid['xygrid'][0]-x*au_to_m))
    i = np.argmin(np.abs(model.skygrid['xygrid'][1]-y*au_to_m))
    if npix>0:
        line = [avg_func(chan[i-npix:i+npix+1, j-npix:j+npix+1]) for chan in data]
        return x, y, np.asarray(line)
    else:
        return x, y, data[:,i,j]

kw_spec = dict(where="mid", linewidth=1.4, color='k')
cspine = '0.55'
line_peaks, vel_peaks = [], []

#**********************************
#READ PARCUBE AND MAKE CUBE FROM IT
#**********************************
if 'double' in args.kernel:
    parcube_up = fits.getdata('parcube_up_%s_%s_%s.fits'%(args.kernel, args.kind, args.type))
    parcube_low = fits.getdata('parcube_low_%s_%s_%s.fits'%(args.kernel, args.kind, args.type))
    fitdata_up = fit_kernel.get_channels_from_parcube(parcube_up, None, vchannels, method=args.kernel, kind=args.kind)
    fitdata_low = fit_kernel.get_channels_from_parcube(None, parcube_low, vchannels, method=args.kernel, kind=args.kind)        
else:
    parcube_up = fits.getdata('parcube_%s_%s.fits'%(args.kernel, args.type))
    parcube_low = None
            
fitdata = fit_kernel.get_channels_from_parcube(parcube_up, parcube_low, vchannels, method=args.kernel, kind=args.kind)
fitcube = Cube(fitdata, cube.header, vchannels, dpc, beam=cube.beam)

#***********************************
#MAKE LINE PROFILE FOR EACH SUBPANEL
#***********************************
for k, az in enumerate(np.arange(0, 2*np.pi, np.pi/6)):
    axi = ax_order[k]
    az_deg = np.degrees(az) + args.phi0
    az_rad = np.radians(az_deg)
    text = k 

    if args.showfit:
        xm, ym, linem = get_lineprofile(fitcube.data, args.radius, az_rad, npix=args.npix)    
        axi.plot(vchannels, linem, lw=3.5, color='orange') #color='limegreen') #color='mediumorchid') #color='0.6') #

    x, y, line = get_lineprofile(cube.data, args.radius, az_rad, npix=args.npix)        
    axi.step(vchannels, line, **kw_spec)

    if args.showfit and 'double' in args.kernel:
        xu, yu, lineu = get_lineprofile(fitdata_up, args.radius, az_rad, npix=args.npix)
        xl, yl, linel = get_lineprofile(fitdata_low, args.radius, az_rad, npix=args.npix)
        axi.fill_between(vchannels, linel, color='magenta', step='mid', alpha=0.3, hatch='////')
        axi.fill_between(vchannels, lineu, color='lime', step='mid', alpha=0.3)
        
    axi.text(0.9, 0.9, text, va='top', ha='right', fontsize=13, color='k', fontweight='bold', transform=axi.transAxes)
    axi.axvspan(vsys, vsys+10, facecolor='tomato', alpha=0.1)
    axi.axvspan(vsys-10, vsys, facecolor='dodgerblue', alpha=0.1)
    
    axi.axvline(vsys, color='0.2', lw=1.5, dash_capstyle='round', dashes=(0.5, 1.5), alpha=0.8)
    if args.channel_id > -1:
        axi.axvline(cube.vchannels[args.channel_id], linestyle='--', color='cyan', lw=1.5)            
    
    for spine in ['left',  'top', 'right', 'bottom']:
        axi.spines[spine].set_color(cspine)        
    axi.tick_params(which='both', color=cspine, labelcolor=cspine)
    
    if args.npix>0:
        rect = patches.Rectangle(
            (x-step*args.npix, y-step*args.npix),
            2*step*args.npix,
            2*step*args.npix,
            lw=1.5,
            edgecolor='k',
            facecolor="none",
        )
        axm.add_patch(rect)
        dxt = step*np.cos(az_rad+PA)*(4+args.npix)
        dyt = step*np.sin(az_rad+PA)*(4+args.npix)        
        axm.text(x+dxt, y+dyt, text, va='center', ha='center', color='k', fontsize=13)

    else:
        axm.text(x, y, text, va='center', ha='center', color='k', fontsize=13)

    line_peaks.append(np.nanmax(line))
    vel_peaks.append(vchannels[np.argmax(line)])
        
disclabel = plot_title.split('$-$')[0]
momentlabel = plot_title.split('$-$')[1]    
axm.text(0.001, 0.98, momentlabel, va='top', ha='left', color='0.1', fontsize=14, transform=axm.transAxes)
axm.text(0.98, 0.02, disclabel, va='bottom', ha='right', color='0.1', fontsize=14, transform=axm.transAxes)
cube.plot_beam(axm, fc='0.8')
    
max_peak = np.max(line_peaks)        
min_vel = np.min(vel_peaks)
max_vel = np.max(vel_peaks)

#************************
#DECORATE SUBPLOTS LAYOUT
#************************
kw_ax = dict(labelsize=11)
for row, axrow in enumerate(ax):
    for col, axi in enumerate(axrow):
        
        if row==0 and col==1:
            make_up_ax(axi, labelbottom=False, labeltop=True, **kw_ax)
        if row==0 and col!=1:
            make_up_ax(axi, labelbottom=False, labeltop=False, labelleft=False, **kw_ax)
            
        if row>0 and col==0:
            make_up_ax(axi, labelbottom=False, labeltop=False, labelleft=False, **kw_ax)
        if row==nrows-2 and col==0:
            make_up_ax(axi, labelbottom=True, labeltop=False, labelleft=True, **kw_ax)            
            axi.set_xlabel(r'$\upsilon_{\rm l.o.s}$ [km/s]', fontsize=13, color=cspine)
            axi.set_ylabel(r'Intensity [K]', fontsize=13, color=cspine)            

        if row==1 and col==1:
            make_up_ax(axi, labelbottom=False, labeltop=True, labelleft=False, labelright=True, **kw_ax)                        
        if row>1 and col==1:
            make_up_ax(axi, labelbottom=False, labeltop=False, labelleft=False, labelright=False, **kw_ax)
            
        if row==nrows-1 and col==ncols-2:
            make_up_ax(axi, labelbottom=True, labeltop=False, labelleft=False, labelright=True, **kw_ax)                        
        if row==nrows-1 and col<ncols-2:
            make_up_ax(axi, labelbottom=False, labeltop=False, labelleft=False, labelright=False, **kw_ax)                        

        mod_major_ticks(axi, axis='x', nbins=5)
        mod_major_ticks(axi, axis='y', nbins=4)

        axi.set_xlim(min_vel-1.5, max_vel+1.5)
        axi.set_ylim(-0.1*max_peak, 1.1*max_peak)

        axi.tick_params(labelsize=12)
        
plt.savefig('line_profiles_%s_%s_%s_%.0fau_phi0_%sdeg_%s.png'%(meta['disc'], meta['mol'],args.kernel,args.radius,args.phi0,args.type), bbox_inches='tight', dpi=200)    
show_output(args)
