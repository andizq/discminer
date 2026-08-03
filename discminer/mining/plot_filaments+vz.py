from discminer.mining_control import _mining_filaments_vz
from discminer.rail import Rail

from discminer.mining_utils import (load_disc_grid,
                                    load_parfile,
                                    load_moments,
                                    init_data_and_model,
                                    get_noise_mask,
                                    get_2d_plot_decorators,
                                    show_output)

from discminer.plottools import (make_up_ax,
                                 make_1d_legend,
                                 make_round_map,
                                 use_discminer_style)

import numpy as np
import matplotlib.pyplot as plt
from fil_finder import Filament2D
from astropy import units as u
from astropy.io import fits
from functools import reduce
from scipy.stats import binned_statistic

import json

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_filaments_vz(None)
    args = parser.parse_args()

with open('filaments_colors.json') as json_file:
    colors_dict = json.load(json_file, parse_float=float)

filids = np.asarray(args.filament_ids)

if len(filids)>0:
    filtag = '_'+reduce(lambda a,b: a+b, filids.astype(str)).replace('-', 'm')
else: #if no filids passed, consider all filaments
    filtag = '_all'
    filids = np.asarray(list(colors_dict.keys())).astype(int)
    
filids_pos = filids[filids>0]
filids_neg = np.abs(filids[filids<0])

#**********************
#JSON AND PARSER STUFF
#**********************
meta, params, custom = load_parfile()

z_pars = params['height_upper']
vsys = params['velocity']['vsys']
incl = params['orientation']['incl']
PA = params['orientation']['PA']
xc = params['orientation']['xc']
yc = params['orientation']['yc']
Rout = params['intensity']['Rout']
au_to_m = u.au.to('m')

gaps = custom['gaps']
rings = custom['rings']
kinks = custom['kinks']

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, unit_simple=True, fmt_vertical=True, args=args)
levels_resid = np.linspace(-clim, clim, 32)

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model()
model.make_model()

R, phi, z = load_disc_grid()
RR = R[args.surface]/au_to_m
PP = np.degrees(phi[args.surface])

Xproj = RR*np.cos(phi[args.surface])
Yproj = RR*np.sin(phi[args.surface])

noise_mean, mask = get_noise_mask(datacube, thres=2,
                                  mask_phi={'map2d': PP,
                                            'lims': args.mask_phi},
                                  mask_R={'map2d': RR,
                                          'lims': args.mask_R}
)

#*************************
#LOAD AND CLIP MOMENT MAPS
#*************************
moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask,
    clip_Rmin=0.0*u.au,
    clip_Rmax=args.Router*Rout*u.au,
    clip_Rgrid=R[args.surface]*u.m
)

if args.type=='residuals': map2d = residuals
elif args.type=='data': map2d = moment_data
elif args.type=='model': map2d = moment_model

if args.type!='residuals' and args.moment=='velocity': #deproject velocity field
    map2d = np.abs((map2d-vsys)/(np.cos(model.projected_coords['phi'][args.surface])*np.sin(incl)))

#*************************
#READ FILAMENT FILES
#*************************
tag = mtags['base'].replace(args.moment, args.filament_moment)
filaments_pos = fits.getdata('filaments_pos_%s_cartesian.fits'%tag)
filaments_neg = fits.getdata('filaments_neg_%s_cartesian.fits'%tag)

fil_pos_obj = [Filament2D.from_pickle('filaments_pos_%s_cartesian_id%d.pkl'%(tag, i)) for i in filids_pos]
fil_neg_obj = [Filament2D.from_pickle('filaments_neg_%s_cartesian_id%d.pkl'%(tag, i)) for i in filids_neg]

respos = np.where(map2d<0.0, np.nan, map2d)
resneg = np.where(map2d>0.0, np.nan, map2d)

#*****************
#IRREGULAR QUIVER
#*****************
R_avg, vz_avg, err_avg = np.loadtxt('radial_profile_velocity_vz.dat').T

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,5))
axr = fig.add_axes([0.93, 0.05, 0.43, 2*0.43])

#Deprojected map
fig, axr = make_round_map(residuals, levels_resid, Xproj*u.au, Yproj*u.au, args.Router*Rout*u.au,
                          fig=fig, ax=axr, 
                          z_func=model.z_upper_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                          cmap=cmap_res, clabel=unit, fmt=cfmt, 
                          gaps=gaps, rings=rings, kinks=kinks,
                          fontsize_radial_grid=args.fontsize-1,
                          fontsize_azimuthal_grid=args.fontsize-2,
                          fontsize_cbar=args.fontsize-2,
                          fontsize_xaxis=args.fontsize-2,
                          fontsize_nskyaxis=args.fontsize,
                          make_nskyaxis=args.show_nsky,
                          make_Rout_proj=args.show_xaxis,
                          make_xaxis=args.show_xaxis,
                          make_cbar=False, 
                          mask_inner=args.Rinner*datacube.beam_size,
                          kwargs_mask={'zorder': 30})


Rfilmax = []
Rfilmin = []
vfilmax = []
   
#POSITIVE VALUES
vzproj = -1/np.cos(incl)

def make_plot(filids, filaments_skel, filaments_obj, component='pos', make_cbar=True):
    
    if component in ['pos', 'positive']:
        resmap = respos
        facecolor = 'tomato'
    elif component in ['neg', 'negative']:
        resmap = resneg
        facecolor = 'dodgerblue'
        
    Rp, vp = [], []
    for i, posi in enumerate(filids):
        fil = filaments_skel[2+posi-1]
        if args.longpath:
            fil *= filaments_skel[1] 
        fp = fil.astype(bool)
        Rp.append(RR[fp])
        vp.append(map2d[fp])
        Rfilmax.append(np.nanmax(RR[fp]))
        Rfilmin.append(np.nanmin(RR[fp]))

        filobj = filaments_obj[i]
        rows = []
        cols = []
        for j in range(len(filobj._xpix)):
            rows.append(filobj._image_rows[filobj._xpix[j], filobj._ypix[j]].astype(int))
            cols.append(filobj._image_cols[filobj._xpix[j], filobj._ypix[j]].astype(int))

        sc = ax.scatter(RR[fp], vzproj*map2d[fp], edgecolor='k', c=PP[fp], s=45, cmap='nipy_spectral', alpha=0.9, zorder=20)
        sc2 = ax.scatter(RR[rows,cols], vzproj*resmap[rows,cols], marker='o', edgecolors='none', c=PP[rows,cols], cmap='nipy_spectral', s=15, alpha=0.2, zorder=19)

        axr.scatter(Xproj[rows,cols], Yproj[rows,cols], marker='o', edgecolors='none', c=PP[rows,cols], cmap='nipy_spectral', s=10, alpha=0.7, zorder=19)
        axr.scatter(Xproj[fp], Yproj[fp], marker='o', edgecolors='none', c=PP[fp], cmap='nipy_spectral', s=35, alpha=0.9, zorder=20)
        
    if len(Rp)>0:
        Rp = np.hstack(Rp)
        vp = np.hstack(vp)
        bin_vals, bin_edges, binnumber = binned_statistic(Rp, vp, statistic='median', bins=10)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        ax.plot(bin_centers, vzproj*bin_vals, lw=4, color=facecolor, label='Filament vz', zorder=19)

    return sc

if len(filids_pos)>0:
    sc = make_plot(filids_pos, filaments_pos, fil_pos_obj, component='pos')

if len(filids_neg)>0:    
    sc = make_plot(filids_neg, filaments_neg, fil_neg_obj, component='neg')

cbar = plt.colorbar(sc)
cbar.set_label("Azimuth [deg]", fontsize=args.fontsize)

ax.plot(R_avg[R_avg<300], vz_avg[R_avg<300], lw=4, color='0.1', label='Axisymmetric vz')

ax.set_ylabel('vz [km/s]', fontsize=args.fontsize)
ax.set_xlabel('Radius [au]', fontsize=args.fontsize)

make_up_ax(ax,
           labelsize=args.fontsize,
           labeltop=False, labelbottom=True,
           xlims=(0.8*np.min(Rfilmin), 1.1*np.max(Rfilmax)),
           ylims=(-clim, clim)
)

make_1d_legend(ax, fontsize=args.fontsize)

plt.savefig('attribute_filaments%s_%s_filmom%s_%s_vz_vs_R.png'%(filtag, mtags['base'], args.filament_moment, args.type), bbox_inches='tight', dpi=200)

show_output(args)
