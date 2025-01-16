from discminer.mining_control import _mining_radial_profiles_wedge
from discminer.mining_utils import (init_data_and_model,
                                    get_noise_mask,
                                    get_1d_plot_decorators,
                                    load_moments,
                                    load_disc_grid,
                                    show_output,
                                    MEDIUM_SIZE)

from discminer.rail import Rail
from discminer.plottools import (get_discminer_cmap,
                                 make_substructures,
                                 make_up_ax,
                                 mod_minor_ticks,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 make_1d_legend,
                                 truncate_colormap,
                                 use_discminer_style)
                                 
import json
import numpy as np
from astropy import units as u
from matplotlib import patches
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_radial_profiles_wedge(None)
    args = parser.parse_args()

#**************************
#JSON AND SOME DEFINITIONS
#**************************
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

#*************************
#ABSOLUTE RESIDUALS    
if args.moment=='velocity':
    residuals_abs = np.abs(moment_data-vsys) - np.abs(moment_model-vsys)

#*************************
#RADIAL BINS AND UTILS
beam_au = datacube.beam_size.to('au').value
if args.absolute_Rinner>=0:
    Rprof_in = args.absolute_Rinner
else:
    Rprof_in = args.Rinner*beam_au

if args.absolute_Router>=0:
    Rprof_out = args.absolute_Router
else:
    Rprof_out = args.Router*Rout
    
R_prof = np.arange(Rprof_in, Rprof_out, beam_au/4.0) #changed to beam/4 from beam/5 before
xlim0, xlim1 = 0.5*R_prof[0], 1.05*R_prof[-1]

def make_zones_code(fig, ax, az_zones, color_zones):
    figx, figy = fig.get_size_inches()
    figr = figy/figx
    axp = ax.get_position()    
    dy = 0.3
    ax1 = fig.add_axes([axp.x1 - dy*figr/2., axp.y1 - dy/2, dy*figr, dy])    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)

    wedges = []
    for i,zone in enumerate(az_zones):
        wedges.append(patches.Wedge((0,0), 1, zone[0], zone[1], hatch=None, fc=color_zones[str(i)], fill=True, alpha=0.8))

    circled = patches.Circle((0,0), 1, lw=3, ec='k', fill=False)
    circlef = patches.Circle((0,0), 1, fc='w', ec='none', fill=True)    

    ax1.add_artist(circlef)

    for wedge in wedges:
        ax1.add_artist(wedge)

    ax1.add_artist(circled)
    ax1.axis('off')

    return ax1

def make_savgol(prof):
    if not args.savgol_filter:
        return prof, None
    try:
        ysav = savgol_filter(prof, 5, 3)
        ysav_deriv = savgol_filter(prof, 5, 3, deriv=1)
    except np.linalg.LinAlgError:
        ysav = prof
        ysav_deriv = None
    return ysav, ysav_deriv 

#**********************************
#AZIM. AVERAGED RESIDUALS PER ZONE
#**********************************
rail = Rail(model, residuals, R_prof)
az_zones = [[a, b] for a,b in zip(args.wedges[::2], args.wedges[1::2])]
cmap = truncate_colormap(plt.get_cmap('nipy_spectral'), minval=0.1, maxval=0.9)
color_list = [cmap(ind) for ind in np.linspace(0, 1, len(az_zones))]
color_zones = {str(i): color_list[i] for i in range(len(az_zones))} #{'0': '#d90429', '1': '#f77f00', '2': '#43ABC9', '3': '#0d3d56'}
zones, zones_error  = rail.get_average_zones(az_zones=az_zones, sigma_thres=args.sigma)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(11, 3))
ax1 = make_zones_code(fig, ax, az_zones, color_zones)

for i in range(len(az_zones)):
    try:
        ysav, ysav_deriv = make_savgol(zones[i])
        ax.plot(R_prof, ysav, c=color_zones[str(i)], lw=2, zorder=11)
        ax.fill_between(R_prof, zones[i]+zones_error[i], zones[i]-zones_error[i], color=color_zones[str(i)], alpha=0.15, zorder=9) #, label='zone '+str(i))
    except np.linalg.LinAlgError:
        ax.scatter(R_prof, zones[i], c=color_zones[str(i)], edgecolor='0.1', zorder=11)
            
for i,lev in enumerate(R_prof):
    ax.scatter([lev]*len(rail._resid_list[i]), rail._resid_list[i], color='0.9', alpha=0.9, s=5, zorder=-1)
ax.scatter(None, None, color='0.8', alpha=0.9, s=10, zorder=-1, label='Points from all azimuths')    

make_up_ax(ax, labelbottom=True, labeltop=False, xlims=(R_prof[0], R_prof[-1]), ylims=np.array([clim0_res, clim1_res]))
mod_major_ticks(ax, axis='x', nbins=8)
mod_major_ticks(ax, axis='y', nbins=5)

make_substructures(ax, gaps=gaps, rings=rings, label_gaps=True, label_rings=True)      

ax.set_xlabel('Radius [au]')
if args.moment in ['velocity', 'v0phi', 'v0r', 'v0z', 'vr_leftover']:
    ax.set_ylabel(r'$\delta\upsilon$ [km/s]', fontsize=MEDIUM_SIZE, labelpad=10)    

make_1d_legend(ax, scatterpoints=3, fontsize=13, loc='lower right', bbox_to_anchor=(0.9,1.0))

plt.savefig('wedge_radprof_residuals_%s.png'%tag_base, bbox_inches='tight', dpi=200)
show_output(args)
