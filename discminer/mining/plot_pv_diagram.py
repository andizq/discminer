from discminer.mining_control import _mining_pv_diagram
from discminer.mining_utils import (get_2d_plot_decorators,
                                    get_noise_mask,
                                    load_moments,
                                    load_disc_grid,
                                    init_data_and_model,
                                    show_output)

from discminer.rail import Contours
from discminer.grid import GridTools
from discminer.plottools import (make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 get_discminer_cmap,
                                 use_discminer_style)

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from astropy import units as u

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_pv_diagram(None)
    args = parser.parse_args()

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

rings = custom['rings']
gaps = custom['gaps']
kinks = []

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment)
    
file_data = meta['file_data']
tag = meta['tag']

dpc = meta['dpc']*u.pc
au_to_m = u.au.to('m')

pvphi = np.radians(args.pvphi)
pvcmap = get_discminer_cmap('peakintensity')

#*******************
#LOAD DATA AND GRID
#*******************
datacube, model = init_data_and_model(Rmin=0, Rmax=1.0)
noise_mean, mask = get_noise_mask(datacube, thres=2)
vchans_shifted = datacube.vchannels - vsys

with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.1*Rout
extent= np.array([-xmax, xmax, -xmax, xmax])

beam_au = datacube.beam_size.to('au').value

moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)
R_prof = np.arange(args.Rinner*beam_au, args.Router*Rout, beam_au/4)

#*********************
#DISC REFERENCE FRAME
#*********************
R, phi, z = load_disc_grid() #No need to make_model
R_au = R[args.surface]/au_to_m
R_au_flat = R_au.flatten()
phi_rad = phi[args.surface]
indices = np.arange(len(R_au_flat))

def get_skypixel_ij(Ri, phii, tol=beam_au):

    phi_diff = np.abs(phi_rad - phii).flatten()

    pind = Ri * phi_diff < tol  #pixels where this arc length is smaller than tol    
    rind = np.abs(Ri - R_au_flat[pind]) < tol #same for dR

    R_tol = R_au_flat[pind][rind]
    phi_tol = phi_diff[pind][rind]
    tot_diff = np.hypot(R_tol-Ri, R_tol*phi_tol) #distance between Ri, phii and nearby pixels
    
    indbest = indices[pind][rind][np.argmin(tot_diff)]

    return np.unravel_index(indbest, R_au.shape)

#**********
#MAKE PLOT
#**********
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,8))

ax[0].set_aspect(1)

pos0 = ax[0].get_position()
height1 = 0.5
ax1 = fig.add_axes([pos0.x1+0.05, pos0.y0+0.5*pos0.height-0.5*height1, 0.4, height1])
ax[1].axis('off')

ax_cbar0 = fig.add_axes([pos0.x0, 0.09, pos0.width, 0.04])


if args.surface=='upper':
    z_prof = model.z_upper_func({'R': R_prof*u.au.to('m')}, **model.params['height_upper'])*u.m.to('au')
elif args.surface=='lower':
    z_prof = model.z_lower_func({'R': R_prof*u.au.to('m')}, **model.params['height_lower'])*u.m.to('au')
else:
    raise InputError(surface, "Only 'upper' or 'lower' are valid surfaces.")


kwargs_axes = dict(ls=':', lw=4.5, dash_capstyle='round', dashes=(0.5, 1.5), alpha=1)

make_ax = lambda ax, x, y, color: ax.plot(x, y, color=color, **kwargs_axes)        

pv_axis0 = np.zeros_like(R_prof) + pvphi
pv_axis1 = np.zeros_like(R_prof) + pvphi - np.pi 

isort = np.argsort(R_prof)
x_cont, y_cont, _ = GridTools.get_sky_from_disc_coords(R_prof[isort], pv_axis0, z_prof[isort], incl, PA, xc, yc)    
make_ax(ax[0], x_cont, y_cont, 'tomato')

x_cont, y_cont, _ = GridTools.get_sky_from_disc_coords(R_prof[isort], pv_axis1, z_prof[isort], incl, PA, xc, yc)    
make_ax(ax[0], x_cont, y_cont, 'dodgerblue')

isky_p, jsky_p = [], []
isky_m, jsky_m = [], []

line_profile_p = []
line_profile_m = []
for Ri in R_prof:
    i, j = get_skypixel_ij(Ri, pvphi)
    isky_p.append(i)
    jsky_p.append(j)
    #ax[0].scatter(model.skygrid['meshgrid'][0][i,j]/au_to_m, model.skygrid['meshgrid'][1][i,j]/au_to_m, s=80, zorder=10)
    line_profile_p.append(datacube.data[:,i,j])

    i, j = get_skypixel_ij(Ri, pvphi-np.pi)
    isky_m.append(i)
    jsky_m.append(j)
    line_profile_m.append(datacube.data[:,i,j])
    
line_profiles = np.append(line_profile_m[::-1], line_profile_p, axis=0)

R_axis = np.append(-R_prof[::-1], R_prof)
ax1.contourf(R_axis, vchans_shifted, line_profiles.T, levels=np.linspace(0, np.nanmax(line_profiles), 32), cmap=pvcmap)

ax1.fill_between([-R_prof[0], R_prof[0]], vchans_shifted[0], vchans_shifted[-1], color='k', alpha=0.4)

#****************
#PLOT MOMENT MAP
#****************
kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=levels_im)
kwargs_cc = dict(colors='k', linestyles='-', extent=extent, levels=levels_cc, linewidths=0.4)
kwargs_cbar = dict(orientation='horizontal', pad=0.03, shrink=0.95, aspect=15)

im = ax[0].contourf(moment_data, extend='both', **kwargs_im)
if args.surface!='lower':
    cc = ax[0].contour(moment_data, **kwargs_cc)
make_up_ax(ax[0], xlims=(-xlim, xlim), ylims=(-xlim, xlim), labelsize=13, color='k', labelcolor='k')

ax[0].scatter(best['orientation']['xc'], best['orientation']['yc'],
            ec='k', fc='w', marker='X', lw=0.5, s=60, zorder=20)        
datacube.plot_beam(ax[0], fc='0.8')
mod_major_ticks(ax[0], axis='both', nbins=8)

make_up_ax(ax1, xlims=(-R_prof[-1], R_prof[-1]), ylims=(vchans_shifted[0], vchans_shifted[-1]), labelsize=13, color='0.3', labelcolor='0.3', labelbottom=True, labeltop=False, labelright=True)
mod_major_ticks(ax1, axis='both', nbins=8)

for axi in [ax1]:
    axi.tick_params(labelleft=False)
    for side in ['top','bottom','left','right']:
        axi.spines[side].set_linewidth(4.0)
        axi.spines[side].set_color('0.5')
        axi.spines[side].set_linestyle((0, (1,1.5)))
        axi.spines[side].set_capstyle('round')
    axi.grid(color='k', ls='--')

Contours.emission_surface(ax[0], R, phi, extent=extent,
                          R_lev=R_prof*u.au.to('m'),
                          which=mtags['surf'],
                          kwargs_R={'colors': '0.1', 'linewidths': 0.5},
                          kwargs_phi={'colors': '0.1', 'linewidths': 0.4}
)
   
ax[0].set_ylabel('Offset [au]', fontsize=15)
ax[0].set_title(ctitle, pad=40, fontsize=17)

ax1.set_title(r'PV along $\phi=%d^\circ$'%args.pvphi, pad=20, fontsize=17, color='0.3')
ax1.set_xlabel('Offset [au]', color='0.3', fontsize=15)
ax1.set_ylabel(r'$\upsilon_{\rm l.o.s}$ [km/s]', fontsize=16, color='0.3')
ax1.yaxis.set_label_position("right")
ax1.yaxis.set_major_formatter(FormatStrFormatter('%2d'))

cbar0 = plt.colorbar(im, cax=ax_cbar0, format='%.1f', **kwargs_cbar)
cbar0.ax.tick_params(labelsize=12) 
mod_nticks_cbars([cbar0], nbins=6)
cbar0.set_label(clabel, fontsize=15)

plt.savefig('pv_diagram_%s_%ddeg.png'%(mtags['base'], args.pvphi), bbox_inches='tight', dpi=200)
show_output(args)
