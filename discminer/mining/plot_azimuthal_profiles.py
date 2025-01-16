from discminer.mining_control import _mining_azimuthal_profiles
from discminer.mining_utils import (load_disc_grid,
                                    load_moments,
                                    init_data_and_model,
                                    get_noise_mask,
                                    show_output,
                                    get_1d_plot_decorators)

from discminer.rail import Rail
from discminer.plottools import (get_discminer_cmap,
                                 make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,                                
                                 use_discminer_style)

import json
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_azimuthal_profiles(None)
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
incl = best['orientation']['incl']
Rout = best['intensity']['Rout']
au_to_m = u.au.to('m')

clabel, clabel_res, clim0, clim0_res, clim1, clim1_res, unit = get_1d_plot_decorators(args.moment, tag=args.type)

if args.type=='residuals':
    clim0 = clim0_res
    clim1 = clim1_res
    clabel = clabel_res

if args.moment=='velocity':
    if args.type=='residuals':
        clabel = r'Velocity %s %s'%(args.type, unit)
    else:
        clabel = r'Deprojected Velocity %s %s'%(args.type, unit)        

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model()

noise_mean, mask = get_noise_mask(datacube, thres=2)
vchannels = datacube.vchannels
model.make_model()

#******************
#LOAD MOMENT MAPS
#******************
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)

if args.type=='residuals': map2d = residuals
elif args.type=='data': map2d = moment_data
elif args.type=='model': map2d = moment_model

if args.type!='residuals' and args.moment=='velocity': #deproject velocity field assuming vphi is dominant
    map2d = np.abs((map2d-vsys)/(np.cos(model.projected_coords['phi'][args.surface])*np.sin(incl)))

#***********
#MAKE PLOT
#***********
beam_au = datacube.beam_size.to('au').value
R_prof = np.arange(args.Rinner*beam_au, args.Router*Rout, beam_au/4)

color_bounds = np.array([0.33, 0.66, 1.0])*Rout
lws = [0.4, 0.6, 0.4]

rail = Rail(model, map2d, R_prof)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,6))
ax2 = fig.add_axes([0.85,0.6,0.3*6/14,0.3])

R_list, phi_list, resid_list, color_list = rail.prop_along_coords(coord_ref=args.radius, surface=args.surface,
                                                                  color_bounds=color_bounds, lws=lws,
                                                                  ax=ax, ax2=ax2, interpgrid=args.interpgrid)

ax.set_xlabel(r'Azimuth [deg]')
ax.set_ylabel(clabel)
ax.set_xlim(-180,180)
ax.set_ylim(clim0, clim1)
ax.grid()

model.make_emission_surface(ax2, which=mtags['surf'])
model.make_disc_axes(ax2)
make_up_ax(ax, labeltop=False)
make_up_ax(ax2, labelbottom=False, labelleft=False, labeltop=True)
ax.tick_params(labelbottom=True, top=True, right=True, which='both', direction='in')

tick_angles = np.arange(-150, 181, 30)
ax.set_xticks(tick_angles)

plt.savefig('azimuthal_%s_%s.png'%(args.moment, args.type), bbox_inches='tight', dpi=200)
show_output(args)
