from discminer.mining_control import _mining_intensdistrib
from discminer.core import Data

from discminer.mining_utils import (_get_mask_tuples,
                                    _merge_R_phi_mask,
                                    get_noise_mask,
                                    load_disc_grid,
                                    get_2d_plot_decorators,
                                    init_data_and_model,
                                    load_moments,
                                    mark_planet_location,
                                    show_output)

from discminer.plottools import (make_up_ax,
                                 make_round_map,
                                 use_discminer_style)

import json
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_intensdistrib(None)
    args = parser.parse_args()

#*******************
#FILL UP EMPTY ARGS
#*******************
nmasks = int(round(0.5*len(args.annuli)))

if len(args.r0)==nmasks:
    rcenters = args.r0    
else:
    nmissing = nmasks - len(args.r0)
    rcenters = np.append(args.r0, [0]*nmissing)
    
if len(args.phi0)==nmasks:
    phicenters = np.radians(args.phi0)
else:
    nmissing = nmasks - len(args.phi0)
    phicenters = np.radians(np.append(args.phi0, [0]*nmissing))

if len(args.colors)==nmasks:
    colors = args.colors
else:
    nmissing = nmasks - len(args.colors)
    colors = np.append(args.colors, ['black']*nmissing)

if len(args.linewidths)==nmasks:
    lws = args.linewidths
else:
    nmissing = nmasks - len(args.linewidths)
    lws = np.append(args.linewidths, [1.5]*nmissing)
    
#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

vel_sign = best['velocity']['vel_sign']
vsys = best['velocity']['vsys']
Rout = best['intensity']['Rout']
incl = best['orientation']['incl']
PA = best['orientation']['PA']
xc = best['orientation']['xc']
yc = best['orientation']['yc']

gaps = custom['gaps']
rings = custom['rings']
try:
    kinks = custom['kinks']
except KeyError:
    kinks = []

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment)

if args.clim!=0:
    clim = args.clim
    
#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc

R_lev = np.arange(25, 0.98*Rout, 50)*au_to_m

#*******************
#LOAD DATA AND MODEL
#*******************
datacube, model = init_data_and_model(Rmin=0, Rmax=1.6)
noise_mean, mask = get_noise_mask(datacube, thres=3)
modelcube = Data('cube_model_%s.fits'%tag, dpc)

datapdf = np.where(mask[None, :, :], np.nan, datacube.data)

#Definitions for plots
beam_au = datacube.beam_size.to('au').value
if args.absolute_Rinner>=0:
    Rmod_in = args.absolute_Rinner
else:
    Rmod_in = args.Rinner*beam_au

if args.absolute_Router>=0:
    Rmod_out = args.absolute_Router
else:
    Rmod_out = args.Router*Rout

xmax = model.skygrid['xmax'] 
xlim = 1.1*Rout
extent= np.array([-xmax, xmax, -xmax, xmax])/au_to_m
#plot_title = ctitle + r' $-$ ' + clabel.split('[')[0]

#****************************
#LOAD DISC GEOMETRY AND MASK
#****************************
R, phi, z = load_disc_grid()

R_s = R[args.surface]*u.m.to('au')
phi_s = np.degrees(phi[args.surface])

Xproj = R_s*np.cos(phi[args.surface])
Yproj = R_s*np.sin(phi[args.surface])

#*************************
#LOAD AND CLIP MOMENT MAPS
#Clip radially
# Masked sections are visually drawn at the end
moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask, #fill masked cells with nans
    clip_Rmin=0.0*u.au,
    clip_Rmax=Rmod_out*u.au,
    clip_Rgrid=R_s*u.au
)

#***********
#MAKE MASKS
#***********
masktuples_R = _get_mask_tuples(args.annuli)
masktuples_phi = _get_mask_tuples(args.wedges)

if len(masktuples_phi)==nmasks:
    pass
else:
    nmissing = nmasks - len(masktuples_phi)
    masktuples_phi = masktuples_phi + [[]]*nmissing
    
masks = []
for i in range(nmasks):

    if rcenters[i] != 0:
        x0 = rcenters[i]*np.cos(phicenters[i])
        y0 = rcenters[i]*np.sin(phicenters[i])
        print (x0, y0)
        Rgrid = np.hypot(Xproj - x0, Yproj-y0)
        phigrid = np.arctan2(Yproj - y0, Xproj - x0)
    else:
        Rgrid = R_s
        phigrid = phi_s

    mask_R={'map2d': Rgrid, 'lims': masktuples_R[i]}        
    mask_phi={'map2d': phigrid, 'lims': masktuples_phi[i]}

    mask0 = _merge_R_phi_mask(mask_R, mask_phi)
    if i>0:
        maski = mask0 & (~masks[i-1]) 
    else:
        maski = mask0
        
    masks.append(maski)

#*********
#MAKE PDF
#*********
def make_pdf(maski):
    intensities = datapdf[:, maski].ravel()
    intensities = intensities[intensities>args.sigma*noise_mean]
    intensities = intensities/np.nanmax(intensities)
    kde = gaussian_kde(intensities) #Normalise to peak intensity
    Igrid = np.linspace(np.nanmin(intensities), np.nanmax(intensities), 300)
    pdf = kde(Igrid)
    return Igrid, pdf
    
#**************
#MAKE PLT AXES
#**************
figx, figy = 12, 4
x2y = figx/figy

fig = plt.figure(figsize=(figx,figy))

axr = fig.add_axes([0.05, 0.05, 0.9/x2y, 0.9])
axp0 = 0.05+0.9/x2y+0.03
axp = fig.add_axes([axp0, 0.14, 0.95-axp0, 0.72])

for i in range(nmasks):
    Igrid, pdf = make_pdf(masks[i])
    #print (i, skew(pdf), kurtosis(pdf))
    axp.plot(Igrid, pdf, color=colors[i], lw=lws[i], zorder=50-i)
    axp.fill_between(Igrid, pdf, color='k', lw=0, alpha=0.05)    

make_up_ax(axp, labelleft=False, labelright=True, labeltop=False, labelbottom=True, labelsize=16)
axp.set_xticks([0.0, 0.5, 1.0])
axp.set_yticks([0, 1, 2])
axp.set_xticklabels([0, r'0.5I$_{\rm peak}$', r'I$_{\rm peak}$'])

#axp.text(0, 1.02, 'Intensity Distribution of Velocity Channels', fontsize=12, transform=axp.transAxes)
axp.text(0, 1.02, 'Probability Density of Intensities across Velocity Channels', fontsize=12, transform=axp.transAxes)

axp.yaxis.set_label_position("right")
#axp.set_ylabel('Probability density', fontsize=14, labelpad=15)
axp.set_xlim(0, 1)
axp.set_ylim(0, None)

#***************
#MAKE ROUND MAP
#***************
kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=levels_im)

if args.surface in ['up', 'upper']:
    z_func = model.z_upper_func
    z_pars = best['height_upper']

elif args.surface in ['low', 'lower']:
    z_func = model.z_lower_func
    z_pars = best['height_lower']

levels_resid = np.linspace(-clim, clim, 32)
        
fig, axr = make_round_map(1000*residuals, levels_resid, Xproj*u.au, Yproj*u.au, Rmod_out*u.au,
                          fig=fig, ax=axr,
                          kwargs_contourf=dict(cmap=cmap_res), #Skip set_under, set_over by overwriting cmap
                          z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                          cmap=cmap_res, clabel='m/s', fmt='%d',
                          gaps=gaps, rings=rings, kinks=kinks,
                          make_cbar=args.colorbar,                         
                          mask_inner=Rmod_in*u.au,
                          fontsize_azimuthal_grid=args.fontsize-6,
                          fontsize_radial_grid=args.fontsize-4, 
                          fontsize_cbar=args.fontsize-6,
                          fontsize_xaxis=args.fontsize-6,
                          fontsize_nskyaxis=args.fontsize-6,                          
                          make_nskyaxis=args.show_nsky,
                          make_Rout_proj=args.show_xaxis,
                          make_xaxis=args.show_xaxis,                          
)

#Filled regions
for i, m in enumerate(masks):
    if np.any(m):
        axr.contourf(
            Xproj, Yproj, m.astype(float),
            levels=[0.5, 1.5],
            colors=[colors[i]],
            alpha=0.3,
            #hatches=['///'],
            zorder=30 - i,
            antialiased=True
        )

if len(args.mask_R)>0 or len(args.mask_phi)>0:
    make_masks(axr, args.mask_R, args.mask_phi, Rmax=Rmod_out, facecolor='k', alpha=0.3)
    
mark_planet_location(axr, args, edgecolors='k', lw=0, s=550, coords='disc', zfunc=z_func, zpars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc, dpc=dpc)    

plt.savefig('intensity_distribution_%s_%s.png'%(meta['disc'], meta['mol']), bbox_inches='tight', dpi=200)    
show_output(args)

