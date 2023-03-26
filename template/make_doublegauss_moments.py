from discminer.core import Data
from discminer.cube import Cube
from discminer.disc2d import General2d
from discminer.rail import Contours
from discminer.plottools import get_discminer_cmap, make_up_ax, mod_major_ticks, use_discminer_style, mod_nticks_cbars

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
import sys

use_discminer_style()
#****************
#SOME DEFINITIONS
#****************
file_data = 'MWC_480_CO_220GHz.robust_0.5.JvMcorr.image.pbcor_clipped_downsamp_10pix.fits'
tag = 'mwc480_12co'
nwalkers = 256
nsteps = 10000
au_to_m = u.au.to('m')

dpc = 162*u.pc
Rmax = 700*u.au

#********
#GRIDDING
#********
downsamp_pro = 10 # Downsampling used for prototype
downsamp_fit = 10 # Downsampling used for MCMC fit
downsamp_factor = (downsamp_fit/downsamp_pro)**2 # Required to correct intensity normalisation for prototype

datacube = Data(file_data, dpc) # Read data and convert to Cube object
vchannels = datacube.vchannels

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
model = General2d(datacube, Rmax, Rmin=0, prototype=True)
# Prototype? If False discminer assumes you'll run an MCMC fit

def intensity_powerlaw_rout(coord, I0=30.0, R0=100, p=-0.4, z0=100, q=0.3, Rout=500):
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    z = coord['z']
    R0*=au_to_m
    z0*=au_to_m
    Rout*=au_to_m
    A = I0*R0**-p*z0**-q
    Ieff = np.where(R<=Rout, A*R**p*np.abs(z)**q, 0.0)
    return Ieff

def z_upper(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

def z_lower(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return -au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

model.z_upper_func = z_upper
model.z_lower_func = z_lower
model.velocity_func = model.keplerian_vertical # vrot = sqrt(GM/r**3)*R
model.line_profile = model.line_profile_bell
model.intensity_func = intensity_powerlaw_rout

#If not redefined, intensity and linewidth are powerlaws 
 #of R and z by default, whereas lineslope is constant.
  #See Table 1 of discminer paper 1.

#Useful definitions for plots
xmax = model.skygrid['xmax'] 
xlim = 1.15*xmax/au_to_m
extent= np.array([-xmax, xmax, -xmax, xmax])/au_to_m
  
#**************
#PROTOTYPE PARS
#**************
best_fit_pars = np.loadtxt('./log_pars_%s_cube_%dwalkers_%dsteps.txt'%(tag, nwalkers, nsteps))[1]
Mstar, vsys, incl, PA, xc, yc, I0, p, q, L0, pL, qL, Ls, pLs, z0_upper, p_upper, Rb_upper, q_upper, z0_lower, p_lower, Rb_lower, q_lower = best_fit_pars

model.params['velocity']['Mstar'] = Mstar
model.params['velocity']['vel_sign'] = -1
model.params['velocity']['vsys'] = vsys
model.params['orientation']['incl'] = incl
model.params['orientation']['PA'] = PA
model.params['orientation']['xc'] = xc
model.params['orientation']['yc'] = yc
model.params['intensity']['I0'] = I0/downsamp_factor #Jy/pix
model.params['intensity']['p'] = p
model.params['intensity']['q'] = q
model.params['intensity']['Rout'] = Rmax.to('au').value
model.params['linewidth']['L0'] = L0 
model.params['linewidth']['p'] = pL
model.params['linewidth']['q'] = qL
model.params['lineslope']['Ls'] = Ls
model.params['lineslope']['p'] = pLs
model.params['height_upper']['z0'] = z0_upper
model.params['height_upper']['p'] = p_upper
model.params['height_upper']['Rb'] = Rb_upper
model.params['height_upper']['q'] = q_upper
model.params['height_lower']['z0'] = z0_lower
model.params['height_lower']['p'] = p_lower
model.params['height_lower']['Rb'] = Rb_lower
model.params['height_lower']['q'] = q_lower

#**************************
#MAKE MODEL (2D ATTRIBUTES)
#**************************
modelcube = model.make_model(make_convolve=True) #Returns model cube and computes disc coordinates projected on the sky
modelcube.convert_to_tb()
datacube.convert_to_tb()

#**********************
#VISUALISE CHANNEL MAPS
#**********************
modelcube.show(compare_cubes=[datacube], extent=model.extent, int_unit='Intensity [K]', show_beam=True, surface_from=model)
modelcube.show_side_by_side(datacube, extent=model.extent, int_unit='Intensity [K]', show_beam=True,  surface_from=model)

#**********************
#MAKE MOMENT MAPS
#**********************
moments_nopriors = datacube.make_moments(method='doublegaussian') #Do not use model priors
moments_gauss = datacube.make_moments(model=model, method='doublegaussian') #Use model priors
moments_bell = datacube.make_moments(model=model, method='doublebell') #Use model priors + bell kernel

#**************************
#MAKE PLOT

cmap_vel = get_discminer_cmap('velocity')
kwargs_im = dict(cmap=cmap_vel, extent=extent, levels=np.linspace(-4.0, 4.0, 48)+vsys)
kwargs_cc = dict(colors='k', linestyles='-', extent=extent, levels=np.linspace(-2, 2, 9)+vsys, linewidths=0.4)
kwargs_cbar = dict(orientation='horizontal', pad=0.03, shrink=0.95, aspect=15)

def make3panels(props, titles=['', '', '']):

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15,6))
    ax_cbar0 = fig.add_axes([0.15, 0.14, 0.450+0.292, 0.04])

    im0 = ax[0].contourf(props[0], extend='both', **kwargs_im)
    im1 = ax[1].contourf(props[1], extend='both', **kwargs_im)
    im2 = ax[2].contourf(props[2], extend='both', **kwargs_im)

    cc0 = ax[0].contour(props[0], **kwargs_cc)
    cc1 = ax[1].contour(props[1], **kwargs_cc)
    cc2 = ax[2].contour(props[2], **kwargs_cc)
    
    cbar0 = plt.colorbar(im0, cax=ax_cbar0, format='%.1f', **kwargs_cbar)
    cbar0.ax.tick_params(labelsize=12) 
    
    mod_nticks_cbars([cbar0], nbins=8)
    
    ax[0].set_ylabel('Offset [au]', fontsize=15)
    for i in range(3):
        ax[i].set_title(titles[i], pad=40, fontsize=17)
    
    cbar0.set_label(r'Centroid Velocity [km s$^{-1}$]', fontsize=14)
    
    for axi in ax:
        make_up_ax(axi, xlims=(-xlim, xlim), ylims=(-xlim, xlim), labelsize=11)
        mod_major_ticks(axi, axis='both', nbins=8)
        datacube.plot_beam(axi, fc='lime')
        axi.set_aspect(1)

        model.make_emission_surface(
            axi,
            kwargs_R={'colors': '0.4', 'linewidths': 0.4},
            kwargs_phi={'colors': '0.4', 'linewidths': 0.3}
        )
        model.make_disc_axes(axi)
    
    for axi in ax[1:]: axi.tick_params(labelleft=False)

    return fig, ax

fig, ax = make3panels([moments_nopriors[0][1], moments_gauss[0][1], moments_bell[0][1]],
                      ['Upper - no priors', 'Upper - Gaussian', 'Upper - Bell'])
    
plt.savefig('twocomponent_centroids_upper.png', bbox_inches='tight', dpi=200)
plt.show()
plt.close()

fig, ax = make3panels([moments_nopriors[2][1], moments_gauss[2][1], moments_bell[2][1]],
                      ['Lower - no priors', 'Lower - Gaussian', 'Lower - Bell'])
    
plt.savefig('twocomponent_centroids_lower.png', bbox_inches='tight', dpi=200)
plt.show()
plt.close()

sys.exit()

#SINGLE GAUSSIAN MOMENTS:
moments_data = datacube.make_moments(method='gaussian', tag='data_'+tag) #Fit single Gaussian
moments_model = modelcube.make_moments(method='gaussian', tag='model_'+tag)

