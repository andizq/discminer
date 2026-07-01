from discminer.mining_control import _mining_residuals_filaments
import discminer.cart as cart

from discminer.mining_utils import (init_data_and_model,
                                    make_and_save_filaments,                                    
                                    get_2d_plot_decorators,
                                    get_noise_mask,
                                    load_moments,
                                    load_disc_grid,
                                    mark_planet_location,
                                    show_output)

from discminer.plottools import (make_round_map,
                                 make_substructures,
                                 make_up_ax,
                                 make_1d_legend,                                 
                                 mod_major_ticks,
                                 use_discminer_style)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

import json

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_residuals_filaments(None)
    args = parser.parse_args()

spids = np.asarray(args.spiral_ids)
spirals_pos = spids[spids>0]
spirals_neg = spids[spids<0]

#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

Rout = best['intensity']['Rout']
incl = best['orientation']['incl']
PA = best['orientation']['PA']
xc = best['orientation']['xc']
yc = best['orientation']['yc']

gaps = custom['gaps']
rings = custom['rings']
kinks = custom['kinks']

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment, unit_simple=True, fmt_vertical=True)

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc
Rmax = 1.1*Rout*u.au #Max model radius, 10% larger than disc Rout

#********************
#LOAD DATA AND GRID
#********************
datacube, model = init_data_and_model()

with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
xlim = 1.15*np.min([xmax, Rmax.value])
extent= np.array([-xmax, xmax, -xmax, xmax])

#*************************
#LOAD DISC GEOMETRY
R, phi, z = load_disc_grid()

noise_mean, mask = get_noise_mask(datacube, thres=2,
                                  mask_phi={'map2d': np.degrees(phi['upper']),
                                            'lims': args.mask_phi},
                                  mask_R={'map2d': R['upper']/au_to_m,
                                          'lims': args.mask_R}
)

RR = R[args.surface]/au_to_m
PP = phi[args.surface]

Xproj = RR*np.cos(phi[args.surface])
Yproj = RR*np.sin(phi[args.surface])

#*************************
#LOAD AND CLIP MOMENT MAPS    
moment_data, moment_model, residuals, mtags = load_moments(
    args,
    mask=mask,
    clip_Rmin=0.0*u.au,
    clip_Rmax=args.Router*Rout*u.au,
    clip_Rgrid=R[args.surface]*u.m
)

#******************************
#INIT MODEL AND MAKE FILAMENTS
#******************************
filtag = mtags['base']+'_'+args.projection
fil_pos_obj, fil_neg_obj, fil_pos_list, fil_neg_list, colors_dict = make_and_save_filaments(residuals, tag=filtag, return_all=True)
fil_pos = fil_pos_list[0]
fil_neg = fil_neg_list[0]

#***********
#MAKE PLOTS
#***********
clabels = {
    'linewidth': r'$\Delta$ Line width [km s$^{-1}$]',
    'lineslope': r'$\Delta$ Line slope',
    'velocity': r'$\Delta$ Centroid [km s$^{-1}$]',
    'v0phi': r'$\Delta$ Centroid [km s$^{-1}$]',
    'v0r': r'$\Delta$ Centroid [km s$^{-1}$]',
    'v0z': r'$\Delta$ Centroid [km s$^{-1}$]',
    'vr_leftover': r'$\Delta$ Centroid [km s$^{-1}$]',        
    'peakintensity': r'$\Delta$ Peak Int. [K]'
}

try:
    ylabel = clabels[args.moment]
except KeyError:
    ylabel = f'{args.moment}'
    
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(14, 7))

axr = fig.add_axes([0.53, 0.05, 0.43, 2*0.43])
ax[0,1].set_visible(False)
ax[1,1].set_visible(False)

axf = ax[0,0] #fit in polar coords
axp = ax[1,0] #pitch angle

#FIT SPIRALS
sp_lin = lambda x, a, b: a + b*x
sp_log = lambda x, a, b: a*np.exp(b*x)

def make_savgol(prof):
    tmp = len(prof)
    wl = tmp if tmp%2 else tmp-1

    try:
        ysav = savgol_filter(prof, wl, 2)
        ysav_deriv = savgol_filter(prof, wl, 2, deriv=1)
    except np.linalg.LinAlgError:
        ysav = prof
        ysav_deriv = None

    return ysav, ysav_deriv

def sort_xy(x, xp, yp):
    #Sort xp, yp based on input x 
    ind = np.argsort(x)
    return xp[ind], yp[ind]

def clean_filament_xy(xp, yp):
    xp, yp = sort_xy(xp, xp, yp)
    dx = np.abs(xp[1:] - xp[:-1])
    ind = np.append([True], [dx > 0.5*np.median(dx)])
    xp = xp[ind]
    yp = yp[ind]
    
    def shift(xp, delta, ind):
        indp = np.arange(1, len(xp))[ind] #Indices with high delta
        i = indp[np.argmax(delta[indp-1])] #Take index with highest delta
        
        left = xp[:i]
        right = xp[i:]

        if len(left)<=len(right):
            xp[:i] += 360
        else:
            xp[i:] -= 360
            
        return xp

    #Attempt to connect substructure split by phi-axis discontinuity
    dy = np.abs(yp[1:] - yp[:-1])    
    #Consider 80% smallest deltas for statistics
    iy = np.argsort(dy)[:-int(0.2*len(dy))] 
    indj = dy > np.nanmean(dy[iy]) + 10*np.nanstd(dy[iy]) 
    if np.sum(indj):
        xp = shift(xp, dy, indj)        
        xp, yp = sort_xy(xp, xp, yp)

    #Check dx too
    dx = np.abs(xp[1:] - xp[:-1])
    #Consider 80% smallest deltas for statistics        
    ix = np.argsort(dx)[:-int(0.2*len(dx))] 
    indc = dx > np.nanmean(dx[ix]) + 10*np.nanstd(dx[ix])     
    if np.sum(indc):
        xp = shift(xp, dx, indc)
        xp, yp = sort_xy(xp, xp, yp)            

    #Remove 10% outliers
    #iy = np.argsort(yp)[:-np.ceil(0.1*len(yp)).astype(int)]
    #xp, yp = sort_xy(xp[iy], xp[iy], yp[iy])

    return xp, yp

def fit_and_plot(ax, xp, yp, color=None, kind='positive'):
    ind, = np.where(np.isnan(xp) | np.isnan(yp) | (yp < args.Rinner*datacube.beam_size.value))
    xp = np.delete(xp, ind)
    xp_rad = np.radians(xp)
    yp = np.delete(yp, ind)    
        
    if kind in ['pos', 'positive']:
        if color is None:
            color = 'r'
        lin_c = 'magenta'
        log_c = 'k'
        sav_c = 'tomato'
    elif kind in ['neg', 'negative']:
        if color is None:
            color = 'b'
        lin_c = 'cyan'
        log_c = 'k'
        sav_c = 'dodgerblue'
    else:
        raise ValueError
   
    ax.scatter(xp, yp, s=20, color=color)
    x_ext = 20*np.linspace(-np.pi, np.pi, 500)
    
    #LINEAR FIT
    popt_lin, pcov_lin = curve_fit(sp_lin, xp_rad, yp)
    ylin = sp_lin(xp_rad, *popt_lin)    
    ylin_ext = sp_lin(x_ext, *popt_lin) #Linear extrapolation

    #LOGARITHMIC FIT
    popt_log, pcov_log = curve_fit(sp_log, xp_rad, yp, p0=[100, 0])
    ylog = sp_log(xp_rad, *popt_log)    
    ylog_ext = sp_log(x_ext, *popt_log) #Log extrapolation        
    
    #ysav, dy = make_savgol(yp)
    #ax.plot(xp, ysav, c=sav_c, ls='-', lw=2)

    #DERIVATIVES
    dylin = np.gradient(ylin, xp_rad)
    dylog = np.gradient(ylog, xp_rad)

    dylin_ext = np.gradient(ylin_ext, x_ext)
    dylog_ext = np.gradient(ylog_ext, x_ext)
    
    #*********
    #MAKE BINS    
    #Estimate optimal nbins
    # using Rice rule or Freedman-Diaconis rule nbins=2*iqr*n**(-1/3)
    #  NOT USED ATM
    r"""
    iqr = np.abs(np.quantile(xp, 0.75) - np.quantile(xp, 0.25)) 
    nbins = int(2*len(xp)**(1/3.)) #int(2*iqr*len(xp)**(-1/3.))

    from scipy import stats
    bin_stats = stats.binned_statistic
    xm, _, _ = bin_stats(xp, xp, statistic='mean', bins=nbins)
    ym, _, _ = bin_stats(xp, yp, statistic='mean', bins=nbins)
    ystd, _, _ = bin_stats(xp, yp, statistic='std', bins=nbins)        

    ind = ~np.isnan(xm)
    xm = xm[ind]
    ym = ym[ind]
    ystd = ystd[ind]

    #popt, pcov = curve_fit(sp_log, xm, ym, p0=[100, 0], sigma=ystd)     
    #ax.scatter(xm, ym, ec='k', fc='none', marker='o', lw=2, s=30)        
    #"""    
    #*********    

    if args.spiral_type == 'linear':
        ax.scatter(xp, ylin, ec='k', fc='none', marker='s', lw=0.3, s=15)        
        return ylin, dylin, ylin_ext, dylin_ext, popt_lin, popt_log
    else:
        ax.scatter(xp, ylog, ec='k', fc='none', marker='s', lw=0.3, s=15)        
        return ylog, dylog, ylog_ext, dylog_ext, popt_lin, popt_log

#***********
#LEFT PANELS
#***********
#POSITIVE SPIRALS
ipos, ineg = [], [] #Filament ids considered in the analysis
popt_lin_pos, popt_log_pos = [], []
popt_lin_neg, popt_log_neg = [], []

for i,fil in enumerate(fil_pos_list[2:]):
    if args.longpath:
        fil = fil * fil_pos_list[1] #1st element stores longpath global skeleton
    fp = fil.astype(bool)
    xp = np.degrees(PP[fp])
    yp = RR[fp]
    xp, yp = clean_filament_xy(xp, yp)
    yfp, dyp, yp_ext, dyp_ext, popt_lin, popt_log = fit_and_plot(axf, xp, yp, color=colors_dict[i+1])
    pitchp_ext = np.degrees(np.arctan(np.abs(dyp_ext)/yp_ext))
    pitchp = np.degrees(np.arctan(np.abs(dyp)/yfp))

    popt_lin_pos.append(popt_lin)
    popt_log_pos.append(popt_log)
    
    if np.sum(fp)>=10:
        axp.plot(yp_ext, pitchp_ext, lw=2.5, ls='--', c=colors_dict[i+1])    
        axp.plot(yfp, pitchp, lw=4.0, c=colors_dict[i+1])
        ipos.append(i)

#NEGATIVE SPIRALS        
for i,fil in enumerate(fil_neg_list[2:]):
    if args.longpath:
        fil = fil * fil_neg_list[1]
    fn = fil.astype(bool)
    xn = np.degrees(PP[fn])
    yn = RR[fn]
    xn, yn = clean_filament_xy(xn, yn)
    yfn, dyn, yn_ext, dyn_ext, popt_lin, popt_log = fit_and_plot(axf, xn, yn, color=colors_dict[-i-1], kind='neg')
    pitchn_ext = np.degrees(np.arctan(np.abs(dyn_ext)/yn_ext))    
    pitchn = np.degrees(np.arctan(np.abs(dyn)/yfn))

    popt_lin_neg.append(popt_lin)
    popt_log_neg.append(popt_log)
    
    if np.sum(fn)>=10:
        axp.plot(yn_ext, pitchn_ext, lw=2.5, ls='--', c=colors_dict[-i-1])    
        axp.plot(yfn, pitchn, lw=4.0, c=colors_dict[-i-1])
        ineg.append(i)
        
axp.plot([None], [None], lw=4.0, ls='-', c='k', label='Measurement')        
axp.plot([None], [None], lw=2.5, ls='--', c='k', label='Extrapolation')
make_1d_legend(axp, fontsize=11)

mod_major_ticks(axp, axis='y', nbins=5)
axp.set_xlim(args.Rinner*datacube.beam_size.value, args.Router*Rout)
#axp.set_ylim(-9, 61)
axp.set_ylim(-3, 43)
axp.set_xlabel('Radius [au]', labelpad=5)
axp.set_ylabel('Pitch angle [deg]')
make_substructures(axp, gaps=gaps, rings=rings, label_gaps=True, label_rings=True)  
mod_major_ticks(axp, axis='x', nbins=10)
mod_major_ticks(axp, axis='y', nbins=5)

axf.set_ylim(0, None)
axf.set_ylim(args.Rinner*datacube.beam_size.value, args.Router*Rout)
axf.set_xlabel('Azimuth [deg]', labelpad=5)
axf.set_ylabel('Radius [au]')
axf.set_xticks(np.arange(-270,270+1,90))
for ang in [-90, 90, -270, 270]:
    axf.axvline(ang, ls=':', lw=2.5, color='0.3', dash_capstyle='round')
axf.set_xlim(-1*200, 1*200)
#axf.set_xlim(-1*181, 1*181)
    
axf.plot([None], [None], ls=':', lw=2.5, color='0.3', dash_capstyle='round', label='Minor axis')
make_1d_legend(axf)

#***********
#RIGHT PANEL
#***********
if args.projection=='cartesian':
    levels_resid = np.linspace(-clim, clim, 32)
    
    if args.surface in ['up', 'upper']:
        z_func = model.z_upper_func
        z_pars = best['height_upper']

    elif args.surface in ['low', 'lower']:
        z_func = model.z_lower_func
        z_pars = best['height_lower']
    
    fig, axr = make_round_map(residuals, levels_resid, Xproj*u.au, Yproj*u.au, args.Router*Rout*u.au,
                              fig=fig, ax=axr, 
                              z_func=z_func, z_pars=z_pars, incl=incl, PA=PA, xc=xc, yc=yc,
                              cmap=cmap_res, clabel=unit, fmt=cfmt, 
                              gaps=gaps, rings=rings, kinks=kinks,
                              fontsize_radial_grid=14, fontsize_azimuthal_grid=13, fontsize_cbar=13, fontsize_xaxis=13, fontsize_nskyaxis=15,
                              make_cbar=False, make_xaxis=True*0, make_Rout_proj=True*0,
                              mask_inner=args.Rinner*datacube.beam_size,
                              kwargs_mask={'zorder': 30})
                              #kwargs_mask={'facecolor': '0.6', 'alpha': 0.4})
    
    make_substructures(axr, gaps=gaps, rings=rings, kinks=kinks, twodim=True, label_rings=True*0, label_kinks=True*0, sposy=-1)

    for i,fil in enumerate(fil_pos_list[2:]):
        if args.longpath:
            fil = fil * fil_pos_list[1]        
        axr.contour(Xproj, Yproj, fil, linewidths=1.0, colors=colors_dict[i+1], alpha=1, zorder=13)
        axr.text(Xproj[fil.astype(bool)][0], Yproj[fil.astype(bool)][0], i+1, fontsize=14, c='k',
                 ha='center', va='center', weight='bold', bbox=dict(facecolor='white', edgecolor='red'), zorder=30)
        
    for i,fil in enumerate(fil_neg_list[2:]):
        if args.longpath:
            fil = fil * fil_neg_list[1]
        axr.contour(Xproj, Yproj, fil, linewidths=1.0, colors=colors_dict[-i-1], alpha=1, zorder=13)
        axr.text(Xproj[fil.astype(bool)][0], Yproj[fil.astype(bool)][0], -i-1, fontsize=14, c='k',
                 ha='center', va='center', weight='bold', bbox=dict(facecolor='white', edgecolor='blue'), zorder=30)
        
    phi_ext = 4*np.linspace(-np.pi, np.pi, 500)

    if args.spiral_type == 'linear':
        sp_func = sp_lin
        popt_pos = popt_lin_pos
        popt_neg = popt_lin_neg
        header_txt = 'a\tb\tcolor\tsign\tid --> Linear fit: R = a + b*phi'
    else:
        sp_func = sp_log
        popt_pos = popt_log_pos
        popt_neg = popt_log_neg        
        header_txt = 'a\tb\tcolor\tsign\tid --> Logarithmic fit: R = a*np.exp(b*phi)'
        
    arr_write = []    
    for k in spirals_pos: #>0 ids
        R_ext = sp_func(phi_ext, *popt_pos[k-1])
        ind = (R_ext > 0) & (R_ext < args.Router*Rout)
        xplot = R_ext[ind]*np.cos(phi_ext[ind])
        yplot = R_ext[ind]*np.sin(phi_ext[ind])
        axr.plot(xplot, yplot, lw=2,  color=colors_dict[k], zorder=20)
        axr.plot(xplot, yplot, lw=5, color='k', zorder=19)

        arr_write.append(np.append(np.round(popt_pos[k-1], 6), [colors_dict[k], 'pos', k]))
        
    for k in spirals_neg: #<0 ids
        R_ext = sp_func(phi_ext, *popt_neg[abs(k)-1])
        ind = (R_ext > 0) & (R_ext < args.Router*Rout)
        xplot = R_ext[ind]*np.cos(phi_ext[ind])
        yplot = R_ext[ind]*np.sin(phi_ext[ind])        
        axr.plot(xplot, yplot, lw=2,  color=colors_dict[k], zorder=20)
        axr.plot(xplot, yplot, lw=5, color='k', zorder=19)

        arr_write.append(np.append(np.round(popt_neg[abs(k)-1], 6), [colors_dict[k], 'neg', k]))

    if len(arr_write)>0:
        arr_write = np.asarray(arr_write)
        np.savetxt(
            'spirals_fit_parameters_%s_%s.txt'%(mtags['base'], args.spiral_type),
            arr_write,
            fmt='%s',
            header=header_txt,
            delimiter='\t'
        )
        
axr.set_title(ctitle, fontsize=16, color='k')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

mark_planet_location(axr, args, facecolors='none', edgecolors='k', lw=4.5, s=650, coords='disc', model=model)    

plt.savefig('residuals_filaments_%s_%s_%s.png'%(mtags['base'], args.projection, args.spiral_type), bbox_inches='tight', dpi=200)
show_output(args)

#**************
#WIDTH ANALYSIS
#**************
def filament_radprof(ax_cols, filaments, colors=None, idsign=1):
    data = []    
    for i in range(ncols):
        axi = ax_cols[i]            
        if i >= len(filaments):
            axi.set_visible(False)
            continue
        fili = filaments[i]
        try:
            fili.plot_radial_profile(xunit=u.au, ax=axi, show_plot=False)
        except:
            print('Radial profile fit failed for Filament id #%d'%((i+1)*idsign))
            axi.set_visible(False)

        lines = axi.get_lines()            

        width, error = fili.radprof_fwhm(u.au)
        for line in lines:
            if line.get_color()=='k' and colors is not None:
                line.set_markerfacecolor(colors[i])
                line.set_markeredgecolor('k')
                line.set_markersize(7)                
                data.append(np.max(line.get_data()[1]))

            else:
                line.set_linewidth(4.5)
                line.set_color('0.5')
                line.set_alpha(0.5)
                line.set_label('FWHM: %.1f +/-  %.1f au'%(width.value, error.value))
        axi.grid(ls='--')
        make_up_ax(axi, labelbottom=True, labeltop=False, labelsize=11)
        make_1d_legend(axi, fontsize=12)
        
    data_max = np.nanmax(data)
    for i in range(ncols): 
        axi = ax_cols[i]
        axi.set_ylim(-np.round(0.05*data_max, 2), np.round(1.2*data_max, 2))
        axi.text(0.92, 0.92, r'%d'%((i+1)*idsign), ha='center', va='center', fontsize=18, fontweight='bold', transform=axi.transAxes)

if args.make_filwidth:
    fil_pos_obj.find_widths()
    fil_neg_obj.find_widths()

    if args.writeobj:
        for i,fil in enumerate(fil_pos_obj.filaments):
            fil.to_pickle('filaments_pos_%s_id%d.pkl'%(filtag, i+1))
        for i,fil in enumerate(fil_neg_obj.filaments):
            fil.to_pickle('filaments_neg_%s_id%d.pkl'%(filtag, i+1))
    
    ncols = np.max([len(ipos), len(ineg)]).astype(int)

    fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(4*ncols, 8))
    ax[0] = np.atleast_1d(ax[0])
    ax[1] = np.atleast_1d(ax[1])
    
    filament_radprof(ax[0], np.asarray(fil_pos_obj.filaments)[ipos], colors=[colors_dict[i+1] for i in ipos], idsign=1)
    filament_radprof(ax[1], np.asarray(fil_neg_obj.filaments)[ineg], colors=[colors_dict[-i-1] for i in ineg], idsign=-1)

    ax00 = ax[0][0]
    x0, x1 = ax00.get_xlim()
    y0, y1 = ax00.get_ylim()
    xp = 0.5*(x0+x1)
    yp = 0.7*(y0+y1)
    dx = datacube.beam_size.value #/(x1-x0)
    ax00.text(0.5 + 0.5*dx/(x1-x0), 0.75, 'beam (%d au)'%datacube.beam_size.value, ha='center', fontsize=12, transform=ax00.transAxes)
    ax00.plot([xp, xp+dx], [yp, yp], ls='-', lw=4.5, color='k')

    if ncols==1:
        for axi in ax:
            axi[0].set_ylabel(ylabel, fontsize=14)

    else:
        for axi in ax[:,0]:
            axi.set_ylabel(ylabel, fontsize=14)

        for i in range(2):
            for axi in ax[i,1:]:
                axi.set_ylabel(None)        
                axi.tick_params(labelleft=False)
            
    for axi in ax[0]:
        axi.set_xlabel(None)    
        axi.tick_params(labelsize=13)
    
    for axi in ax[1]:
        axi.set_xlabel('')
        axi.tick_params(labelsize=13)    
    ax[1][0].set_xlabel('Distance from spine [au]', fontsize=14)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.25)

    plt.savefig('residuals_filaments_widths_%s_%s_%s.png'%(mtags['base'], args.projection, args.spiral_type), bbox_inches='tight', dpi=200)
    show_output(args)
