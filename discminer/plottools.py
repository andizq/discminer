"""
plottools module
===========
"""
import os
import copy
import matplotlib
import numpy as np
from math import ceil
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections.abc import Iterable

from .tools.utils import weighted_std, InputError
from .tools.fit_kernel import _gauss
from .grid import GridTools

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 22

_residuals_colors = {
    'velocity': ["#"+tmp for tmp in ["000b14","004f8f","1f9aff","fff7db","ff5a47","cc0033","140200"]],
    'linewidth': ["#"+tmp for tmp in ["000b14","036407","81ae71","fff7db","ff5a47","cc0033","140200"]],
    'intensity': ["#"+tmp for tmp in ["000b14","382406","6a593c","fff7db","ff5a47","cc0033","140200"]]
}

_residuals_cranges = {
    'velocity': None,
    'linewidth': None,
    'intensity': None,
}

_attribute_colors = {
    'velocity': ["#"+tmp for tmp in ["000b14","004f8f","1f9aff","fff7db","ff5a47","cc0033","140200"]],
    'linewidth': ["#"+tmp for tmp in ["001219","005f73","0a9396","94d2bd","e9d8a6","ee9b00","ca6702","bb3e03","ae2012","9b2226"]],
    'intensity': "terrain_r",
}

_attribute_cranges = {
    'velocity': None,
    'linewidth': None,
    'intensity': "matplotlib",    
}

def use_discminer_style():
    tools = os.path.dirname(os.path.realpath(__file__))+'/tools/'
    plt.style.use(tools+'discminer.mplstyle') 

#*************
#AX DECORATORS
#*************
def mod_nticks_cbars(cbars, nbins=5):
    for cb in cbars:
        cb.locator = matplotlib.ticker.MaxNLocator(nbins=nbins)
        cb.update_ticks()
        
def mod_major_ticks(ax, axis='both', nbins=6):
    ax.locator_params(axis=axis, nbins=nbins)
    
def mod_minor_ticks(ax):
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2)) #1 minor tick per major interval
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

def make_up_ax(ax, xlims=(None, None), ylims=(None, None), 
               mod_minor=True, mod_major=True, **kwargs_tick_params):
    kwargs_t = dict(labeltop=True, labelbottom=False, top=True, right=True, which='both', direction='in')
    kwargs_t.update(kwargs_tick_params)
    if mod_major: mod_major_ticks(ax)
    if mod_minor: mod_minor_ticks(ax)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.tick_params(**kwargs_t)

#**********************
#COLORMAP CUSTOMISATION
#**********************
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_cmap_from_color(color, lev=3):
    cmap = matplotlib.colors.to_rgba(color)
    newcolors = np.tile(cmap, lev).reshape(lev,4) #Repeats the colour lev times
    newcolors[:,-1] = np.linspace(0.25, 0.95, lev) #Modifies alpha only
    new_cmap = ListedColormap(newcolors)
    return new_cmap

def mask_cmap_interval(cmap, cmap_lims, mask_lims, mask_color=np.ones(4), append=False):
    if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
    cmap0, cmap1 = cmap_lims
    mask0, mask1 = mask_lims
    c0 = (mask0-cmap0)/(cmap1-cmap0)
    c1 = (mask1-cmap0)/(cmap1-cmap0)
    id0 = int(round(c0*(cmap.N)))
    id1 = int(round(c1*(cmap.N)))
    new_cmap = copy.copy(cmap)
    new_cmap._init()
    """#This block does not work, mpl does not know where to put the newly added colors
    if append:
       mask_color_arr = np.broadcast_to(mask_color, (id1-id0, 4))
       new_cmap._lut = np.insert(new_cmap._lut, id0, mask_color_arr, axis=0)
       new_cmap.N = cmap.N + id1-id0
       #Next line redoes the continuous linearsegmented colormap, thus the masked color block is reduced to a single color  
       #new_cmap = new_cmap._resample(new_cmap.N) 
    """
    new_cmap._lut[id0:id1,:] = mask_color 
    return new_cmap

def get_continuous_cmap(hex_list, float_list=None):                                                                               
    """
    Taken from https://github.com/KerryHalupka/custom_colormap 
 
    Creates and returns a color map that can be used in heat map figures.                                                             
    If float_list is not provided, the color map returned is a homogeneous gradient of the colors in hex_list.
    If float_list is provided, each color in hex_list is set to start at the corresponding location in float_list. 

    Parameters                                                                                        
    ----------                                                                                          
    hex_list: list of hex code strings                                                                
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns     
    ----------
    matplotlib cmap

    Examples
    ----------
    fig, ax = plt.subplots(1,1)
    hex_list = ['#0091ad', '#fffffc', '#ffd166']
    x, y = np.mgrid[-5:5:0.05, -5:5:0.05]                                
    z = (np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2))
    im = ax.imshow(z, cmap=get_continuous_cmap(hex_list))                                                         
    fig.colorbar(im)  
    ax.yaxis.set_major_locator(plt.NullLocator()) # remove y axis ticks 
    ax.xaxis.set_major_locator(plt.NullLocator()) # remove x axis ticks
    plt.show()
    """

    rgb_list = [matplotlib.colors.to_rgb(i) for i in hex_list]
    if float_list is None: float_list = np.linspace(0,1,len(rgb_list))

    cdict = dict()                                                                                
    for num, col in enumerate(['red', 'green', 'blue']):                                               
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
        cmap_new = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmap_new

def get_discminer_cmap(observable, kind='attribute'):
    if kind=='attribute':
        colors = _attribute_colors[observable]
        cranges = _attribute_cranges[observable]
    elif kind=='residuals':
        colors = _residuals_colors[observable]
        cranges = _residuals_cranges[observable]
    else:
        raise InputError(
            kind, "kind must be 'attribute' or 'residuals'"
        )

    if cranges=='matplotlib':
        cmap = copy.copy(plt.get_cmap(colors))
    elif isinstance(cranges, Iterable) or cranges is None:
        cmap = get_continuous_cmap(colors, float_list=cranges)
        
    return cmap
        
#**************
#COLORBAR STUFF
#**************
def add_cbar_ax(fig, ax, perc=4, orientation='horizontal', subplots=True):
    figx, figy = fig.get_size_inches()
    figr = figy/figx
    axp = ax.get_position()
    x0, x1, y0, y1 = axp.x0, axp.x1, axp.y0, axp.y1
    w = x1 - x0
    h = y1 - y0
    dx = dy = 0.01*perc
    
    if subplots:
        if orientation=='horizontal': return fig.add_axes([x0, y0, w, dy])
        if orientation=='vertical': return fig.add_axes([x1+0.5*dx, y0+0.1*h, dx, h-0.2*h])        
    else:
        if orientation=='horizontal': return fig.add_axes([x0, y0-0.5*dy, w, dy])
        if orientation=='vertical': return fig.add_axes([x1+0.5*dx, y0, dx, h])        
        
def make_round_cbar(ax, Rout, levels,
                    rwidth=0.06, cmap=get_discminer_cmap('velocity'),
                    quadrant=2, clabel='km/s', fmt='%4.1f'):

    sign_xy = {1: [1,1],
               2: [-1,1],
               3: [-1,-1],
               4: [1,-1],
    }[quadrant]
    
    phi0 = {1: 0.0,
            2: np.pi,
            3: -2*np.pi/3,
            4: -np.pi/3,
    }[quadrant]

    phi1 = {1: np.pi/3,
            2: 2*np.pi/3,
            3: -np.pi,
            4: 0.0,
    }[quadrant]

    rot_quad = {1: lambda phi_lev: np.degrees(phi0+phi_lev),
                2: lambda phi_lev: -np.degrees(phi0-phi_lev),
                3: lambda phi_lev: -np.degrees(phi1-phi_lev),
                4: lambda phi_lev: np.degrees(phi1+phi_lev),
    }[quadrant]

    cticks_ha = {1: 'left',
                 2: 'right',
                 3: 'right',
                 4: 'left'
    }[quadrant]

    text_R = {1: 1}
    
    phi_max = np.max([phi0,phi1])
    phi_min = np.min([phi0,phi1])
    
    xlims = ax.get_xlim()
    xcbar = np.linspace(*xlims, 1000)
    ycbar = np.linspace(-Rout, Rout, 1000)
    xx, yy = np.meshgrid(xcbar, ycbar)
    rr = np.hypot(xx, yy)
    pp = np.arctan2(yy, xx)

    cbar_phi2lev = lambda phi: (levels[-1]-levels[0])*np.abs((phi0-phi)/(phi1-phi0)) + levels[0]
    cbar_lev2phi = lambda lev: (phi1-phi0)*(lev-levels[0])/np.abs(levels[-1]-levels[0]) + phi0

    r0 = Rout+0.05*Rout
    r1 = r0+rwidth*Rout
    cbar_polar = np.zeros_like(rr) + np.nan
    mask_polar = (rr>=r0) & (rr<=r1) & (pp>=phi_min) & (pp<=phi_max)
    cbar_polar[mask_polar] = cbar_phi2lev(pp[mask_polar])
    cbar_im = ax.contourf(xx,yy, cbar_polar, levels=levels, cmap=cmap, extend='both', origin='lower') #np.ma.array(cbar_polar, mask=~mask_polar )

    cbar_levels_pol = np.linspace(levels[0], levels[-1], 5)
    cbar_levels_phi = cbar_lev2phi(cbar_levels_pol)

    for i,cbi in enumerate(cbar_levels_phi):
        Rtext = r1 + 0.03*Rout
        ax.text(Rtext*np.cos(cbi), Rtext*np.sin(cbi), fmt%cbar_levels_pol[i], fontsize=SMALL_SIZE+2, c='0.7',
                ha=cticks_ha, va='center', weight='bold', rotation_mode='anchor', rotation=rot_quad(cbi))

    ax.text(sign_xy[0]*Rtext, -sign_xy[1]*0.1*Rout, clabel, fontsize=SMALL_SIZE+3, c='0.7', ha='center', va='center', weight='bold', rotation=0)

#********************
#DISC PLOT DECORATORS
#********************
def _make_text_2D(ax, Rlist, posx=0.0, sposy=1, fmt='%d', va=None, **kwargs_text):
    if va is not None:
        _va = va
        dy = 0
    else:
        if sposy<0:
            _va = 'top'
            dy = 2
        elif sposy>0:
            _va = 'bottom'
            dy = -1
        else:
            return 0

    kwargs = dict(fontsize=SMALL_SIZE+3, ha='center', va=_va, weight='bold', zorder=20, rotation=0)
    kwargs.update(kwargs_text)
    for Ri in Rlist:
        ax.text(posx, sposy*(Ri+dy), fmt%Ri, **kwargs)
        
def _make_text_1D_substructures(ax, gaps=[], rings=[], kinks=[],
                                label_gaps=False, label_rings=False, label_kinks=False):
    kwargs_text = dict(fontsize=SMALL_SIZE+2, ha='center', va='bottom', transform=ax.transAxes, weight='bold', rotation=90)    
    xlims = ax.get_xlim()
    xext = xlims[1]-xlims[0]

    def text_it(R, text):
        for Ri in R:
            if Ri < xlims[0]: continue
            posx = (Ri-xlims[0])/xext
            ax.text(posx, 1.02, text%Ri, **kwargs_text)        

    if label_gaps: text_it(gaps, r'D%d')
    if label_rings: text_it(rings, r'B%d')
    if label_kinks: text_it(kinks, r'K%d')

def _make_radial_grid_2D(ax, Rout, gaps=[], rings=[], kinks=[], make_labels=True, label_freq=2):
    get_intdigits = lambda n: len(str(n).split('.')[0])
    
    angs = np.linspace(0, 2*np.pi, 100)
    cos_angs = np.cos(angs)
    sin_angs = np.sin(angs)
    
    subst = np.concatenate([gaps, rings, kinks], axis=None)
    if len(subst)>0:
        Rref = np.max(subst)
    else:
        Rref = 0.0
    Rref_digits = get_intdigits(Rref)        
        
    R_after_ref = 10**(Rref_digits-1)*ceil(Rref/(10**(Rref_digits-1)))
    Rgrid_polar = np.arange(R_after_ref, Rout, 50)
    
    for j,Ri in enumerate(Rgrid_polar[0:-1:2]):
        if Ri == Rref: continue
        ax.plot(Ri*cos_angs, Ri*sin_angs, color='k', lw=1.2, alpha=0.5,
                dash_capstyle='round', dashes=(0.5, 3.5)) 

    for j,Ri in enumerate(Rgrid_polar[1::2]):
        ax.plot(Ri*cos_angs, Ri*sin_angs, color='k', ls=':', lw=0.4, alpha=1.0)

    if make_labels:
        _make_text_2D(ax, Rgrid_polar[1::label_freq], sposy=-1, fmt='%d',
                      fontsize=SMALL_SIZE+3, color='0.1', va='center') #label in the north
        _make_text_2D(ax, Rgrid_polar[1::label_freq], sposy=1, fmt='%d',
                      fontsize=SMALL_SIZE+3, color='0.1', va='center') # and south

    ax.plot(0.98*Rout*cos_angs, 0.98*Rout*sin_angs, color='0.4', ls='-', lw=3.0, alpha=1.0)
    ax.plot(0.99*Rout*cos_angs, 0.99*Rout*sin_angs, color='0.2', ls='-', lw=3.0, alpha=1.0)
    ax.plot(1.00*Rout*cos_angs, 1.00*Rout*sin_angs, color='0.0', ls='-', lw=3.0, alpha=1.0)
    
def _make_azimuthal_grid_2D(ax, Rout):
    for j, phii in enumerate(np.arange(0, 2*np.pi, np.pi/6)):
        ax.plot([0, Rout*np.cos(phii)], [0, Rout*np.sin(phii)], color='k', ls=':', lw=0.4, alpha=1.0)

    for deg in np.linspace(0, 90, 4):
        deg_rad = np.radians(deg)
        txt = ax.text(1.04*Rout*np.cos(deg_rad), 1.04*Rout*np.sin(deg_rad), r'$%d$'%deg, c='0.0',
                      fontsize=SMALL_SIZE, ha='center', va='center', weight='bold', rotation=-(90-deg))
        txt.set_text(r'$%d^{\circ}$'%deg)

def _make_nsky_2D(ax, Rout, xlim, z_func, z_pars, incl, PA, xc=0.0, yc=0.0):
    ynorth = np.linspace(-1.1*Rout, 1.1*Rout, 100)
    xnorth = np.zeros_like(ynorth)
    xn, yn = [], []
    
    for i in range(len(ynorth)):
        xni, yni = GridTools.get_disc_from_sky_coords(xnorth[i], ynorth[i], z_func, z_pars, incl, PA, xc=0, yc=0)
        xn.append(xni)
        yn.append(yni)

    xn, yn = np.asarray(xn), np.asarray(yn)
    ax.plot(xn, yn, color='0.0', lw=1.7, dash_capstyle='round', dashes=(1.5, 2.5))
    
    text_nsky = lambda x, y: ax.text(x, y, r'$\vec{\rm N}$$_{\rm sky}$', fontsize=MEDIUM_SIZE,
                                     ha='center', va='bottom', weight='bold', rotation=np.degrees(-PA))

    if yni>xni:
        ci = yni
        cn = yn
    else:
        ci = xni
        cn = xn
        
    if ci>xlim:
        ii = np.argmin(np.abs(cn-xlim))
        text_nsky(xn[ii], yn[ii])
    else:
        text_nsky(xni, yni)

    return xni, yni

def make_substructures(ax, gaps=[], rings=[], kinks=[],
                       twodim=False,
                       coords='disc', polar=False,
                       model=None, surface='upper', #relevant if coords='sky'
                       make_legend=False,                       
                       label_gaps=False, label_rings=False, label_kinks=False,
                       kwargs_gaps={}, kwargs_rings={}, kwargs_kinks={}, func1d='axvline'):
    
    '''Overlay ring-like (if twodim) or vertical lines (if not twodim) to illustrate the radial location of substructures in the disc'''
    kwargs_g = dict(color='0.2', ls='--', lw=1.7, dash_capstyle='round', dashes=(3.0, 2.5), alpha=1.0)
    kwargs_r = dict(color='0.2', ls='-', lw=1.7, dash_capstyle='round', alpha=1.0)
    kwargs_k = dict(color='purple', ls=':', lw=2.5, dash_capstyle='round', dashes=(0.5, 1.5), alpha=0.9)
    kwargs_g.update(kwargs_gaps)
    kwargs_r.update(kwargs_rings)
    kwargs_k.update(kwargs_kinks)
    
    if twodim:
        nphi = 100
        phi = np.linspace(0, 2*np.pi, nphi)
        phi_deg = np.degrees(phi-np.pi)
        subst_fmt = zip([gaps, rings, kinks],
                        ['D%d', 'B%d', 'K%d'],
                        [label_gaps, label_rings, label_kinks],
                        [kwargs_g['color'], kwargs_r['color'], kwargs_k['color']])

        if coords in ['disc', 'disk']:

            if polar:
                for R in gaps: ax.plot(phi_deg, [R]*nphi, **kwargs_g)
                for R in rings: ax.plot(phi_deg, [R]*nphi, **kwargs_r)
                for R in kinks: ax.plot(phi_deg, [R]*nphi, **kwargs_k)

                for subst, fmt, label, color in subst_fmt:
                    if label:
                        _make_text_2D(ax, subst, posx=-45, fmt=fmt, color=color)                
            else:
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                for R in gaps: ax.plot(R*cos_phi, R*sin_phi, **kwargs_g)
                for R in rings: ax.plot(R*cos_phi, R*sin_phi, **kwargs_r)
                for R in kinks: ax.plot(R*cos_phi, R*sin_phi, **kwargs_k)
                
                for subst, fmt, label, color in subst_fmt:
                    if label:
                        _make_text_2D(ax, subst, sposy=-1, fmt=fmt, color=color)
                        
        elif coords=='sky':
            if model is None:
                raise InputError(
                    model, "model must be a discminer model instance"
                )

            au_m = u.au.to('m')
            get_sky_from_disc_coords = GridTools.get_sky_from_disc_coords
            cont_gaps, cont_rings, cont_kinks = [], [], []

            incl = model.params['orientation']['incl']
            PA = model.params['orientation']['PA']
            xc = model.params['orientation']['xc']
            yc = model.params['orientation']['yc']
            orient = (incl, PA, xc, yc)
            
            get_z = lambda r: model.z_upper_func({'R': r*au_m}, **model.params['height_%s'%surface])/au_m
            get_0 = lambda r: 0

            if surface=='midplane':
                get_fz = get_0
            else:
                get_fz = get_z
                
            for gap in gaps:
                z = get_fz(gap)
                x_cont, y_cont,_ = get_sky_from_disc_coords(gap, phi, z, *orient)
                ax.plot(x_cont, y_cont, **kwargs_g)
                
            for ring in rings:
                z = get_fz(ring)
                x_cont, y_cont,_ = get_sky_from_disc_coords(ring, phi, z, *orient)
                ax.plot(x_cont, y_cont, **kwargs_r)
                
            for kink in kinks:
                z = get_fz(kink)
                x_cont, y_cont,_ = get_sky_from_disc_coords(kink, phi, z, *orient)
                ax.plot(x_cont, y_cont, **kwargs_k)
            
        else:
            raise InputError(
                coords, "coords must be 'disc' [or 'disk'] or 'sky'"
            )

    else:
        if func1d=='axvline': func1d=ax.axvline
        elif func1d=='axhline': func1d=ax.axhline            
        for R in gaps: func1d(R, **kwargs_g)
        for R in rings: func1d(R, **kwargs_r)
        for R in kinks: func1d(R, **kwargs_k)

        _make_text_1D_substructures(ax, gaps=gaps, rings=rings, kinks=kinks,
                                    label_gaps=label_gaps, label_rings=label_rings, label_kinks=label_kinks)
        
    if make_legend and len(gaps)>0: ax.plot([None], [None], label='Gaps', **kwargs_g)
    if make_legend and len(rings)>0: ax.plot([None], [None], label='Rings', **kwargs_r)
    if make_legend and len(kinks)>0: ax.plot([None], [None], label='Kinks', **kwargs_k)

    return ax

#*********************
#MAKE DEPROJECTED MAPS
#*********************
def make_round_map(
        map2d, levels, X, Y, Rout,
        z_func=None, z_pars=None, incl=None, PA=None, xc=0, yc=0, #Optional, make N-sky axis
        fig=None, ax=None,
        rwidth=0.06, cmap=get_discminer_cmap('velocity'), clabel='km/s', fmt='%5.2f', quadrant=None, #cbar kwargs
        gaps=[], rings=[], kinks=[],
        mask_wedge=None, mask_inner=None
):

    #SOME DEFINITIONS
    if fig is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))

    X = np.nan_to_num(X.to('au').value)
    Y = np.nan_to_num(Y.to('au').value)
    Rout = Rout.to('au').value
        
    xlim_rec = 1.15*Rout
    
    cmap_c = copy.copy(cmap)
    cmap_c.set_under('0.4')
    cmap_c.set_over('0.4')
    
    #MAIN PLOT
    im = ax.contourf(X, Y, map2d, levels=levels, cmap=cmap_c, extend='both', origin='lower')

    #RADIAL GRID
    _make_radial_grid_2D(ax, Rout, gaps=gaps, rings=rings, kinks=kinks, label_freq=2)
    
    #AZIMUTHAL GRID
    _make_azimuthal_grid_2D(ax, Rout)
    
    #SKY AXIS
    if np.all(np.asarray([z_func, z_pars, incl, PA])!=None):
        xni, yni = _make_nsky_2D(ax, Rout, xlim_rec,
                                 z_func, z_pars, incl, PA, xc=xc, yc=yc)
    else:
        xni, yni = None, None

    #SET LIMITS AND AXES
    for side in ['left','top','right']: ax.spines[side].set_visible(False)
    make_up_ax(ax,
               xlims=(-xlim_rec, xlim_rec),
               ylims=(-xlim_rec, xlim_rec),
               labelleft=False, left=False, right=False, labeltop=False, top=False, labelbottom=True, bottom=True,
               labelsize=SMALL_SIZE+3, rotation=45)
    mod_major_ticks(ax, axis='x', nbins=10)
    ax.set_xlabel('Offset [au]', fontsize=MEDIUM_SIZE+2)
    ax.set_aspect(1)

    #MAKE ROUND COLORBAR
    if quadrant is None: #Guess best quadrant based on Nsky position
        if xni is None:
            quadrant = 2
        elif (yni>0 and xni<0) or (yni<0 and xni>0):
            quadrant = 3
        else:
            quadrant = 2
            
    make_round_cbar(ax, Rout, levels, cmap=cmap_c, clabel=clabel, fmt=fmt, quadrant=quadrant)

    sq = {1: -1,
          2: 1,
          3: 1,
          4: -1,
    }[quadrant]
        
    ax.plot([sq*Rout, sq*Rout], [0, -xlim_rec], color='0.0', lw=1.0, dash_capstyle='round', dashes=(1.5, 2.5)) #Rout projected onto Cartesian xaxis

    #MASK
    kw_mask = dict(facecolor='0.5', edgecolor='k', lw=1.0, alpha=0.6)
    if mask_wedge is not None:

        for wedge in mask_wedge:

            if np.isscalar(wedge.value): #single wedge
                w = patches.Wedge((0,0), Rout, *mask_wedge.to('deg').value, **kw_mask)
                ax.add_artist(w)
                break            

            else: #list of wedges
                w = patches.Wedge((0,0), Rout, *wedge.to('deg').value, **kw_mask)
                ax.add_artist(w)
    
    if mask_inner is not None:
        inner = patches.Circle((0,0), mask_inner.to('au').value, **kw_mask)
        ax.add_artist(inner)
        
    return fig, ax


def make_polar_map(
        map2d, levels, R, PHI, Rout,
        Rin = 0.0,
        fig=None, ax=None, 
        cmap=get_discminer_cmap('velocity'),
        fmt='%5.2f', clabel=None, gradient=0,
        **kwargs_cbar
        #,filaments=[],                            
):
    from scipy.interpolate import griddata

    kwargs_cb = dict(orientation='vertical', subplots=False, perc=2)
    kwargs_cb.update(kwargs_cbar)
    
    #SOME DEFINITIONS
    if fig is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 3))
    Rout = Rout.to('au').value
    try:
        Rin = Rin.to('au').value
    except AttributeError:
        Rin = Rin

    #GRID AND PLOT
    phi_nonan_deg = np.nan_to_num(PHI, nan=10*np.nanmax(PHI)).to('deg').value
    R_nonan_au = np.nan_to_num(R).to('au').value

    dR = int((Rout-Rin)/50)
    R1d = np.arange(Rin, Rout+dR, dR)
    phi1d = np.linspace(-180, 180, 1000)
    dP = phi1d[1] - phi1d[0]

    PP, RR = np.meshgrid(phi1d, R1d, indexing='xy')
    
    pmap2d = griddata((phi_nonan_deg.flatten(), R_nonan_au.flatten()), map2d.flatten(), (PP, RR), method='linear')
    
    make_plot = lambda map2d: ax.contourf(map2d, cmap=cmap,
                                          extent=[PP.min(), PP.max(), RR.min(), RR.max()],
                                          levels=levels,
                                          extend='both', origin='lower')
    
    if not gradient:
        im = make_plot(pmap2d)
        map2d = pmap2d
        
    else:
        dr, dphi = np.gradient(pmap2d) #in pix-1 units
        dr /= dR # in au-1 units
        dphi *= 1/np.radians(dP) * 1/RR # in au-1 rad-1 units 
        dmax = np.sqrt(dphi**2 + dr**2)
        
        if gradient=='phi':
            im = make_plot(dphi)
            map2d = dphi
        elif gradient=='r':
            im = make_plot(dr)
            map2d = dr
        elif gradient=='peak':
            im = make_plot(dmax)
            map2d = dmax
        else:
            raise InputError(
                kind, "kind must be 'attribute' or 'residuals'"
            )
   
        
    """
    kw_fil = dict(s=20, lw=1, marker='+')
    for i,filament in enumerate(filaments):
        if i==0 and filament is not None:
            kw_fil.update(dict(fc='red', ec='firebrick'))
        elif i==1 and filament is not None:
            kw_fil.update(dict(fc='blue', ec='navy'))
        rrr=R_nonan_up_au[filament.skeleton.astype('bool')]
        ppr=phi_nonan_up_deg[filament.skeleton.astype('bool')]
        ax.scatter(ppr, rrr, **kw_fil)
    """

    #DECORATIONS
    ax.axvline(-90, ls=':', lw=2.5, color='0.3', dash_capstyle='round')
    ax.axvline(90, ls=':', lw=2.5, color='0.3', dash_capstyle='round')
    ax.axvline(0, ls=':', lw=2.5, color='0.3', dash_capstyle='round')

    make_up_ax(ax, labelbottom=True, labeltop=False, labelsize=SMALL_SIZE+1,
               xlims=(-180.1, 180.1), ylims=(1.1*Rin, 0.95*Rout))
    ax.set_xticks(np.linspace(-180, 180, 13))
    ax.set_xlabel('Azimuth [deg]', fontsize=MEDIUM_SIZE)
    ax.set_ylabel('Radius [au]', fontsize=MEDIUM_SIZE)
    mod_major_ticks(ax, axis='y', nbins=10)
                       
    cax = add_cbar_ax(fig, ax, **kwargs_cb)
    cbar = plt.colorbar(im, cax=cax, format=fmt, orientation='vertical', ticks=np.linspace(levels.min(), levels.max(), 5))
    cbar.ax.tick_params(which='major', direction='in', width=2.7, size=4.8, pad=4, labelsize=SMALL_SIZE)
    cbar.ax.tick_params(which='minor', direction='in', width=2.7, size=3.3)
    cbar.set_label(clabel, fontsize=SMALL_SIZE, labelpad=20, rotation=270)
    mod_minor_ticks(cbar.ax)

    return fig, ax, cbar


def make_pie_map(
        X, Y, Rout,
        quadrant_map2d = {1: None, 2: None, 3: None, 4: None},
        quadrant_levels = {1: None, 2: None, 3: None, 4: None},
        quadrant_cmap = {1: 'gnuplot2', 2: 'jet', 3: 'plasma', 4: 'seismic'},
        quadrant_clabel = {1: None, 2: None, 3: None, 4: None},
        quadrant_fmt = {1: '%4d', 2: '%4d', 3: '%4d', 4: '%4d'},        
        gaps=[], rings=[], kinks=[],
        fig=None, ax=None,        
):
    #SOME DEFINITIONS
    if fig is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))

    Rout = Rout.to('au').value
    X = X.to('au').value
    Y = Y.to('au').value

    quadrant_reg = {4: ((X>=0) & (Y<0)),
                    3: ((X<0) & (Y<=0)),
                    2: ((X<=0) & (Y>0)),
                    1: ((X>0) & (Y>=0))
    }
    
    #CBAR STUFF
    figx, figy = fig.get_size_inches()
    figr = figy/figx
    axp = ax.get_position()
    x0, x1, y0, y1 = axp.x0, axp.x1, axp.y0, axp.y1
    w = x1 - x0
    h = y1 - y0
    dx = 0.02
    dy = 6*dx

    cbar_pos = {1: [x1, y1-0.4*h, dx, dy],
                2: [x0-dx, y1-0.4*h, dx, dy],
                3: [x0-dx, y0+0.4*h-dy, dx, dy],
                4: [x1, y0+0.4*h-dy, dx, dy]}

    #MAKE ALL
    for quad in [1,2,3,4]:
        inquad = quadrant_reg[quad]
        map2d = quadrant_map2d[quad]
        levels = quadrant_levels[quad]
        cmap = quadrant_cmap[quad]
        clabel = quadrant_clabel[quad]
        cfmt = quadrant_fmt[quad]
        
        if map2d is None:
            map2d = np.zeros_like(X)
        else:
            map2d[~inquad] = np.nan

        if levels is None:
            levels = np.linspace(np.nanmin(map2d), np.nanmax(map2d), 64)
            
        im = ax.contourf(X, Y, map2d, levels=levels, extend='both', cmap=cmap)
    
        ax_cbar = fig.add_axes(cbar_pos[quad])
        cbar_ticks = np.linspace(np.min(levels), np.max(levels), 5)
        cbar = plt.colorbar(im, cax=ax_cbar, format=cfmt, orientation='vertical', ticks=cbar_ticks)
        cbar.ax.tick_params(which='major', direction='in', width=2.7, size=4.8, pad=7, labelsize=MEDIUM_SIZE-2)
        cbar.ax.tick_params(which='minor', direction='in', width=2.7, size=3.3)
        cbar.set_label(clabel, fontsize=MEDIUM_SIZE, rotation=90, labelpad=7)        
        mod_minor_ticks(cbar.ax)

        if quad==2 or quad==3:
            cbar.ax.tick_params(which='both', left=True, right=False, labelleft=True, labelright=False)
        else:
            cbar.ax.yaxis.set_label_position('left')
            
    _make_radial_grid_2D(ax, Rout, gaps=gaps, rings=rings, kinks=kinks, make_labels=True, label_freq=2)

    for side in ['left','top','right']: ax.spines[side].set_visible(False)
    for side in ['left']: ax.spines[side].set_linewidth(3.5)

    make_up_ax(ax,
               xlims=(-1.15*Rout, 1.15*Rout),
               ylims=(-Rout-5, Rout+5),
               labelleft=False, left=False, right=False,
               labeltop=False, top=False,
               labelbottom=True, bottom=True,
               labelsize=MEDIUM_SIZE+2, rotation=45)
    mod_major_ticks(ax, axis='x', nbins=10)
    
    ax.set_xlabel('Offset [au]', fontsize=MEDIUM_SIZE+2)
    ax.set_aspect(1)
    
    return fig, ax


#***************************
#PEAK RESIDUALS AND CLUSTERS
#***************************
def append_sigma_panel(fig, ax, values, ax_std=None, weights=None, hist=False, fit_gauss_hist=False): #attach sigma panel to AxesSubplot, based on input values
    ax = np.atleast_1d(ax)

    if ax_std is None:
        axp = ax[-1].get_position()
        ax = np.append(ax, fig.add_axes([axp.x0 + axp.width, axp.y0, 0.1, axp.height]))
    else:
        ax = np.append(ax, ax_std)
        
    #gauss = lambda x, A, mu, sigma: A*np.exp(-(x-mu)**2/(2.*sigma**2))
    ax1_ylims = ax[-2].get_ylim()

    for axi in ax[:-1]:
        axi.tick_params(which='both', right=False, labelright=False)

    ax[-1].tick_params(which='both', top=False, bottom=False, labelbottom=False, 
                       left=False, labelleft=False, right=True, labelright=True)
    ax[-1].yaxis.set_label_position('right')
    ax[-1].spines['left'].set_color('0.6')
    ax[-1].spines['left'].set_linewidth(3.5)

    if weights is not None:
        values_mean = np.sum(weights*values)/np.sum(weights)
        values_std = weighted_std(values, weights, weighted_mean=values_mean)
    else:
        values_mean = np.mean(values)
        values_std = np.std(values)

    max_y = 1.0

    if hist:
        n, bins, patches = ax[-1].hist(values, bins=2*int(round(len(values)**(1/3.)))-1, orientation='horizontal', 
                                       density=True, linewidth=1.5, facecolor='0.95', edgecolor='k', alpha=1.0)
        max_y = np.max(n)
        if fit_gauss_hist: #Fit Gaussian to histogram to compare against data distribution
            coeff, var_matrix = curve_fit(_gauss, 0.5*(bins[1:]+bins[:-1]), n, p0=[max_y, values_mean, values_std])
            values_x = np.linspace(values_mean-4*values_std, values_mean+4*values_std, 100)
            values_y = _gauss(values_x, *coeff)
            ax[-1].plot(values_y, values_x, color='tomato', ls='--', lw=2.0)

    values_x = np.linspace(values_mean-4*values_std, values_mean+4*values_std, 100)
    values_pars =  [max_y, values_mean, values_std]
    values_y = _gauss(values_x, *values_pars)
    ax[-1].plot(values_y, values_x, color='limegreen', lw=3.5)
    ax[-1].set_xlim(-0.2*max_y, 1.2*max_y)

    for i in range(0,4): 
        values_stdi = values_mean+i*values_std
        gauss_values_stdi = _gauss(values_stdi, *values_pars)
        ax[-1].plot([-0.2*max_y, gauss_values_stdi], [values_stdi]*2, color='0.6', ls=':', lw=2.)
        for axi in ax[:-1]: axi.axhline(values_stdi, color='0.6', ls=':', lw=2.)
        if values_stdi < ax1_ylims[-1] and i>0:
            ax[-1].text(gauss_values_stdi+0.2*max_y, values_stdi, r'%d$\sigma$'%i, 
                        fontsize=14, ha='center', va='center', rotation=-90)

    for axi in ax:
        axi.set_ylim(*ax1_ylims)

