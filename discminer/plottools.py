"""
plottools module
===========
"""
import copy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

_attribute_colors = {
    'residuals': ["#010b14","#1d4c72","#3d97e1","#ffffff","#f36353","#b6163e","#140200"],
    'velocity': ["#010b14","#1d4c72","#3d97e1","#ffffff","#f36353","#b6163e","#140200"],
    'linewidth': ["#"+tmp for tmp in ["001219","005f73","0a9396","94d2bd","e9d8a6","ee9b00","ca6702","bb3e03","ae2012","9b2226"]],
    'intensity': "terrain_r",
}

_attribute_cranges = {
    'residuals': [0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0],
    'velocity': [0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0],
    'linewidth': None,
    'intensity': "matplotlib",    
}


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
    """#The following does not work, plt does not know where to locate the newly added colorss
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

def get_attribute_cmap(attribute):
    cranges = _attribute_cranges[attribute]
    if cranges=='matplotlib':
        cmap = copy.copy(plt.get_cmap("terrain_r"))
    elif isinstance(cranges, Iterable) or cranges is None:
        colors = _attribute_colors[attribute]
        cmap = get_continuous_cmap(colors, float_list=cranges)
    return cmap
        
