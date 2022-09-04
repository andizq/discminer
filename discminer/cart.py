"""
cart module
===========
Classes:
"""

import copy
from collections.abc import Iterable
from .plottools import get_continuous_cmap
import numpy as np
import numbers
from sf3dmodels.utils import constants as sfc
from sf3dmodels.utils import units as sfu
from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib.pyplot as plt


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


def get_attribute_cmap(attribute):
    cranges = _attribute_cranges[attribute]
    if cranges=='matplotlib':
        cmap = copy.copy(plt.get_cmap("terrain_r"))
    elif isinstance(cranges, Iterable) or cranges is None:
        colors = _attribute_colors[attribute]
        cmap = get_continuous_cmap(colors, float_list=cranges)
    return cmap
        
