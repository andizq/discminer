import autograd.numpy as np
from autograd.extend import primitive, defvjp
from scipy.interpolate import LinearNDInterpolator


def get_griddata(old_coord, new_coord):
    
    
    xi = np.array(new_coord).T
    old_xi_shape = xi.shape
    xi = xi.reshape(-1, xi.shape[-1])
    
    if isinstance(old_coord, tuple):
        old_coord =  np.array(old_coord).T
    interp = LinearNDInterpolator(old_coord, np.ones((old_coord.shape[0],1)))
    
    print(old_coord.shape)
    print(xi.shape)
    print(np.ones((old_coord.shape[0],1)).shape)
    c = np.zeros((xi.shape[0], old_coord.shape[0]))
    simplex_indices = interp.tri.find_simplex(xi)
    print(simplex_indices.shape)
    for n in range(c.shape[0]):
        isimplex = simplex_indices[n]
        if isimplex == -1:
            c[n,:] = np.nan
        else:
            c[n,interp.tri.simplices[isimplex]] = [*(interp.tri.transform[isimplex, :ndim,:ndim]@(xi[n] - interp.tri.transform[isimplex, ndim, :])), 1-(interp.tri.transform[simplex_indices[n], :ndim,:ndim]@(xi[n] - interp.tri.transform[simplex_indices[n], ndim, :])).sum()]
    
    return lambda values: (c@values).reshape(*old_xi_shape[:-1]).T


