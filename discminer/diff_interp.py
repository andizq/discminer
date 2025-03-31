import numpy as np
from scipy.interpolate import LinearNDInterpolator
import scipy.sparse as sp

def get_griddata_sparse(old_coord, new_coord):
    
    #print('generating new interpolation function')
    
    xi = np.array(new_coord).T
    old_xi_shape = xi.shape
    xi = xi.reshape(-1, xi.shape[-1])

    #old coord can be an array with shape (N, ndim) or a tuple of arrays of N elements
    if isinstance(old_coord, tuple):
        old_coord = np.array(old_coord).T
    ndim = old_coord.shape[1]
        
    #construct the triangulation using LinearNDInterpolator
    interp = LinearNDInterpolator(old_coord, np.ones((old_coord.shape[0], 1)))
    simplex_indices = interp.tri.find_simplex(xi)

    #construct the interpolation matrix in COO format
    row_indices, col_indices, values = [], [], []

    #TODO: I think this can be done without the python for loop
    for n in range(xi.shape[0]):
        isimplex = simplex_indices[n]
        if isimplex == -1:
            continue  
        
        indices = interp.tri.simplices[isimplex]
        weights = [*(interp.tri.transform[isimplex, :ndim, :ndim] @ 
                     (xi[n] - interp.tri.transform[isimplex, ndim, :])), 
                   1 - (interp.tri.transform[isimplex, :ndim, :ndim] @ 
                        (xi[n] - interp.tri.transform[isimplex, ndim, :])).sum()]

        row_indices.extend([n] * len(indices))
        col_indices.extend(indices)
        values.extend(weights)

    #Convert to CSR format (more efficient for matrix multiplication)
    c = sp.coo_matrix((values, (row_indices, col_indices)), 
                      shape=(xi.shape[0], old_coord.shape[0])).tocsr()

    return lambda values: (c @ values).reshape(*old_xi_shape[:-1]).T
