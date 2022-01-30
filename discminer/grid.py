import numpy as np
from .disc2d import Tools

def grid(xmax, nx, indexing="xy", verbose=True):
    """
    Compute Cartesian (x,y) and polar (R, phi) grid. Assuming square grid, namely ymax=xmax, and ny=nx.
    
    Parameters
    ----------
    xmax : `~astropy.units.Quantity`
        Maximum spatial extent of the grid. 
        The 2D extent of the output grid is [-xmax, xmax] in x and [-xmax, xmax] in y.
    nx : int
        Number of grid cells along each spatial dimension.
    indexing : str, optional
        Cartesian (‘xy’, default) or matrix (‘ij’) indexing of output xy meshgrid. See `~numpy.meshgrid`.
    verbose : bool, optional
        if True, print informative messages.

    Returns
    -------    
    grid : dict
        Dictionary containing grid information.
    
    Examples
    --------        
    """

    Tools._break_line()
    xymax = np.array([xmax, xmax])
    nx = np.int32(nx)
    step = 2 * xmax / (nx - 1)

    if verbose:
        print("Computing square grid...")
        print("Grid maximum extent:", xmax)
        print("Grid step (cell size):", step)

    xgrid = np.linspace(-xmax, xmax, nx)
    xygrid = [xgrid, xgrid]
    XY = np.meshgrid(xgrid, xgrid, indexing=indexing)
    xList, yList = [xy.flatten() for xy in XY]
    RList = np.linalg.norm([xList, yList], axis=0)
    phiList = np.arctan2(yList, xList)
    phiList = np.where(phiList < 0, phiList + 2 * np.pi, phiList)
    Tools._break_line()

    return {
        "x": xList,
        "y": yList,
        "R": RList,
        "phi": phiList,
        "meshgrid": XY,
        "xygrid": xygrid,
        "nx": nx,
        "xmax": xmax,
        "ncells": nx ** 2,
        "step": step,
    }
