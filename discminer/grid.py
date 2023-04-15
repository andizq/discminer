from .tools.utils import FrontendUtils
from astropy import units as u
from scipy.optimize import root
import numpy as np

_break_line = FrontendUtils._break_line
au_to_m = u.au.to('m')

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
        Dictionary containing grid information. Cartesian and polar grids are returned in units of metres.
    
    Examples
    --------        

    """
    _break_line()
    xmax = xmax.to(u.m).value
    xymax = np.array([xmax, xmax])
    nx = np.int32(nx)
    step = 2 * xmax / (nx - 1)

    if verbose:
        print("Computing grid...")
        print("Grid maximum extent:", xmax)
        print("Grid step (cell size):", step)

    xgrid = np.linspace(-xmax, xmax, nx)
    xygrid = [xgrid, xgrid]
    XY = np.meshgrid(xgrid, xgrid, indexing=indexing)
    xList, yList = [xy.flatten() for xy in XY]
    RList = np.linalg.norm([xList, yList], axis=0)
    phiList = np.arctan2(yList, xList)
    #phiList = np.where(phiList < 0, phiList + 2 * np.pi, phiList)
    _break_line()

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
        "extent": np.array([-xmax, xmax, -xmax, xmax])*u.m.to(u.au)
    }

class GridTools:
    @staticmethod
    def _rotate_sky_plane(x, y, ang):
        xy = np.array([x,y])
        cos_ang = np.cos(ang)
        sin_ang = np.sin(ang)
        rot = np.array([[cos_ang, -sin_ang],
                        [sin_ang, cos_ang]])
        return np.dot(rot, xy)

    @staticmethod
    def _rotate_sky_plane3d(x, y, z, ang, axis='z'):
        xyz = np.array([x,y,z])
        cos_ang = np.cos(ang)
        sin_ang = np.sin(ang)
        if axis == 'x':
            rot = np.array([[1, 0, 0],
                            [0, cos_ang, -sin_ang],
                            [0, sin_ang, cos_ang]])
        if axis == 'y':
            rot = np.array([[cos_ang, 0, -sin_ang],
                            [0, 1, 0],
                            [sin_ang, 0, cos_ang]])
            
        if axis == 'z':
            rot = np.array([[cos_ang, -sin_ang , 0],
                            [sin_ang, cos_ang, 0], 
                            [0, 0, 1]])
        return np.dot(rot, xyz)

    @staticmethod
    def _project_on_skyplane(x, y, z, cos_incl, sin_incl):
        x_pro = x
        y_pro = y * cos_incl - z * sin_incl
        z_pro = y * sin_incl + z * cos_incl
        return x_pro, y_pro, z_pro

    @staticmethod
    def get_sky_from_disc_coords(R, az, z, incl, PA, xc=0, yc=0):
        xp = R*np.cos(az)
        yp = R*np.sin(az)
        zp = z
        xp, yp, zp = GridTools._project_on_skyplane(xp, yp, zp, np.cos(incl), np.sin(incl))
        xp, yp = GridTools._rotate_sky_plane(xp, yp, PA)
        return xp+xc, yp+yc, zp

    @staticmethod
    def get_disc_from_sky_coords(xs, ys, z_func, z_pars, incl, PA, xc=0, yc=0, midplane=False):
        #xs, ys: x and y on sky plane
        xs, ys = GridTools._rotate_sky_plane(xs-xc, ys-yc, -PA) 
        xd = xs
        cos_incl = np.cos(incl)
        sin_incl = np.sin(incl)
        def find_yd(yd):
            R = np.sqrt(xd**2+yd[0]**2)
            if midplane: zd = 0
            else: zd = z_func({'R': R*au_to_m}, **z_pars)/au_to_m
            return yd[0]*cos_incl - zd*sin_incl - ys #see _project_on_skyplane() 
        yd = root(find_yd, [100], method='hybr')
        return xd, yd.x[0]
    
