import numpy as np
from astropy import units as u
from astropy import constants as ct
from scipy.interpolate import interp1d, RectBivariateSpline

from .grid import GridTools
from .tools.utils import hypot_func

au_to_m = u.au.to('m')

#*******************
#VELOCITY FUNCTIONS
#*******************
def keplerian(coord, Mstar=1.0, vel_sign=1, vsys=0):
    Mstar *= u.M_sun.to('kg')
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R'] 
    return vel_sign*np.sqrt(ct.G.value*Mstar/R) * 1e-3

def keplerian_vertical(coord, Mstar=1.0, vel_sign=1, vsys=0):
    Mstar *= u.M_sun.to('kg')
    if 'R' not in coord.keys():
        R = hypot_func(coord['x'], coord['y'])
    else:
        R = coord['R'] 
    if 'r' not in coord.keys():
        r = hypot_func(R, coord['z'])
    else:
        r = coord['r']
    return vel_sign*np.sqrt(ct.G.value*Mstar/r**3)*R * 1e-3 

def velocity_hydro2d(coord, func_interp_R=None, func_interp_phi=None, Mstar=1.0, vel_sign=1, vsys=0, phip=0.0, **dummies):
    Mstar *= u.M_sun.to('kg')
    y = coord['x'] #90 deg shift so that phip=0 aligns with disc major axis
    x = coord['y']
    R = coord['R']

    if phip != 0.0:
        phip = np.radians(phip)
        x, y = GridTools._rotate_sky_plane(x, y, phip)
        
    if 'r' not in coord.keys():
        r = hypot_func(R, coord['z'])
        
    else:
        r = coord['r']

    if func_interp_R is None and func_interp_phi is not None:
        vphi = func_interp_phi(x,y, grid=False)                    
        vR = np.zeros_like(vphi)

    elif func_interp_R is not None and func_interp_phi is None:
        vR = func_interp_R(x,y, grid=False)        
        vphi = np.zeros_like(vR)

    elif func_interp_R is None and func_interp_phi is None:
        return [0, 0, 0]    

    else:
        vR = func_interp_R(x,y, grid=False)
        vphi = func_interp_phi(x,y, grid=False)
        
    #vkep = np.sqrt(ct.G.value*Mstar/r**3)*R * 1e-3
    vz = np.zeros_like(vR)
    
    return [vel_sign*(vphi), vR, 0]
    
#******************
#EMISSION SURFACES
#******************
def z_upper_exp_tapered(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

def z_lower_exp_tapered(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return -au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

def z_upper_powerlaw(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return au_to_m*(z0*(R/R0)**p - Rb*(R/R0)**q)

def z_lower_powerlaw(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return -au_to_m*(z0*(R/R0)**p - Rb*(R/R0)**q)

def z_upper_irregular(coord, z0='0.txt', kwargs_interp1d={}, **dummies):
    R = coord['R']/au_to_m
    Rmax = np.nanmax(R)
    Rf, zf = np.loadtxt(z0)
    Rf = np.append(0.0, Rf)
    zf = np.append(0.0, zf)
    if np.max(Rf) < Rmax:
        Rf = np.append(Rf, Rmax)
        zf = np.append(zf, 0.0)
    z_interp = interp1d(Rf, zf, **kwargs_interp1d)
    return au_to_m*z_interp(R)

def z_lower_irregular(coord, z0='0.txt', kwargs_interp1d={}, **dummies):
    R = coord['R']/au_to_m
    Rmax = np.nanmax(R)
    Rf, zf = np.loadtxt(z0)
    Rf = np.append(0.0, Rf)
    zf = np.append(0.0, zf)
    if np.max(Rf) < Rmax:
        Rf = np.append(Rf, Rmax)
        zf = np.append(zf, 0.0)
    z_interp = interp1d(Rf, zf, **kwargs_interp1d)
    return -au_to_m*z_interp(R)

#***************
#PEAK INTENSITY
#***************
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

def intensity_powerlaw_rbreak(coord, I0=30.0, p0=-0.4, p1=-0.4, z0=100, q=0.3, Rbreak=20, Rout=500, p=0):
    #p is a dummy variable here
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    z = coord['z']
    z0*=au_to_m
    Rout*=au_to_m
    Rbreak*=au_to_m
    A = I0*Rbreak**-p0*z0**-q
    B = I0*Rbreak**-p1*z0**-q
    Ieff = np.where(R<=Rbreak, A*R**p0*np.abs(z)**q, B*R**p1*np.abs(z)**q)
    ind = R>Rout
    Ieff[ind] = 0.0
    return Ieff

def intensity_powerlaw_rbreak_nosurf(coord, I0=1.0, p0=-2.5, p1=-1.5,
                                     Rbreak=100, Rout=300, p=0, q=0):
    #p and q are dummy variables here
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    Rnorm=Rbreak
    Rnorm*=au_to_m
    Rbreak*=au_to_m
    Rout*=au_to_m
    pwl0 =  I0*(R/Rnorm)**p0
    pwl1 =  I0*(R/Rnorm)**p1
    Ieff = np.where(R<=Rbreak, pwl0, pwl1)
    Ieff_rout = np.where(R<=Rout, Ieff, 0.0)
    return Ieff_rout

def intensity_powerlaw_rout_hydro(coord, I0=30.0, R0=100, p=-0.4, z0=100, q=0.3, Rout=500, func_interp_sigma=None, phip=0.0, weight=1.0):
    
    y = coord['x'] 
    x = coord['y']

    if phip != 0.0:
        phip = np.radians(phip)
        x, y = GridTools._rotate_sky_plane(x, y, phip)

    if 'R' not in coord.keys():
        R = hypot_func(coord['x'], coord['y'])
    else:
        R = coord['R']

    if func_interp_sigma is None:
        sigma = np.zeros_like(R)
    else:
        sigma = func_interp_sigma(x,y, grid=False)
        
    z = coord['z']
    R0*=au_to_m
    z0*=au_to_m
    Rout*=au_to_m
    A = I0*R0**-p*z0**-q
    Ieff = np.where(R<=Rout, A*R**p*np.abs(z)**q, 0.0) * sigma**weight

    return Ieff

def intensity_powerlaw_rbreak_hydro(coord, I0=30.0, p0=-0.4, p1=-0.4, z0=100, q=0.3, Rbreak=20, Rout=500, p=0, func_interp_sigma=None, phip=0.0, weight=1.0):
    
    y = coord['x'] 
    x = coord['y']

    if phip != 0.0:
        phip = np.radians(phip)
        x, y = GridTools._rotate_sky_plane(x, y, phip)

    if 'R' not in coord.keys():
        R = hypot_func(coord['x'], coord['y'])
    else:
        R = coord['R']

    if func_interp_sigma is None:
        sigma = np.zeros_like(R)
    else:
        sigma = func_interp_sigma(x,y, grid=False)

    z = coord['z']
    z0*=au_to_m
    Rout*=au_to_m
    Rbreak*=au_to_m
    A = I0*Rbreak**-p0*z0**-q
    B = I0*Rbreak**-p1*z0**-q
    Ieff = np.where(R<=Rbreak, A*R**p0*np.abs(z)**q, B*R**p1*np.abs(z)**q) * (sigma/np.max(sigma))**weight
    #print (np.nanmax(sigma), np.nanmin(sigma))
    ind = R>Rout
    Ieff[ind] = 0.0
    return Ieff

#***********
#LINE WIDTH 
#***********
def linewidth_powerlaw(coord, L0=0.2, p=-0.4, q=0.3, R0=100, z0=100):
    R0*=au_to_m
    z0*=au_to_m
    if 'R' not in coord.keys():
        R = hypot_func(coord['x'], coord['y'])
    else:
        R = coord['R'] 
    z = coord['z']        
    A = L0*R0**-p*z0**-q
    return A*R**p*np.abs(z)**q

#***********
#LINE SLOPE 
#***********
def lineslope_powerlaw(coord, Ls=5.0, p=0.0, q=0.0, R0=100, z0=100):
    R0*=au_to_m
    z0*=au_to_m
    if p==0.0 and q==0.0:
        return Ls
    else:
        if 'R' not in coord.keys():
            R = hypot_func(coord['x'], coord['y'])
        else:
            R = coord['R'] 
        z = coord['z']        
        A = Ls*R0**-p*z0**-q
        return A*R**p*np.abs(z)**q

#**************
#LINE PROFILES
#**************
def line_profile_bell(v_chan, v, v_sigma, b_slope):
    return 1/(1+np.abs((v-v_chan)/v_sigma)**(2*b_slope))        
    
def line_profile_gaussian(v_chan, v, v_sigma, *dummies):
    return np.exp(-0.5*((v-v_chan)/v_sigma)**2)

#***********************
#UPPER + LOWER PROFILES
#***********************
def line_uplow_mask(Iup, Ilow):
    #velocity nans might differ from intensity nans when a z=0 and SG is active, nanmax must be used
    return np.nanmax([Iup, Ilow], axis=0)

def line_uplow_sum(Iup, Ilow):
    return Iup + Ilow    
