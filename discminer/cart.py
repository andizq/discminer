import numpy as np
from astropy import units as u

au_to_m = u.au.to('m')

#****************
#CUSTOM SURFACES
#****************
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

#****************
#CUSTOM INTENSITY
#****************
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
