from discminer.mining_control import _mining_stackcube
from discminer.core import Data
from discminer.mining_utils import (_get_mask_tuples,
                                    get_noise_mask,
                                    load_disc_grid,
                                    get_2d_plot_decorators,
                                    init_data_and_model,
                                    load_moments,
)

import os
import sys
import json
import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy.interpolate import interp1d


if __name__ == '__main__':
    parser = _mining_stackcube(None)
    args = parser.parse_args()

def update_spectral_header(header, vcenters):
    """Return a FITS header with axis-3 updated to match vcenters."""
    hdr = header.copy()

    if len(vcenters) < 2:
        raise ValueError("vcenters must contain at least two channels")

    dv = float(np.median(np.diff(vcenters)))

    hdr['NAXIS'] = 3
    hdr['NAXIS3'] = len(vcenters)
    hdr['CRPIX3'] = 1.0
    hdr['CRVAL3'] = float(vcenters[0])
    hdr['CDELT3'] = dv

    # Keep existing spectral metadata when possible.
    if 'CTYPE3' not in hdr:
        hdr['CTYPE3'] = 'VELO-LSR'
    if 'CUNIT3' not in hdr:
        hdr['CUNIT3'] = 'km/s'

    hdr['HISTORY'] = 'Spectra shifted by projected vphi and interpolated onto a common velocity grid'
    hdr['HISTORY'] = 'discminer stackcube'

    return hdr

#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']

vel_sign = best['velocity']['vel_sign']
vsys = best['velocity']['vsys']
Rout = best['intensity']['Rout']
incl = best['orientation']['incl']
PA = best['orientation']['PA']
xc = best['orientation']['xc']
yc = best['orientation']['yc']

gaps = custom['gaps']
rings = custom['rings']
try:
    kinks = custom['kinks']
except KeyError:
    kinks = []

#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']
au_to_m = u.au.to('m')

dpc = meta['dpc']*u.pc

#*******************
#LOAD DATA AND MODEL
#*******************
_, model = init_data_and_model(Rmin=0, Rmax=1.6)

file_data = 'cube_data_%s_convtb.fits'%tag
file_model = 'cube_model_%s_convtb.fits'%tag
file_residuals = 'cube_residuals_%s_convtb.fits'%tag

datacube = Data(file_data, dpc)
modelcube = Data(file_model, dpc)

#Definitions for plots
beam_au = datacube.beam_size.to('au').value
if args.absolute_Rinner>=0:
    Rmod_in = args.absolute_Rinner
else:
    Rmod_in = args.Rinner*beam_au

if args.absolute_Router>=0:
    Rmod_out = args.absolute_Router
else:
    Rmod_out = args.Router*Rout

#****************************
#LOAD DISC GEOMETRY AND MASK
#****************************
R, phi, z = load_disc_grid()

R_s = R[args.surface]*u.m.to('au')
phi_s = np.degrees(phi[args.surface])

#**********************************
#LOAD EXTERNAL RADIAL vphi PROFILE
#**********************************
vphi_datafile = 'radial_profile_velocity_data.dat'
vphi_modelfile = 'radial_profile_velocity_model.dat'

def get_vphi_interp(vphi_filename):
    try:
        vphi_file = np.loadtxt(vphi_filename, comments='#')
    except Exception:
        sys.exit(
            f"\nERROR: Could not load '{vphi_filename}'.\n\n"
            "Run:\n"
            "    discminer radprof\n"
            "to generate a radial velocity profile of your data, or use:\n"
            "    discminer stack -keplerian 1\n"
            "to fall back to a pure Keplerian profile for the stacking.\n"
        )

    R_vphi_au = vphi_file[:, 0] # first column: radius [au]
    vphi_prof = vphi_file[:, 1] # second column: vphi profile
    vphi_std  = vphi_file[:, 2] # optional third column: stddev

    isort = np.argsort(R_vphi_au)
    R_vphi_au = R_vphi_au[isort]
    vphi_prof = vphi_prof[isort]
    vphi_std  = vphi_std[isort]

    vphi_interp = interp1d(
        R_vphi_au,
        vphi_prof,
        bounds_error=False,
        fill_value=np.nan
    )
    return vphi_interp
    
#*******************
#GET VPHI CENTROIDS
#*******************
chanwidth = np.abs(np.median(np.diff(datacube.vchannels)))
dv_native = chanwidth/args.binsperchan

vel_range = np.max(datacube.vchannels) - np.min(datacube.vchannels)
vmin = -0.5*vel_range 
vmax = 0.5*vel_range 
bins   = np.arange(vmin, vmax + 0.1*dv_native, dv_native)
vcenters = 0.5*(bins[:-1] + bins[1:])

#GET VPHI CENTROID
zupi = model.z_upper_func({'R': R_s*au_to_m}, **best['height_upper'])

if args.keplerian:
    vphii_data = vphii_model = model.velocity_func({'R': R_s*au_to_m, 'z': zupi}, **best['velocity'])
else:
    vphi_interp_data = get_vphi_interp(vphi_datafile)
    vphi_interp_model = get_vphi_interp(vphi_modelfile)    
    vphii_data = vel_sign * vphi_interp_data(R_s)
    vphii_model = vel_sign * vphi_interp_model(R_s)    
        
vcenti_data = vsys + vphii_data * np.sin(incl) * np.cos(np.radians(phi_s))
vcenti_model = vsys + vphii_model * np.sin(incl) * np.cos(np.radians(phi_s))

#*********************************
#SHIFT EACH PIXEL AND INTERPOLATE
#*********************************
nchan_new = len(vcenters)
nchan_old, ny, nx = datacube.data.shape
stackcube_data = np.full((nchan_new, ny, nx), np.nan, dtype=datacube.data.dtype)
stackcube_model = np.full((nchan_new, ny, nx), np.nan, dtype=datacube.data.dtype)

def fill_stackcube(stackcube, cube, vcenti):
    for j in range(ny):
        for i in range(nx):
            spec = cube.data[:, j, i]
            v0 = vcenti[j, i]

            if not np.isfinite(v0):
                continue
            if not np.any(np.isfinite(spec)):
                continue

            vdep_ij = cube.vchannels - v0
            good = np.isfinite(vdep_ij) & np.isfinite(spec)
            if np.count_nonzero(good) < 2:
                continue

            order = np.argsort(vdep_ij[good])
            x = vdep_ij[good][order]
            y = spec[good][order]

            interp_spec = interp1d(
                x,
                y,
                kind='linear',
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            stackcube[:, j, i] = interp_spec(vcenters)

fill_stackcube(stackcube_data, datacube, vcenti_data)
fill_stackcube(stackcube_model, modelcube, vcenti_model)

#****************
#WRITE FITS CUBE
#****************
header_data = update_spectral_header(datacube.header, vcenters)
header_model = update_spectral_header(modelcube.header, vcenters)

def write_stackcube(file_in, stackcube, header_out):
    path, filename = os.path.split(file_in)
    name, ext = os.path.splitext(filename) # Split into name + extension
    new_filename = f'{name}_stackedcube{ext}'
    new_path = os.path.join(path, new_filename)

    fits.writeto(new_path, stackcube, header_out, overwrite=True)
    print(f'Wrote shifted cube to {new_path}')
    print(f'Stacked cube shape: {stackcube.shape}')
    print(f'Common velocity grid from {vcenters[0]:.6f} to {vcenters[-1]:.6f} with dv={np.median(np.diff(vcenters)):.6f}')

write_stackcube(file_data, stackcube_data, header_data)
write_stackcube(file_model, stackcube_model, header_model)
write_stackcube(file_residuals, stackcube_data-stackcube_model, header_data)
