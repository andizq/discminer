from discminer.mining_control import _mining_destackcube
from discminer.core import Data
from discminer.mining_utils import load_disc_grid, init_data_and_model

import os
import sys
import json
import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy.interpolate import interp1d

if __name__ == '__main__':
    parser = _mining_destackcube(None)
    args = parser.parse_args()

def add_history(header, text):
    """Append a FITS HISTORY card without overwriting previous HISTORY entries."""
    hdr = header.copy()
    hdr.add_history(text)
    return hdr

def get_vphi_interp(vphi_filename):
    try:
        vphi_file = np.loadtxt(vphi_filename, comments='#')
    except Exception:
        sys.exit(
            f"\nERROR: Could not load '{vphi_filename}'.\n\n"
            "Run:\n"
            "    discminer radprof\n"
            "to generate a radial velocity profile of your data, or use:\n"
            "    discminer destack -keplerian 1\n"
            "to fall back to a pure Keplerian profile for the destacking.\n"
        )

    R_vphi_au = vphi_file[:, 0]
    vphi_prof = vphi_file[:, 1]

    isort = np.argsort(R_vphi_au)
    R_vphi_au = R_vphi_au[isort]
    vphi_prof = vphi_prof[isort]

    return interp1d(
        R_vphi_au,
        vphi_prof,
        bounds_error=False,
        fill_value=np.nan,
    )

def destack_cube(stackcube, out_vchannels, vcenti):
    """
    Undo make_stackcube.py.

    make_stackcube.py stores, for each pixel,
        S(v_dep) = I(v_dep + v0), with v_dep = v - v0.

    Therefore, to recover the original cube on velocity grid v,
        I(v) = S(v - v0).
    """
    nchan_out = len(out_vchannels)
    nchan_stack, ny, nx = stackcube.data.shape

    if vcenti.shape != (ny, nx):
        raise ValueError(
            f"vcenti has shape {vcenti.shape}, but cube spatial shape is {(ny, nx)}"
        )

    destacked = np.full((nchan_out, ny, nx), np.nan, dtype=stackcube.data.dtype)
    stack_vchannels = np.asarray(stackcube.vchannels, dtype=float)

    for j in range(ny):
        for i in range(nx):
            spec = stackcube.data[:, j, i]
            v0 = vcenti[j, i]

            if not np.isfinite(v0):
                continue
            if not np.any(np.isfinite(spec)):
                continue

            good = np.isfinite(stack_vchannels) & np.isfinite(spec)
            if np.count_nonzero(good) < 2:
                continue

            order = np.argsort(stack_vchannels[good])
            x = stack_vchannels[good][order]
            y = spec[good][order]

            interp_spec = interp1d(
                x,
                y,
                kind='linear',
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )

            #Inverse shift: evaluate stacked spectrum at v_dep = v_original - v0.
            destacked[:, j, i] = interp_spec(out_vchannels - v0)

    return destacked

def default_output_name(stackcube_filename):
    path, filename = os.path.split(stackcube_filename)
    name, ext = os.path.splitext(filename)
    if ext == '':
        ext = '.fits'
    return os.path.join(path, f'destacked_{name}{ext}')

#**********************
#JSON AND PARSER STUFF
#**********************
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']

vel_sign = best['velocity']['vel_sign']
vsys = best['velocity']['vsys']
Rout = best['intensity']['Rout']
incl = best['orientation']['incl']
tag = meta['tag']

au_to_m = u.au.to('m')
dpc = meta['dpc'] * u.pc

#*******************
#LOAD DATA AND MODEL
#*******************
_, model = init_data_and_model(Rmin=0, Rmax=1.6)

if args.refcube is None:
    reference_cube = f'cube_data_{tag}_convtb.fits'
else:
    reference_cube = args.refcube

stackcube = Data(args.stackcube, dpc)
refcube = Data(reference_cube, dpc)

beam_au = refcube.beam_size.to('au').value
if args.absolute_Rinner >= 0:
    Rmod_in = args.absolute_Rinner
else:
    Rmod_in = args.Rinner * beam_au

if args.absolute_Router >= 0:
    Rmod_out = args.absolute_Router
else:
    Rmod_out = args.Router * Rout

#****************************
#LOAD DISC GEOMETRY AND MASK
#****************************
R, phi, z = load_disc_grid()
R_s = R[args.surface] * u.m.to('au')
phi_s = np.degrees(phi[args.surface])

#**********************************
#LOAD / COMPUTE ROTATION CENTROIDS
#**********************************
zupi = model.z_upper_func({'R': R_s * au_to_m}, **best['height_upper'])

if args.keplerian:
    vphii_data = model.velocity_func({'R': R_s * au_to_m, 'z': zupi}, **best['velocity'])
else:
    vphi_interp_data = get_vphi_interp('radial_profile_velocity_data.dat')
    vphii_data = vel_sign * vphi_interp_data(R_s)

vcenti_data = vsys + vphii_data * np.sin(incl) * np.cos(np.radians(phi_s))

#*********************************
#DESTACK EACH PIXEL
#*********************************
if stackcube.data.shape[1:] != refcube.data.shape[1:]:
    raise ValueError(
        'Input stackcube and reference cube have different spatial shapes: '
        f'{stackcube.data.shape[1:]} vs {refcube.data.shape[1:]}'
    )

destacked_cube = destack_cube(stackcube, refcube.vchannels, vcenti_data)

#****************
#WRITE FITS CUBE
#****************
header_out = add_history(
    refcube.header,
    'discminer destackcube: shifted stacked spectra back to original projected velocities',
)
header_out['NAXIS'] = 3
header_out['NAXIS3'] = len(refcube.vchannels)

output = args.output if args.output is not None else default_output_name(args.stackcube)
fits.writeto(output, destacked_cube, header_out, overwrite=True)

print(f'Wrote destacked cube to {output}')
print(f'Destacked cube shape: {destacked_cube.shape}')
print(
    'Output velocity grid copied from reference cube: '
    f'{refcube.vchannels[0]:.6f} to {refcube.vchannels[-1]:.6f} '
    f'with dv={np.median(np.diff(refcube.vchannels)):.6f}'
)
