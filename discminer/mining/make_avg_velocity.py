from discminer.mining_control import _mining_avg_velocity
from discminer.mining_utils import get_moment_tag, load_disc_grid

import json
import os

import numpy as np
from astropy import units as u
from astropy.io import fits


BASELINE_COMPONENTS = {
    'v0phi': ('phi',),
    'v0r': ('phi', 'r'),
    'v0z': ('phi', 'z'),
    'v0all': ('phi', 'r', 'z'),
}


def load_velocity_profile(filename):
    """Load and validate the radius and velocity columns of a radprof file."""
    try:
        profile = np.loadtxt(filename, comments='#', ndmin=2)
    except OSError as exc:
        raise FileNotFoundError(
            f"Could not load radial velocity profile {filename!r}. "
            "Run 'discminer radprof -m velocity' first."
        ) from exc

    if profile.shape[1] < 2:
        raise ValueError(
            f"Radial velocity profile {filename!r} must have at least two columns"
        )

    radius = profile[:, 0]
    velocity = profile[:, 1]
    finite = np.isfinite(radius) & np.isfinite(velocity)
    radius = radius[finite]
    velocity = velocity[finite]

    if len(radius) < 2:
        raise ValueError(
            f"Radial velocity profile {filename!r} needs at least two finite rows"
        )

    order = np.argsort(radius)
    radius = radius[order]
    velocity = velocity[order]
    radius, unique_indices = np.unique(radius, return_index=True)
    velocity = velocity[unique_indices]

    if len(radius) < 2:
        raise ValueError(
            f"Radial velocity profile {filename!r} needs at least two unique radii"
        )

    return radius, velocity


def interpolate_profile(radius_grid, profile):
    """Interpolate a radial profile without extrapolating beyond its domain."""
    radius, velocity = profile
    radius_eval = np.asarray(radius_grid, dtype=float).copy()
    at_inner_edge = np.isclose(
        radius_eval, radius[0], rtol=1e-10, atol=1e-10
    )
    at_outer_edge = np.isclose(
        radius_eval, radius[-1], rtol=1e-10, atol=1e-10
    )
    radius_eval[at_inner_edge] = radius[0]
    radius_eval[at_outer_edge] = radius[-1]
    return np.interp(
        radius_eval,
        radius,
        velocity,
        left=np.nan,
        right=np.nan,
    )


def project_axisymmetric_velocity(
        radius_au,
        phi,
        profiles,
        components,
        incl,
        vel_sign,
        vsys,
):
    """Project selected axisymmetric velocity components onto the line of sight."""
    baseline = np.full(np.shape(radius_au), float(vsys), dtype=float)

    if 'phi' in components:
        vphi = interpolate_profile(radius_au, profiles['phi'])
        baseline += vel_sign * vphi * np.sin(incl) * np.cos(phi)

    if 'r' in components:
        vr = interpolate_profile(radius_au, profiles['r'])
        baseline -= vr * np.sin(incl) * np.sin(phi)

    if 'z' in components:
        vz = interpolate_profile(radius_au, profiles['z'])
        baseline -= vz * np.cos(incl)

    return baseline


def _set_product_header(header, moment, components, product):
    output = header.copy()
    output['DMAVGVEL'] = (True, 'Axisymmetric data-derived velocity product')
    output['DMMOMENT'] = (moment, 'discminer averaged-velocity moment name')
    output['DMCOMPS'] = ('+'.join(components), 'Axisymmetric baseline components')
    output['DMPROD'] = (product, 'Observed data or averaged baseline')
    output['HISTORY'] = (
        'discminer avgvel: generated from radial velocity profiles'
    )
    return output


def make_average_velocity_maps(args):
    with open('parfile.json') as json_file:
        pars = json.load(json_file)

    meta = pars['metadata']
    best = pars['best_fit']
    dir_data = meta.get('dir_data', './')
    dir_model = meta.get('dir_model', './')

    surface = {'up': 'upper', 'low': 'lower'}.get(args.surface, args.surface)
    if surface not in ['upper', 'lower']:
        raise ValueError("avgvel requires surface='upper' or 'lower'")

    source_tag, _, _ = get_moment_tag(
        'velocity',
        kernel=args.kernel,
        surface=surface,
        kind=args.kind,
    )
    source_filename = os.path.join(dir_data, f'{source_tag}_data.fits')
    velocity_data = fits.getdata(source_filename).squeeze()
    source_header = fits.getheader(source_filename)

    if velocity_data.ndim != 2:
        raise ValueError(
            f"Input velocity moment map must be 2D; got shape {velocity_data.shape}"
        )

    radius, phi, _ = load_disc_grid()
    radius_au = radius[surface] * u.m.to('au')
    phi_surface = phi[surface]

    if radius_au.shape != velocity_data.shape or phi_surface.shape != velocity_data.shape:
        raise ValueError(
            "Disc grid and input velocity moment map have different shapes: "
            f"R={radius_au.shape}, phi={phi_surface.shape}, "
            f"velocity={velocity_data.shape}"
        )

    prefix = args.profile_prefix
    profile_filenames = {
        'phi': f'{prefix}_data.dat',
        'r': f'{prefix}_vr.dat',
        'z': f'{prefix}_vz.dat',
    }
    profiles = {
        component: load_velocity_profile(filename)
        for component, filename in profile_filenames.items()
    }

    velocity_pars = best['velocity']
    orientation = best['orientation']
    incl = orientation['incl']
    vel_sign = velocity_pars['vel_sign']
    vsys = velocity_pars['vsys']

    outputs = []
    invalid_data = ~np.isfinite(velocity_data)

    for moment, components in BASELINE_COMPONENTS.items():
        baseline = project_axisymmetric_velocity(
            radius_au,
            phi_surface,
            profiles,
            components,
            incl,
            vel_sign,
            vsys,
        )
        baseline[invalid_data] = np.nan

        output_tag, _, _ = get_moment_tag(
            moment,
            kernel=args.kernel,
            surface=surface,
            kind=args.kind,
        )
        data_filename = os.path.join(dir_data, f'{output_tag}_data.fits')
        model_filename = os.path.join(dir_model, f'{output_tag}_model.fits')

        data_header = _set_product_header(
            source_header, moment, components, 'observed'
        )
        model_header = _set_product_header(
            source_header, moment, components, 'average'
        )

        fits.writeto(
            data_filename,
            velocity_data,
            header=data_header,
            overwrite=bool(args.overwrite),
        )
        fits.writeto(
            model_filename,
            baseline,
            header=model_header,
            overwrite=bool(args.overwrite),
        )
        outputs.extend([data_filename, model_filename])
        print(
            f"Wrote {moment} ({'+'.join(components)}) products to "
            f"{data_filename} and {model_filename}"
        )

    return outputs


if __name__ == '__main__':
    parser = _mining_avg_velocity(None)
    make_average_velocity_maps(parser.parse_args())
elif 'args' in globals():
    make_average_velocity_maps(args)
