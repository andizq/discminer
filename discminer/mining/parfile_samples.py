"""Read-only selection and parameter mapping for parfile ensembles."""

from pathlib import Path

import numpy as np


def infer_parameter_mapping(log_parameters, template, ndim):
    """Recover the column order of an unambiguous legacy DiscMiner JSON log.

    The legacy writer copies fixed values unchanged into p0/best_fit/errneg/
    errpos. It replaces these four entries for sampled parameters and writes
    the nested dictionaries in the same order used to assemble the fit vector.
    Shared destinations also receive sampled summaries, so do not try to
    collapse matching summaries when the candidate count differs from ndim.
    """
    if ndim is None or not isinstance(log_parameters, dict):
        raise ValueError(
            'The fit log has no metadata.parameter_order. Automatic inference '
            'requires a JSON log with p0, best_fit (or params), errneg and '
            'errpos, and the backend dimensionality. Otherwise use a matching '
            'JSON fit log with explicit metadata.parameter_order.'
        )
    best_key = 'best_fit' if 'best_fit' in log_parameters else 'params'
    sections = ('p0', best_key, 'errneg', 'errpos')
    order = []
    try:
        for category, parameters in template.items():
            for parameter in parameters:
                values = [log_parameters[key][category][parameter] for key in sections]
                if all(value == values[0] for value in values[1:]):
                    continue  # Fixed values, including False, are copied verbatim.
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not np.isfinite(value) for value in values):
                    raise ValueError(
                        'Cannot infer the sampling status of %s.%s from its '
                        'non-numeric/non-finite or inconsistent boolean summaries.'
                        % (category, parameter)
                    )
                order.append('%s.%s' % (category, parameter))
    except (KeyError, TypeError) as exc:
        raise ValueError(
            'Automatic parameter inference requires complete p0, best_fit '
            '(or params), errneg and errpos sections. Use a complete matching '
            'JSON fit log or one with explicit metadata.parameter_order.'
        ) from exc
    if len(order) != ndim:
        raise ValueError(
            'Legacy log inference found %d potentially sampled parameters, '
            'but the backend has %d columns. Fixed/shared parameters or '
            'degenerate summaries cannot be resolved unambiguously; use a '
            'matching JSON fit log with metadata.parameter_order and '
            'metadata.shared_parameters.'
            % (len(order), ndim)
        )
    return {'parameter_order': order, 'shared_parameters': {}}


def load_parameter_mapping(metadata, template, log_parameters=None, ndim=None):
    """Prefer explicit mappings; infer older JSON logs only when counts agree."""
    source = metadata
    mapping_source = 'log_metadata'
    if 'parameter_order' not in metadata:
        source = infer_parameter_mapping(log_parameters, template, ndim)
        mapping_source = 'legacy_log_inference'
    if not isinstance(source, dict) or 'parameter_order' not in source:
        raise ValueError('The fit log metadata must contain parameter_order.')
    if mapping_source == 'legacy_log_inference':
        import warnings
        warnings.warn(
            'Inferred backend parameter order from legacy JSON fit summaries '
            'and their saved dictionary order (%d columns). The mapping is '
            'recorded in samples_manifest.json.' % ndim,
            UserWarning,
        )
    order = source['parameter_order']
    shared = source.get('shared_parameters', {})
    if not isinstance(order, list) or not order:
        raise ValueError('parameter_order must be a non-empty list.')
    if not isinstance(shared, dict):
        raise ValueError('shared_parameters must be a dictionary.')

    def check_name(name):
        if not isinstance(name, str) or len(name.split('.')) != 2:
            raise ValueError('Parameter names must use category.parameter: %r' % (name,))
        category, parameter = name.split('.')
        if category not in template or parameter not in template[category]:
            raise ValueError('Parameter %s is absent from the template.' % name)

    destinations = set()
    for name in order:
        check_name(name)
        if name in destinations:
            raise ValueError('Duplicate parameter column: %s' % name)
        destinations.add(name)
    for name, targets in shared.items():
        if name not in order:
            raise ValueError('Shared source %s is not a sampled parameter.' % name)
        if not isinstance(targets, list) or not targets:
            raise ValueError('Shared destinations must be non-empty lists.')
        for target in targets:
            check_name(target)
            if target in destinations:
                raise ValueError('Parameter %s has multiple sources.' % target)
            destinations.add(target)
    return {
        'parameter_order': order,
        'shared_parameters': shared,
        'mapping_source': mapping_source,
    }


def read_backend_samples(filename, nsamples=100, nfrac=0.1, seed=12345):
    """Select complete vectors uniformly from completed tail steps and all walkers."""
    if nsamples < 1:
        raise ValueError('--nsamples must be a positive integer.')
    if not 0 < nfrac <= 1:
        raise ValueError('--nfrac must be in (0, 1].')
    if seed < 0:
        raise ValueError('--seed must be a non-negative integer.')

    # Backend-only dependency; ordinary parfile generation does not import h5py.
    import h5py

    path = Path(filename).resolve()
    with h5py.File(str(path), 'r') as handle:
        group = handle['mcmc']
        chain = group['chain']
        nsteps = int(group.attrs['iteration'])
        if chain.ndim != 3 or not 0 < nsteps <= chain.shape[0]:
            raise ValueError('Backend has no valid completed chain steps.')
        _, nwalkers, ndim = chain.shape
        if nwalkers < 1 or ndim < 1:
            raise ValueError('Backend has no walkers or parameters.')
        if (int(group.attrs['nwalkers']) != nwalkers
                or int(group.attrs['ndim']) != ndim):
            raise ValueError('Backend dimensions disagree with its metadata.')
        nkeep = int(np.ceil(nfrac * nsteps))
        start = nsteps - nkeep
        count = nkeep * nwalkers
        if nsamples > count:
            raise ValueError('--nsamples exceeds the %d available tail states.' % count)
        probabilities = group['log_prob']
        if (probabilities.ndim != 2 or probabilities.shape[0] < nsteps
                or probabilities.shape[1] != nwalkers):
            raise ValueError('Backend log probabilities do not match the chain.')
        samples = chain[start:nsteps].reshape(count, ndim)
        log_prob = probabilities[start:nsteps].reshape(-1)
        if not np.isfinite(samples).all() or not np.isfinite(log_prob).all():
            raise ValueError('Selected tail contains non-finite parameters/log probabilities.')

    rng = np.random.default_rng(seed)
    indices = rng.choice(count, size=nsamples, replace=False)
    records = [
        {
            'step': int(start + index // nwalkers),
            'walker': int(index % nwalkers),
            'log_prob': float(log_prob[index]),
        }
        for index in indices
    ]
    provenance = {
        'backend': str(path),
        'completed_steps': nsteps,
        'nwalkers': nwalkers,
        'ndim': ndim,
        'nsamples': nsamples,
        'nfrac': nfrac,
        'seed': seed,
        'rng': 'numpy.default_rng/PCG64',
        'numpy_version': np.__version__,
        'first_step': start,
        'last_step': nsteps - 1,
        'steps_per_walker': nkeep,
        'candidate_states': count,
        'index_base': 0,
        'replacement': False,
    }
    return samples[indices], records, provenance


def apply_draw(parameters, values, mapping):
    """Update only sampled parameters and their explicitly shared destinations."""
    if len(values) != len(mapping['parameter_order']):
        raise ValueError('Backend dimensionality does not match parameter_order.')
    for name, value in zip(mapping['parameter_order'], values):
        for destination in [name] + mapping['shared_parameters'].get(name, []):
            category, parameter = destination.split('.')
            parameters[category][parameter] = float(value)
