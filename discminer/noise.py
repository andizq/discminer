"""
Noise models for the discminer likelihood
=========================================
Functions: estimate_inv_psd, validate_finite
"""

import warnings

import numpy as np

from .tools.utils import InputError

__all__ = ['estimate_inv_psd', 'validate_finite']

_fast_log_likelihood_func = None

NCHAN_PSD_MIN = 5
NCHAN_PSD_RECOMMENDED = 20


def _beam_boozle():
    #Imported lazily so this module loads without the optional dependency, and so the
    # import also runs inside multiprocessing workers
    try:
        import beam_boozle
        import beam_boozle.utils
    except ImportError as e:
        raise ImportError(
            "Correlated-noise likelihoods require the 'beam-boozle' package, which is not on "
            "PyPI yet. Install it from source:\n"
            "    pip install git+https://github.com/tomhilder/fast_corr_likelihoods"
        ) from e
    return beam_boozle


def get_fast_log_likelihood():
    """Return beam_boozle's FFT log-likelihood, caching the lookup between calls."""
    global _fast_log_likelihood_func
    if _fast_log_likelihood_func is None:
        _fast_log_likelihood_func = _beam_boozle().fast_log_likelihood
    return _fast_log_likelihood_func


def validate_finite(data, name='data'):
    """
    Raise if *data* contains non-finite pixels.

    The correlated-noise likelihood is evaluated with FFTs, which cannot skip pixels, and
    zero-filling them biases uncertainties badly, so the fitted region must be fully finite.

    Parameters
    ----------
    data : array_like
        Array to check. Non-finite pixels are collapsed over all but the last two axes.

    name : str, optional
        Name used in the error message.
    """
    data = np.asarray(data)
    bad = ~np.isfinite(data)
    if not bad.any():
        return

    frac = 100.0 * bad.sum() / bad.size
    bad2d = bad.any(axis=tuple(range(data.ndim - 2))) if data.ndim > 2 else bad
    npix = _largest_finite_clip(bad2d)

    if npix > 0:
        hint = ('The largest fully finite window centred on the image is %dx%d pixels; '
                'try datacube.clip(npix=%d).' % (2 * npix, 2 * npix, npix))
    else:
        hint = ('The non-finite pixels are not confined to the border, so clipping to a '
                'centred window will not remove them.')

    raise InputError(
        name,
        'Found %.2f%% non-finite pixels. The correlated-noise likelihood requires a fully '
        'finite rectangular region. %s Alternatively, pass noise_stddev=... to use the '
        'uncorrelated likelihood, which does mask non-finite pixels.' % (frac, hint)
    )


def _largest_finite_clip(bad2d):
    #Largest npix whose centred 2*npix x 2*npix window contains no bad pixel, matching the
    # convention of Cube.clip. Returns 0 if there is no such window. The summed-area table
    # makes each candidate window O(1) to test.
    ny, nx = bad2d.shape
    integral = np.zeros((ny + 1, nx + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(bad2d.astype(np.int64), axis=0), axis=1)
    ic, jc = ny // 2, nx // 2

    for npix in range(min(ic, jc, ny - ic, nx - jc), 0, -1):
        y0, y1 = ic - npix, ic + npix
        x0, x1 = jc - npix, jc + npix
        if integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0] == 0:
            return npix
    return 0


def estimate_inv_psd(datacube, channels=None, white_floor=1e-2, smooth=3, mask=None):
    """
    Estimate the inverse noise power spectral density from line-free channels.

    Pass the result to `~discminer.disc2d.Model.run_mcmc` as *noise_psd_inv* to account for the
    correlation the beam introduces between neighbouring pixels. The noise is assumed stationary
    and identically distributed across channels; its correlation structure is measured rather
    than modelled, so nothing is assumed about the shape of the beam.

    Parameters
    ----------
    datacube : `~discminer.core.Data` or `~discminer.cube.Cube`
        Datacube to take the line-free channels from. Must be on the same pixel grid as the cube
        being fitted, so clip or downsample first and pass the result here.

    channels : array_like, optional
        Indices of the line-free channels. Defaults to the first and last five.

    white_floor : float, optional
        Fraction of the noise variance treated as uncorrelated. Regularises the inversion, since
        the PSD falls to nearly zero at spatial frequencies suppressed by the beam. Results are
        insensitive to it over several orders of magnitude. Defaults to 1e-2.

    smooth : int, optional
        Side length in pixels of a boxcar applied to the PSD before inversion, which reduces the
        bias incurred when estimating from few channels. Must be odd, or None to disable. Keep it
        small: heavy smoothing washes out real structure. Defaults to 3.

    mask : array_like, optional
        Binary mask with shape (nx, ny): 1 to include a pixel, 0 to exclude it. Applies to the
        PSD estimate only, e.g. to keep faint line wings out of it. The likelihood itself cannot
        be masked and requires fully finite data (see `~discminer.noise.validate_finite`).

    Returns
    -------
    inv_psd : `numpy.ndarray`
        Inverse PSD with shape (nx, ny), on the unshifted FFT grid.
    """
    bb = _beam_boozle()

    data = np.asarray(datacube.data, dtype=np.float64)
    if data.ndim != 3:
        raise InputError(data.shape, 'Input datacube must have shape (nchan, nx, ny).')

    if channels is None:
        nchan = data.shape[0]
        if nchan < 2 * NCHAN_PSD_MIN:
            raise InputError(nchan,
                             'Cube has too few channels (%d) to take the default first and '
                             'last five as line-free. Specify channels explicitly.' % nchan)
        channels = np.r_[0:5, nchan - 5:nchan]

    noise_images = data[np.atleast_1d(np.asarray(channels))]
    nused = len(noise_images)

    if nused < NCHAN_PSD_MIN:
        raise InputError(nused,
                         'At least %d line-free channels are required to estimate the noise '
                         'PSD (got %d).' % (NCHAN_PSD_MIN, nused))
    if nused < NCHAN_PSD_RECOMMENDED:
        warnings.warn('Estimating the noise PSD from only %d channels biases the inverse PSD '
                      'high, making posteriors somewhat overconfident. %d or more line-free '
                      'channels are recommended.' % (nused, NCHAN_PSD_RECOMMENDED))

    validate_finite(noise_images, name='line-free channels')

    return bb.utils.estimate_noise_inv_psd_from_data(
        noise_images, mask=mask, mode='white', eps=white_floor, smooth=smooth
    )
