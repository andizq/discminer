"""Benchmark cached, block-wise beam FFT convolution.

This script reports two distinct measurements:

1. Isolated convolution of a synthetic channel cube.
2. Complete ``Model.get_cube()`` generation using precomputed model fields.

Existing ``astropy``, ``astropy_fft``, ``scipy``, and ``scipy_fft`` backends
are not modified. The optimized backend is named ``scipy_fft_cached``.

Example:

    python benchmark_beam_fft.py \
        --npix 128 \
        --nchan 51 \
        --block-sizes 1 4 8 16 \
        --save-fits benchmark_beam_fits
"""

from argparse import ArgumentParser
from contextlib import redirect_stdout
import gc
import io
import os
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter
import warnings

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from radio_beam import Beam
from scipy.signal import fftconvolve

from discminer.disc2d import _BeamFFTConvolver
from discminer.model import ReferenceModel


def block_size_argument(value):
    if str(value).lower() == "auto":
        return "auto"
    parsed = int(value)
    if parsed < 1:
        raise ValueError("block size must be at least 1")
    return parsed


def median_runtime(function, repeat, before_each=None):
    elapsed = []
    result = None
    for _ in range(repeat):
        if before_each is not None:
            before_each()
        gc.collect()
        start = perf_counter()
        result = function()
        elapsed.append(perf_counter() - start)
    return median(elapsed), elapsed, result


def convolve_scipy_fft(cube, kernel):
    kernel_array = kernel.array if hasattr(kernel, "array") else kernel
    return np.stack(
        [
            fftconvolve(channel, kernel_array, mode="same")
            for channel in cube
        ]
    )


def convolve_cached(cube, convolver, block_size):
    result = np.empty_like(cube, dtype=np.result_type(cube, float))
    for start in range(0, cube.shape[0], block_size):
        stop = min(start + block_size, cube.shape[0])
        result[start:stop] = convolver(cube[start:stop])
    return result


def convolve_cached_rebuild(cube, kernel, block_size):
    convolver = _BeamFFTConvolver(kernel, cube.shape[-2:])
    return convolve_cached(cube, convolver, block_size)


def comparison(reference, candidate):
    equivalent = np.allclose(
        reference,
        candidate,
        rtol=1e-11,
        atol=1e-11,
        equal_nan=True,
    )
    finite = np.isfinite(reference) & np.isfinite(candidate)
    max_difference = (
        np.max(np.abs(reference[finite] - candidate[finite]))
        if np.any(finite)
        else 0.0
    )
    return equivalent, max_difference


def approximate_working_memory_mib(convolver, block_size, dtype):
    """Conservative estimate of the largest FFT block temporaries."""
    real_bytes = np.dtype(dtype).itemsize
    complex_bytes = np.dtype(np.complex128).itemsize
    fy, fx = convolver.fft_shape
    rfft_elements = fy * (fx // 2 + 1)
    real_elements = fy * fx
    image_elements = np.prod(convolver.image_shape)

    # Kernel FFT, image FFT, multiplication temporary, inverse FFT buffer,
    # input block, and cropped output.
    total = convolver.kernel_fft.nbytes
    total += 2 * block_size * rfft_elements * complex_bytes
    total += block_size * real_elements * real_bytes
    total += 2 * block_size * image_elements * real_bytes
    return total / 1024**2


def validate_kernel_representations(cube, gaussian_kernel, rng):
    custom_odd = rng.normal(size=(7, 5))
    custom_even = rng.normal(size=(6, 4))
    kernels = {
        "Astropy Gaussian2DKernel": gaussian_kernel,
        "NumPy Gaussian array": gaussian_kernel.array.copy(),
        "NumPy custom odd": custom_odd,
        "NumPy custom even": custom_even,
    }

    print("Kernel representation validation")
    print("--------------------------------")
    validation_cube = cube[:min(3, cube.shape[0])]
    for name, kernel in kernels.items():
        expected = convolve_scipy_fft(validation_cube, kernel)
        convolver = _BeamFFTConvolver(kernel, cube.shape[-2:])
        actual = convolver(validation_cube)
        equivalent, max_difference = comparison(expected, actual)
        kernel_array = kernel.array if hasattr(kernel, "array") else kernel
        print(
            f"{name:<27} shape={str(kernel_array.shape):<10} "
            f"equivalent={str(equivalent):<5} "
            f"max|difference|={max_difference:.3e}"
        )
    print()


def make_benchmark_model(npix, nchan, beam_arcsec):
    channels = np.linspace(-2.0, 2.0, nchan)
    beam = Beam(
        major=beam_arcsec * u.arcsec,
        minor=0.8 * beam_arcsec * u.arcsec,
        pa=25 * u.deg,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = Path.cwd()
        try:
            os.chdir(tmpdir)
            with redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", Warning)
                reference = ReferenceModel(
                    npix=npix,
                    vchannels=channels,
                    Rmax=500 * u.au,
                    Rmin=0,
                    beam=beam,
                    convolve_func="scipy_fft",
                    write_extent=False,
                    filename="benchmark_reference.fits",
                )
        finally:
            os.chdir(old_cwd)

    model = reference.model
    model.prototype = False
    model.verbose = False
    properties = model.make_model()
    return model, properties


def configure_backend(model, backend, block_size):
    model.beam_convolve_backend = backend
    model.beam_convolve_func = model._get_beam_convolve_func(backend)
    model.beam_fft_block_size = block_size


def model_cube(model, properties):
    return model.get_cube(
        model.vchannels,
        *properties,
        return_data_only=True,
    )


def benchmark_model_cube(model, properties, block_sizes, repeat, save_block):
    configure_backend(model, "scipy_fft", 1)
    baseline_median, baseline_times, baseline_cube = median_runtime(
        lambda: model_cube(model, properties),
        repeat,
    )

    def invalidate_cache(model=model):
        model._beam_fft_convolver = None
        model._beam_fft_cache_key = None

    results = {
        "baseline_median": baseline_median,
        "baseline_times": baseline_times,
        "baseline_cube": baseline_cube,
        "blocks": {},
        "saved_cube": None,
    }
    for block_size in block_sizes:
        configure_backend(model, "scipy_fft_cached", block_size)
        invalidate_cache()
        model._get_beam_fft_convolver(
            np.shape(properties[1]["upper"])
        )
        fixed_median, fixed_times, fixed_cube = median_runtime(
            lambda: model_cube(model, properties),
            repeat,
        )
        rebuild_median, rebuild_times, rebuild_cube = median_runtime(
            lambda: model_cube(model, properties),
            repeat,
            before_each=invalidate_cache,
        )

        fixed_equivalent, fixed_difference = comparison(
            baseline_cube,
            fixed_cube,
        )
        rebuild_equivalent, rebuild_difference = comparison(
            baseline_cube,
            rebuild_cube,
        )
        results["blocks"][block_size] = {
            "selected_block_size": model.beam_fft_last_block_size,
            "fixed_median": fixed_median,
            "fixed_times": fixed_times,
            "fixed_equivalent": fixed_equivalent,
            "fixed_difference": fixed_difference,
            "rebuild_median": rebuild_median,
            "rebuild_times": rebuild_times,
            "rebuild_equivalent": rebuild_equivalent,
            "rebuild_difference": rebuild_difference,
        }
        if block_size == save_block:
            results["saved_cube"] = fixed_cube

    return results


def save_fits_cubes(model, baseline_cube, cached_cube, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "cube_scipy_fft.fits"
    cached_path = output_dir / "cube_scipy_fft_cached.fits"

    baseline_header = model.header.copy()
    baseline_header["HISTORY"] = "Channel-wise scipy.signal.fftconvolve"
    cached_header = model.header.copy()
    cached_header["HISTORY"] = "Cached block-wise spatial beam FFT"

    fits.writeto(
        baseline_path,
        baseline_cube,
        header=baseline_header,
        overwrite=True,
    )
    fits.writeto(
        cached_path,
        cached_cube,
        header=cached_header,
        overwrite=True,
    )
    return baseline_path, cached_path


def format_times(times):
    return " ".join(f"{value:.4f}s" for value in times)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--npix", type=int, default=128)
    parser.add_argument("--nchan", type=int, default=51)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
    )
    parser.add_argument(
        "--fits-block-size",
        type=block_size_argument,
        default="auto",
        help=(
            "Block size for the end-to-end benchmark and saved FITS cube; "
            "defaults to auto."
        ),
    )
    parser.add_argument(
        "--kernel-sigma-pixels",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--beam-arcsec",
        type=float,
        default=0.15,
        help="Beam major axis used by the end-to-end reference model.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--save-fits",
        type=Path,
        metavar="DIR",
    )
    args = parser.parse_args()

    if args.npix < 2:
        parser.error("--npix must be at least 2")
    if args.nchan < 2:
        parser.error("--nchan must be at least 2")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if any(size < 1 for size in args.block_sizes):
        parser.error("--block-sizes values must be at least 1")
    rng = np.random.default_rng(args.seed)
    synthetic_cube = rng.normal(
        size=(args.nchan, args.npix, args.npix)
    )
    gaussian_kernel = Gaussian2DKernel(args.kernel_sigma_pixels)

    print(
        f"Synthetic cube: {synthetic_cube.shape}; "
        f"kernel: {gaussian_kernel.array.shape}; repeats: {args.repeat}"
    )
    print(
        "All FFTs below operate only on spatial axes (-2, -1); "
        "velocity channels are independent."
    )
    print()
    validate_kernel_representations(
        synthetic_cube,
        gaussian_kernel,
        rng,
    )

    print("Isolated convolution benchmark")
    print("------------------------------")
    baseline_median, baseline_times, baseline = median_runtime(
        lambda: convolve_scipy_fft(
            synthetic_cube,
            gaussian_kernel,
        ),
        args.repeat,
    )
    print(f"SciPy FFT per channel: {format_times(baseline_times)}")
    print(f"SciPy FFT median:      {baseline_median:.4f}s")
    print()
    print(
        "block  cached FFT   speed-up  rebuild/cube  speed-up  "
        "approx memory  equivalent  max|difference|"
    )

    isolated_results = {}
    for block_size in args.block_sizes:
        convolver = _BeamFFTConvolver(
            gaussian_kernel,
            synthetic_cube.shape[-2:],
        )
        fixed_median, _, fixed = median_runtime(
            lambda: convolve_cached(
                synthetic_cube,
                convolver,
                block_size,
            ),
            args.repeat,
        )
        rebuild_median, _, rebuild = median_runtime(
            lambda: convolve_cached_rebuild(
                synthetic_cube,
                gaussian_kernel,
                block_size,
            ),
            args.repeat,
        )
        fixed_equivalent, fixed_difference = comparison(
            baseline,
            fixed,
        )
        rebuild_equivalent, rebuild_difference = comparison(
            baseline,
            rebuild,
        )
        equivalent = fixed_equivalent and rebuild_equivalent
        max_difference = max(fixed_difference, rebuild_difference)
        memory = approximate_working_memory_mib(
            convolver,
            block_size,
            synthetic_cube.dtype,
        )
        isolated_results[block_size] = {
            "fixed_median": fixed_median,
            "rebuild_median": rebuild_median,
        }
        print(
            f"{block_size:>5}  "
            f"{fixed_median:>10.4f}s  "
            f"{baseline_median / fixed_median:>7.2f}x  "
            f"{rebuild_median:>11.4f}s  "
            f"{baseline_median / rebuild_median:>7.2f}x  "
            f"{memory:>10.1f} MiB  "
            f"{str(equivalent):>10}  "
            f"{max_difference:.3e}"
        )

    print()
    print("Complete Model.get_cube() benchmark")
    print("-----------------------------------")
    print(
        "This includes line-profile generation and surface merging as well "
        "as beam convolution."
    )
    model, properties = make_benchmark_model(
        args.npix,
        args.nchan,
        args.beam_arcsec,
    )
    save_block = (
        args.fits_block_size
        if args.save_fits is not None
        else None
    )
    model_block_sizes = list(dict.fromkeys(
        [
            *args.block_sizes,
            "auto",
            *([] if save_block is None else [save_block]),
        ]
    ))
    model_results = benchmark_model_cube(
        model,
        properties,
        model_block_sizes,
        args.repeat,
        save_block,
    )

    print(
        "SciPy FFT per-channel times: ",
        format_times(model_results["baseline_times"]),
    )
    print(
        f"SciPy FFT median:             "
        f"{model_results['baseline_median']:.4f}s"
    )
    print()
    print(
        "block  fixed beam   speed-up  rebuild/cube  speed-up  "
        "equivalent  max|difference|"
    )
    for block_size in model_block_sizes:
        block_result = model_results["blocks"][block_size]
        block_label = (
            f"auto→{block_result['selected_block_size']}"
            if block_size == "auto"
            else str(block_size)
        )
        equivalent = (
            block_result["fixed_equivalent"]
            and block_result["rebuild_equivalent"]
        )
        max_difference = max(
            block_result["fixed_difference"],
            block_result["rebuild_difference"],
        )
        print(
            f"{block_label:>7}  "
            f"{block_result['fixed_median']:>9.4f}s  "
            f"{model_results['baseline_median'] / block_result['fixed_median']:>7.2f}x  "
            f"{block_result['rebuild_median']:>11.4f}s  "
            f"{model_results['baseline_median'] / block_result['rebuild_median']:>7.2f}x  "
            f"{str(equivalent):>10}  "
            f"{max_difference:.3e}"
        )

    if args.save_fits is not None:
        baseline_path, cached_path = save_fits_cubes(
            model,
            model_results["baseline_cube"],
            model_results["saved_cube"],
            args.save_fits,
        )
        print(f"Baseline FITS:                {baseline_path.resolve()}")
        print(f"Cached FITS:                  {cached_path.resolve()}")

    print()
    print(
        "Note: fixed-beam timing reuses the kernel FFT across cubes. "
        "Rebuild/cube timing includes one kernel FFT per cube, representing "
        "a fitted beam that changes once per likelihood evaluation."
    )
    print(
        "Larger blocks are not guaranteed to be faster. Automatic selection "
        "targets a small FFT working set; explicit integer blocks remain "
        "available for machine-specific tuning."
    )


if __name__ == "__main__":
    main()
