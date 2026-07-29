"""Benchmark the original and cleaned channel-loop likelihoods.

The default test runs in the current process:

    python benchmark_likelihood.py

Use ``--workers`` to include multiprocessing serialization and contention:

    python benchmark_likelihood.py --workers 14
    python benchmark_likelihood.py --npix 128 --workers 14 --repeat 10

Model generation is excluded. Synthetic non-finite data and inner/outer
model regions verify that both implementations use the same valid pixels.
"""

from argparse import ArgumentParser
import gc
from multiprocessing import Pool
from statistics import median
from time import perf_counter

import numpy as np


def legacy_likelihood(data_cube, model_cube, noise):
    """Original likelihood implementation with redundant temporaries."""
    lnlike = 0.0
    for channel in range(data_cube.shape[0]):
        model_channel = model_cube[channel]
        mask_data = np.isfinite(data_cube[channel])
        mask_model = np.isfinite(model_channel)
        data = np.where(
            np.logical_and(mask_model, ~mask_data),
            0,
            data_cube[channel],
        )
        model = np.where(
            np.logical_and(mask_data, ~mask_model),
            0,
            model_channel,
        )
        mask = np.logical_and(mask_data, mask_model)
        squared_residual = np.where(
            mask,
            np.power((data - model) / noise, 2),
            0,
        )
        lnlike += -0.5 * np.sum(squared_residual)

    return lnlike if np.isfinite(lnlike) else -np.inf


def clean_likelihood(data_cube, model_cube, noise):
    """Clean channel loop used by the current implementation."""
    lnlike = 0.0
    for channel in range(data_cube.shape[0]):
        data_channel = data_cube[channel]
        model_channel = model_cube[channel]
        valid = (
            np.isfinite(data_channel)
            & np.isfinite(model_channel)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            squared_residual = np.where(
                valid,
                np.square(
                    (data_channel - model_channel) / noise
                ),
                0.0,
            )

        lnlike += -0.5 * np.sum(squared_residual)

    return lnlike if np.isfinite(lnlike) else -np.inf


class LikelihoodWorkload:
    """Picklable workload used by the multiprocessing benchmark."""

    def __init__(self, data, model, noise):
        self.data = data
        self.model = model
        self.noise = noise

    def legacy(self, unused):
        return legacy_likelihood(self.data, self.model, self.noise)

    def clean(self, unused):
        return clean_likelihood(self.data, self.model, self.noise)


def timed_pair(first, second, repeat, warmup):
    for _ in range(warmup):
        first()
        second()

    elapsed = [[], []]
    functions = (first, second)
    for repetition in range(repeat):
        order = (0, 1) if repetition % 2 == 0 else (1, 0)
        for index in order:
            gc.collect()
            start = perf_counter()
            functions[index]()
            elapsed[index].append(perf_counter() - start)
    return elapsed


def make_inputs(args):
    rng = np.random.default_rng(args.seed)
    shape = (args.nchan, args.npix, args.npix)
    data = rng.normal(size=shape)
    model = data + args.residual_stddev * rng.normal(size=shape)
    noise = args.noise_stddev * (
        0.8 + 0.4 * rng.random((args.npix, args.npix))
    )

    data[rng.random(shape) < args.data_nan_fraction] = np.nan

    y, x = np.ogrid[:args.npix, :args.npix]
    centre = 0.5 * (args.npix - 1)
    radius = np.hypot(x - centre, y - centre) / args.npix
    model_mask = (
        (radius < args.inner_mask_radius)
        | (radius > args.outer_mask_radius)
    )
    model[:, model_mask] = np.nan
    return data, model, noise


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--npix", type=int, default=60)
    parser.add_argument("--nchan", type=int, default=101)
    parser.add_argument("--repeat", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes used in each timed batch.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        help="Likelihood calls per timed batch; defaults to 4 per worker.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-stddev", type=float, default=0.1)
    parser.add_argument("--residual-stddev", type=float, default=0.02)
    parser.add_argument("--data-nan-fraction", type=float, default=0.02)
    parser.add_argument("--inner-mask-radius", type=float, default=0.08)
    parser.add_argument("--outer-mask-radius", type=float, default=0.47)
    args = parser.parse_args()

    data, model, noise = make_inputs(args)
    workload = LikelihoodWorkload(data, model, noise)
    tasks = args.tasks or max(1, 4 * args.workers)
    inputs = range(tasks)

    legacy_value = workload.legacy(None)
    clean_value = workload.clean(None)
    equivalent = np.isclose(
        legacy_value,
        clean_value,
        rtol=1e-12,
        atol=1e-10,
    )

    if args.workers == 1:
        def legacy():
            for value in inputs:
                workload.legacy(value)

        def clean():
            for value in inputs:
                workload.clean(value)

        legacy_times, clean_times = timed_pair(
            legacy,
            clean,
            args.repeat,
            args.warmup,
        )
    else:
        with Pool(args.workers) as pool:
            def legacy():
                pool.map(workload.legacy, inputs)

            def clean():
                pool.map(workload.clean, inputs)

            legacy_times, clean_times = timed_pair(
                legacy,
                clean,
                args.repeat,
                args.warmup,
            )

    legacy_median = median(legacy_times) / tasks
    clean_median = median(clean_times) / tasks

    print(
        f"Cube: {args.nchan} x {args.npix} x {args.npix}; "
        f"workers: {args.workers}; tasks: {tasks}"
    )
    print(f"Legacy median per call:     {legacy_median:.6f}s")
    print(f"Clean median per call:      {clean_median:.6f}s")
    print(
        "Clean-loop speed-up:        "
        f"{legacy_median / clean_median:.2f}x"
    )
    print(f"Equivalent:                 {equivalent}")
    print(
        "Absolute likelihood diff:   "
        f"{abs(legacy_value - clean_value):.3e}"
    )


if __name__ == "__main__":
    main()
