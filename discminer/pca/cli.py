"""Command-line interface for discminer's PCA workflow."""

from pathlib import Path

from astropy import units as u


def _component_values(values):
    if values is None:
        return None
    components = []
    for value in values:
        components.extend(
            int(item.strip())
            for item in str(value).split(",")
            if item.strip()
        )
    return components


def _without_pca_prefix(stem):
    return stem[4:] if stem.startswith("pca_") else stem


def _default_output(input_path, product, extension):
    input_path = Path(input_path)
    stem = _without_pca_prefix(input_path.stem)
    prefix = "pca" if product is None else f"pca_{product}"
    return input_path.with_name(f"{prefix}_{stem}{extension}")


def _load_parfile_context(parfile):
    parfile = Path(parfile)
    if not parfile.exists():
        return {}

    from discminer.mining_utils import load_parfile

    metadata, parameters, _ = load_parfile(parfile=str(parfile))
    return {
        "distance_pc": float(metadata["dpc"]),
        "outer_radius_au": float(parameters["intensity"]["Rout"]),
        "systemic_velocity": float(parameters["velocity"]["vsys"]),
    }


def _resolve_context(args):
    context = _load_parfile_context(args.parfile)
    distance_pc = getattr(args, "distance_pc", None)
    if distance_pc is None:
        distance_pc = context.get("distance_pc")
    if distance_pc is None:
        raise ValueError(
            "No source distance is available. Provide --distance-pc or a "
            "discminer parfile with metadata.dpc."
        )
    context["distance_pc"] = distance_pc
    return context


def add_pca_parser(subparsers):
    """Register ``discminer pca`` and its nested subcommands."""

    parser = subparsers.add_parser(
        "pca",
        help="Decompose, inspect, and reconstruct stacked cubes with PCA",
        description="Principal-component analysis of data cubes.",
    )
    commands = parser.add_subparsers(dest="pca_command", required=True)

    run = commands.add_parser(
        "run", help="Run PCA and write one complete FITS artifact"
    )
    run.add_argument("input", help="Input FITS cube")
    run.add_argument(
        "-o",
        "--output",
        help=(
            "Output PCA artifact. Default: pca_<input>.fits. The covariance "
            "plot defaults to pca_covariance_<input>.png."
        ),
    )
    run.add_argument(
        "--parfile",
        default="parfile.json",
        help="discminer parfile used for distance and outer radius",
    )
    run.add_argument(
        "--distance-pc",
        type=float,
        default=None,
        help="Source distance in pc; overrides the parfile value",
    )
    run.add_argument(
        "-neigs",
        "--n-components",
        type=int,
        default=-1,
        help="Components used for width measurements; -1 uses all",
    )
    run.add_argument(
        "--mean-sub",
        action="store_true",
        help="Subtract channel means before decomposition",
    )
    run.add_argument(
        "--spatial-method",
        default="contour",
        choices=["contour", "fit", "interpolate", "xinterpolate"],
    )
    run.add_argument(
        "--spectral-method",
        default="walk-down",
        choices=["walk-down", "fit", "interpolate"],
    )
    run.add_argument(
        "--no-beam-correct",
        action="store_true",
        help="Disable TurbuStat's spatial beam correction",
    )
    run.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the covariance progress display",
    )
    run.add_argument(
        "--cov-central-fraction",
        type=float,
        default=0.4,
        help="Fraction of the velocity span shown in the covariance plot",
    )
    run.add_argument(
        "--cov-vlim",
        type=float,
        default=None,
        help="Optional covariance half-width in km/s",
    )
    run.add_argument("--dpi", type=int, default=200)
    run.add_argument("--show", action="store_true")
    run.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Fail instead of replacing existing output files",
    )
    run.set_defaults(overwrite=True)

    components = commands.add_parser(
        "plot-components", help="Plot selected eigenimages"
    )
    components.add_argument("artifact", help="Input PCA artifact")
    components.add_argument(
        "-c",
        "--components",
        nargs="+",
        required=True,
        help="Zero-based components, separated by spaces or commas",
    )
    components.add_argument(
        "-o",
        "--output",
        help="Output figure. Default: pca_components_<input>.png",
    )
    components.add_argument("--cmap", default="RdBu_r")
    components.add_argument("--robust", action="store_true")
    components.add_argument("--percentile", type=float, default=99.5)
    components.add_argument("--share-scale", action="store_true")
    components.add_argument("--dpi", type=int, default=200)
    components.add_argument("--show", action="store_true")

    widths = commands.add_parser(
        "plot-widths", help="Plot spectral widths against spatial widths"
    )
    widths.add_argument("artifact", help="Input PCA artifact")
    widths.add_argument(
        "-o",
        "--output",
        help="Output figure. Default: pca_widths_<input>.png",
    )
    widths.add_argument(
        "-i",
        "--beam-multiple",
        "--Rinner",
        dest="beam_multiple",
        type=float,
        default=1.0,
        help=(
            "Minimum resolved spatial width in major-axis beam FWHM units. "
            "Default: 1"
        ),
    )
    widths.add_argument(
        "-ne",
        "--n-fit-components",
        "--neigs",
        dest="n_fit_components",
        type=int,
        default=6,
        help=(
            "Number of leading PCA components considered for the fit. "
            "Default: 6"
        ),
    )
    widths.add_argument(
        "--spectral-error-scale",
        type=float,
        default=0.2,
        help=(
            "Multiplier applied to TurbuStat spectral-width errors, matching "
            "the original PCA width script. Default: 0.2"
        ),
    )
    widths.add_argument("--dpi", type=int, default=200)
    widths.add_argument("--show", action="store_true")

    diagnostics = commands.add_parser(
        "plot-diagnostics",
        help="Plot spatial and spectral width-fitting diagnostics",
    )
    diagnostics.add_argument("artifact", help="Input PCA artifact")
    diagnostics.add_argument(
        "--spatial-output",
        help=(
            "Output spatial figure. "
            "Default: pca_spatialwidths_<input>.png"
        ),
    )
    diagnostics.add_argument(
        "--spectral-output",
        help=(
            "Output spectral figure. "
            "Default: pca_spectralwidths_<input>.png"
        ),
    )
    diagnostics.add_argument(
        "-n",
        "--n-components",
        type=int,
        default=9,
        help="Number of leading components to plot, at most 9. Default: 9",
    )
    diagnostics.add_argument(
        "--max-lag",
        type=float,
        default=None,
        help=(
            "Maximum spectral lag in channels. "
            "Default: full non-negative lag range"
        ),
    )
    diagnostics.add_argument("--dpi", type=int, default=200)
    diagnostics.add_argument("--show", action="store_true")

    reconstruct = commands.add_parser(
        "reconstruct", help="Reconstruct a cube from selected components"
    )
    reconstruct.add_argument("artifact", help="Input PCA artifact")
    reconstruct.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Keep only these zero-based components",
    )
    reconstruct.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Exclude these zero-based components",
    )
    reconstruct.add_argument(
        "--exclude-tail",
        type=int,
        default=None,
        help="Exclude this zero-based component and every component after it",
    )
    reconstruct.add_argument(
        "-o",
        "--output",
        help="Output FITS cube. Default: pca_reconstructed_<input>.fits",
    )
    reconstruct.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Fail instead of replacing an existing reconstructed cube",
    )
    reconstruct.set_defaults(overwrite=True)

    channels = commands.add_parser(
        "plot-channels", help="Plot channel maps from a reconstructed cube"
    )
    channels.add_argument("cube", help="Input reconstructed FITS cube")
    channels.add_argument(
        "-o",
        "--output",
        help="Output figure. Default: pca_channels_<input>.png",
    )
    channels.add_argument("--parfile", default="parfile.json")
    channels.add_argument("--distance-pc", type=float, default=None)
    channels.add_argument("--chan-ids", nargs="+", type=int, default=None)
    channels.add_argument("--step", type=int, default=1)
    channels.add_argument("--n-channels", type=int, default=8)
    channels.add_argument(
        "--vsys",
        type=float,
        default=0.0,
        help="Velocity at the centre of the displayed channels in km/s",
    )
    channels.add_argument("--cmap", default="inferno")
    channels.add_argument("--vmin", type=float, default=None)
    channels.add_argument("--vmax", type=float, default=None)
    channels.add_argument("--xlim", type=float, default=None)
    channels.add_argument("--dpi", type=int, default=200)
    channels.add_argument("--show", action="store_true")

    return parser


def run_from_namespace(args):
    """Execute a parsed PCA subcommand."""

    from .artifact import (
        read_pca_artifact,
        write_pca_artifact,
        write_reconstructed_cube,
    )
    from .plotting import (
        plot_channels,
        plot_components,
        plot_covariance,
        plot_spatial_width_diagnostics,
        plot_spectral_width_diagnostics,
        plot_widths,
    )

    command = args.pca_command

    if command == "run":
        from .analysis import run_pca

        context = _resolve_context(args)
        output = (
            Path(args.output)
            if args.output
            else _default_output(args.input, None, ".fits")
        )
        result = run_pca(
            args.input,
            context["distance_pc"] * u.pc,
            n_components=args.n_components,
            mean_sub=args.mean_sub,
            spatial_method=args.spatial_method,
            spectral_method=args.spectral_method,
            beam_correct=not args.no_beam_correct,
            show_progress=not args.no_progress,
            outer_radius_au=context.get("outer_radius_au"),
        )
        write_pca_artifact(result, output, overwrite=args.overwrite)
        covariance_output = _default_output(
            output, "covariance", ".png"
        )
        plot_covariance(
            result,
            covariance_output,
            central_fraction=args.cov_central_fraction,
            velocity_limit=args.cov_vlim,
            dpi=args.dpi,
            show=args.show,
        )
        print(f"Wrote PCA artifact to {output}")
        print(f"Wrote covariance plot to {covariance_output}")
        return 0

    if command == "plot-components":
        result = read_pca_artifact(args.artifact)
        components = _component_values(args.components)
        output = (
            Path(args.output)
            if args.output
            else _default_output(args.artifact, "components", ".png")
        )
        plot_components(
            result,
            components,
            output,
            cmap=args.cmap,
            robust=args.robust,
            percentile=args.percentile,
            share_scale=args.share_scale,
            dpi=args.dpi,
            show=args.show,
        )
        print(f"Wrote component plot to {output}")
        return 0

    if command == "plot-widths":
        result = read_pca_artifact(args.artifact)
        output = (
            Path(args.output)
            if args.output
            else _default_output(args.artifact, "widths", ".png")
        )
        plot_widths(
            result,
            output,
            beam_multiple=args.beam_multiple,
            n_fit_components=args.n_fit_components,
            spectral_error_scale=args.spectral_error_scale,
            dpi=args.dpi,
            show=args.show,
        )
        print(f"Wrote width plot to {output}")
        return 0

    if command == "plot-diagnostics":
        result = read_pca_artifact(args.artifact)
        spatial_output = (
            Path(args.spatial_output)
            if args.spatial_output
            else _default_output(
                args.artifact,
                "spatialwidths",
                ".png",
            )
        )
        spectral_output = (
            Path(args.spectral_output)
            if args.spectral_output
            else _default_output(
                args.artifact,
                "spectralwidths",
                ".png",
            )
        )
        plot_spatial_width_diagnostics(
            result,
            spatial_output,
            n_components=args.n_components,
            dpi=args.dpi,
            show=args.show,
        )
        plot_spectral_width_diagnostics(
            result,
            spectral_output,
            n_components=args.n_components,
            max_lag=args.max_lag,
            dpi=args.dpi,
            show=args.show,
        )
        print(f"Wrote spatial-width diagnostics to {spatial_output}")
        print(f"Wrote spectral-width diagnostics to {spectral_output}")
        return 0

    if command == "reconstruct":
        result = read_pca_artifact(args.artifact)
        include = _component_values(args.include)
        exclude = _component_values(args.exclude)
        output = (
            Path(args.output)
            if args.output
            else _default_output(args.artifact, "reconstructed", ".fits")
        )
        write_reconstructed_cube(
            result,
            output,
            include=include,
            exclude=exclude,
            exclude_tail=args.exclude_tail,
            overwrite=args.overwrite,
        )
        print(f"Wrote reconstructed cube to {output}")
        return 0

    if command == "plot-channels":
        context = _resolve_context(args)
        output = (
            Path(args.output)
            if args.output
            else _default_output(args.cube, "channels", ".png")
        )
        plot_channels(
            args.cube,
            context["distance_pc"] * u.pc,
            output,
            channel_ids=args.chan_ids,
            step=args.step,
            n_channels=args.n_channels,
            systemic_velocity=args.vsys,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            xlim=args.xlim,
            dpi=args.dpi,
            show=args.show,
        )
        print(f"Wrote channel plot to {output}")
        return 0

    raise ValueError(f"Unknown PCA command: {command}")
