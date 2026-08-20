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


def _resolve_characterization_parfiles(
    artifacts,
    explicit_parfiles=None,
    circular_sky=False,
):
    if circular_sky:
        return [None] * len(artifacts)
    if explicit_parfiles is not None:
        parfiles = [Path(value) for value in explicit_parfiles]
        if len(parfiles) == 1:
            return parfiles * len(artifacts)
        if len(parfiles) != len(artifacts):
            raise ValueError(
                "--parfile must contain one shared path or one path per "
                "PCA artifact"
            )
        return parfiles

    cwd_parfile = Path("parfile.json")
    return [
        artifact.parent / "parfile.json"
        if (artifact.parent / "parfile.json").exists()
        else cwd_parfile if cwd_parfile.exists() else None
        for artifact in artifacts
    ]


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
        "plot-components", help="Plot selected eigenimages and eigenvectors"
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

    acf = commands.add_parser(
        "plot-acf",
        aliases=["plot-diagnostics"],
        help="Plot eigenimage and eigenvector autocorrelations",
    )
    acf.add_argument("artifact", help="Input PCA artifact")
    acf.add_argument(
        "--spatial-output",
        help=(
            "Output spatial figure. "
            "Default: pca_spatialacf_<input>.png"
        ),
    )
    acf.add_argument(
        "--spectral-output",
        help=(
            "Output spectral figure. "
            "Default: pca_spectralacf_<input>.png"
        ),
    )
    acf.add_argument(
        "-n",
        "--n-components",
        type=int,
        default=9,
        help="Number of leading components to plot, at most 9. Default: 9",
    )
    acf.add_argument(
        "--max-lag",
        type=float,
        default=None,
        help=(
            "Maximum spectral lag in channels. "
            "Default: full non-negative lag range"
        ),
    )
    acf.add_argument("--dpi", type=int, default=200)
    acf.add_argument("--show", action="store_true")

    characterize = commands.add_parser(
        "characterize",
        help="Measure variance and ACF morphology of PCA components",
    )
    characterize.add_argument(
        "artifacts",
        nargs="+",
        help="One or more PCA artifacts to characterize",
    )
    characterize.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels corresponding to the input artifacts",
    )
    selection = characterize.add_mutually_exclusive_group()
    selection.add_argument(
        "-c",
        "--components",
        nargs="+",
        default=None,
        help="Zero-based components, separated by spaces or commas",
    )
    selection.add_argument(
        "-n",
        "--n-components",
        type=int,
        default=9,
        help="Number of leading components to measure. Default: 9",
    )
    characterize.add_argument(
        "--acf-level",
        type=float,
        default=0.36787944117144233,
        help=(
            "Fraction of the ACF peak used for the ellipse contour and "
            "multipole core. Default: 1/e"
        ),
    )
    characterize.add_argument(
        "--azimuth-samples",
        type=int,
        default=360,
        help="Angular samples per radial ring. Default: 360",
    )
    characterize.add_argument(
        "--minimum-azimuthal-coverage",
        type=float,
        default=0.75,
        help=(
            "Minimum valid fraction of an eigenimage ring used for angular "
            "metrics. Default: 0.75"
        ),
    )
    characterize.add_argument(
        "--phase-minimum-relative-power",
        type=float,
        default=0.01,
        help=(
            "Minimum ring-mode power relative to its peak used for phase "
            "coherence. Default: 0.01"
        ),
    )
    characterize.add_argument(
        "--excursion-percentile",
        type=float,
        default=90.0,
        help=(
            "Percentile of absolute eigenimage amplitude defining the "
            "compactness and Euler excursion set. Default: 90"
        ),
    )
    characterize.add_argument(
        "--plot-excursions",
        action="store_true",
        help=(
            "Write an optional eigenimage grid showing the positive, "
            "negative, and combined excursion-set contours used for "
            "compactness"
        ),
    )
    characterize.add_argument(
        "--plot-m0-residuals",
        action="store_true",
        help=(
            "Write an optional eigenimage grid comparing each original PC, "
            "its projected m=0 field, and the m=0-subtracted residual"
        ),
    )
    characterize.add_argument(
        "--plot-mpeak-residuals",
        action="store_true",
        help=(
            "Write an optional grid comparing each m=0-subtracted PC, its "
            "dominant fitted non-axisymmetric mode, and the remaining "
            "non-axisymmetric residual"
        ),
    )
    residual_scaling = characterize.add_mutually_exclusive_group()
    residual_scaling.add_argument(
        "--robust",
        action="store_true",
        dest="mode_residual_robust",
        help=(
            "Use percentile-based symmetric scaling for mode-residual "
            "plots. This is the default"
        ),
    )
    residual_scaling.add_argument(
        "--no-robust",
        action="store_false",
        dest="mode_residual_robust",
        help="Use the full finite range for mode-residual plots",
    )
    characterize.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        dest="mode_residual_percentile",
        help=(
            "Absolute-amplitude percentile used by robust mode-residual "
            "scaling. Default: 99.5"
        ),
    )
    characterize.add_argument(
        "--maximum-angular-mode",
        type=int,
        default=6,
        help=(
            "Highest mode in the simultaneous eigenimage angular spectrum. "
            "Default: 6"
        ),
    )
    characterize.add_argument(
        "--angular-normalization",
        choices=("total", "nonaxisymmetric"),
        default="total",
        help=(
            "Power normalization shown in the eigenimage and angular-mode "
            "figures. Both variants are always stored in the ECSV table. "
            "Default: total"
        ),
    )
    ring_geometry = characterize.add_mutually_exclusive_group()
    ring_geometry.add_argument(
        "--parfile",
        nargs="+",
        default=None,
        help=(
            "One shared DiscMiner parfile or one per PCA artifact. By "
            "default, search beside each artifact and then in the working "
            "directory"
        ),
    )
    ring_geometry.add_argument(
        "--circular-deproj",
        "--circular-sky",
        action="store_true",
        dest="circular_deproj",
        help="Use legacy circular sky-plane rings instead of deprojection",
    )
    characterize.add_argument(
        "--deprojection-surface",
        "--surface",
        choices=("upper", "lower", "midplane"),
        default="upper",
        dest="deprojection_surface",
        help=(
            "Emission surface used to project disc-plane rings. "
            "Default: upper"
        ),
    )
    characterize.add_argument(
        "--plot-group",
        choices=("all", "variance", "acf", "eigenimage", "angularmode"),
        default="all",
        help="Retained diagnostic group to plot. Default: all",
    )
    characterize.add_argument(
        "--plot-prefix",
        help=(
            "Shared path prefix for retained diagnostic figures. Default: "
            "the ECSV output path without its suffix"
        ),
    )
    characterize.add_argument(
        "--include-pc0-cumulative",
        action="store_true",
        help="Include PC 0 in the cumulative-variance panel",
    )
    characterize.add_argument(
        "-o",
        "--output",
        help=(
            "Output ECSV table. For one artifact the default is "
            "pca_characterization_<input>.ecsv; for multiple artifacts it "
            "is pca_characterization.ecsv."
        ),
    )
    characterize.add_argument("--dpi", type=int, default=200)
    characterize.add_argument("--show", action="store_true")
    characterize.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Fail instead of replacing any existing output",
    )
    characterize.set_defaults(overwrite=True, mode_residual_robust=True)

    spectra = commands.add_parser(
        "characterize-spectra",
        help="Compare eigenspectra with empirical profile templates",
    )
    spectra.add_argument("artifact", help="Input PCA artifact")
    spectrum_selection = spectra.add_mutually_exclusive_group()
    spectrum_selection.add_argument(
        "-c",
        "--components",
        nargs="+",
        default=None,
        help="Zero-based components, separated by spaces or commas",
    )
    spectrum_selection.add_argument(
        "-n",
        "--n-components",
        type=int,
        default=9,
        help="Number of leading components to analyze. Default: 9",
    )
    spectra.add_argument(
        "--reference-component",
        type=int,
        default=0,
        help="Component used as the empirical line profile. Default: 0",
    )
    spectra.add_argument(
        "--smoothing-window",
        type=int,
        default=11,
        help="Odd Savitzky-Golay window in channels. Default: 11",
    )
    spectra.add_argument(
        "--smoothing-order",
        type=int,
        default=4,
        help="Savitzky-Golay polynomial order, at least 4. Default: 4",
    )
    spectra.add_argument(
        "--center-velocity",
        type=float,
        default=None,
        help=(
            "Reflection and dilation center in km/s. Default: zero when "
            "available, otherwise the reference-profile peak"
        ),
    )
    spectra.add_argument(
        "-o",
        "--output",
        help=(
            "Output ECSV table. Default: "
            "pca_spectral_characterization_<input>.ecsv"
        ),
    )
    spectra.add_argument(
        "--plot-output",
        help=(
            "Output diagnostic figure. Default: "
            "pca_spectral_templates_<input>.png"
        ),
    )
    spectra.add_argument("--dpi", type=int, default=200)
    spectra.add_argument("--show", action="store_true")
    spectra.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Fail instead of replacing existing outputs",
    )
    spectra.set_defaults(overwrite=True)

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

    if command in {"plot-acf", "plot-diagnostics"}:
        result = read_pca_artifact(args.artifact)
        legacy_name = command == "plot-diagnostics"
        spatial_product = "spatialwidths" if legacy_name else "spatialacf"
        spectral_product = (
            "spectralwidths" if legacy_name else "spectralacf"
        )
        spatial_output = (
            Path(args.spatial_output)
            if args.spatial_output
            else _default_output(
                args.artifact,
                spatial_product,
                ".png",
            )
        )
        spectral_output = (
            Path(args.spectral_output)
            if args.spectral_output
            else _default_output(
                args.artifact,
                spectral_product,
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
        print(f"Wrote spatial ACF plot to {spatial_output}")
        print(f"Wrote spectral ACF plot to {spectral_output}")
        return 0

    if command == "characterize":
        from astropy.table import vstack

        from .characterization import (
            characterize_result,
            core_characterization_paths,
            excursion_characterization_path,
            load_disc_ring_geometry,
            m0_residual_characterization_path,
            mpeak_residual_characterization_path,
            plot_core_characterization,
            plot_excursion_sets,
            plot_m0_residuals,
            plot_mpeak_residuals,
            write_characterization_table,
        )

        artifacts = [Path(value) for value in args.artifacts]
        if args.labels is None:
            labels = [str(value) for value in artifacts]
        elif len(args.labels) != len(artifacts):
            raise ValueError("--labels must contain one label per artifact")
        else:
            labels = args.labels

        explicit_components = _component_values(args.components)
        parfiles = _resolve_characterization_parfiles(
            artifacts,
            explicit_parfiles=args.parfile,
            circular_sky=args.circular_deproj,
        )
        tables = []
        results = []
        component_lists = []
        ring_geometries = []
        for artifact, label, parfile in zip(artifacts, labels, parfiles):
            result = read_pca_artifact(artifact)
            if parfile is None:
                ring_geometry = None
                reason = (
                    "requested by --circular-deproj"
                    if args.circular_deproj
                    else "no DiscMiner parfile was found"
                )
                print(
                    f"Using circular sky-plane rings for {artifact}; "
                    f"{reason}"
                )
            else:
                ring_geometry = load_disc_ring_geometry(
                    parfile,
                    result.source_header,
                    surface=args.deprojection_surface,
                )
                print(
                    f"Using deprojected {args.deprojection_surface}-surface "
                    f"rings from {parfile} for {artifact}"
                )
            if explicit_components is None:
                if args.n_components < 1:
                    raise ValueError("--n-components must be positive")
                components = range(
                    min(args.n_components, result.n_components)
                )
            else:
                components = explicit_components
            components = list(components)
            results.append(result)
            component_lists.append(components)
            ring_geometries.append(ring_geometry)
            tables.append(
                characterize_result(
                    result,
                    label=label,
                    artifact=artifact,
                    components=components,
                    acf_level=args.acf_level,
                    n_azimuth=args.azimuth_samples,
                    minimum_azimuthal_coverage=(
                        args.minimum_azimuthal_coverage
                    ),
                    phase_minimum_relative_power=(
                        args.phase_minimum_relative_power
                    ),
                    excursion_percentile=args.excursion_percentile,
                    maximum_angular_mode=args.maximum_angular_mode,
                    ring_geometry=ring_geometry,
                )
            )

        table = vstack(tables, metadata_conflicts="silent")
        if args.output:
            output = Path(args.output)
        elif len(artifacts) == 1:
            output = _default_output(
                artifacts[0], "characterization", ".ecsv"
            )
        else:
            output = Path("pca_characterization.ecsv")
        plot_prefix = (
            Path(args.plot_prefix)
            if args.plot_prefix
            else output.with_suffix("")
        )
        group_keys = {
            "variance": ("component_importance",),
            "acf": ("acf_morphology",),
            "eigenimage": ("eigenimage_structure",),
            "angularmode": ("angularmode",),
            "all": (
                "component_importance",
                "acf_morphology",
                "eigenimage_structure",
                "angularmode",
            ),
        }
        available_outputs = core_characterization_paths(plot_prefix)
        plot_outputs = [
            available_outputs[key]
            for key in group_keys[args.plot_group]
        ]
        excursion_output = excursion_characterization_path(plot_prefix)
        if args.plot_excursions:
            plot_outputs.append(excursion_output)
        m0_residual_output = m0_residual_characterization_path(plot_prefix)
        if args.plot_m0_residuals:
            plot_outputs.append(m0_residual_output)
        mpeak_residual_output = mpeak_residual_characterization_path(
            plot_prefix
        )
        if args.plot_mpeak_residuals:
            plot_outputs.append(mpeak_residual_output)

        if not args.overwrite:
            existing = [
                path for path in [output] + plot_outputs if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "Output already exists: "
                    + ", ".join(str(path) for path in existing)
                )

        write_characterization_table(
            table,
            output,
            overwrite=args.overwrite,
        )
        print(f"Wrote PCA characterization table to {output}")
        groups = (
            ("variance", "acf", "eigenimage", "angularmode")
            if args.plot_group == "all"
            else (args.plot_group,)
        )
        outputs = plot_core_characterization(
            table,
            plot_prefix,
            groups=groups,
            include_pc0_cumulative=args.include_pc0_cumulative,
            angular_normalization=args.angular_normalization,
            dpi=args.dpi,
            show=args.show,
        )
        for plot_output in outputs.values():
            print(f"Wrote PCA characterization plot to {plot_output}")
        if args.plot_excursions:
            plot_excursion_sets(
                results,
                labels,
                component_lists,
                excursion_output,
                percentile=args.excursion_percentile,
                dpi=args.dpi,
                show=args.show,
            )
            print(
                "Wrote PCA excursion-set plot to "
                f"{excursion_output}"
            )
        if args.plot_m0_residuals:
            plot_m0_residuals(
                results,
                labels,
                component_lists,
                ring_geometries,
                m0_residual_output,
                n_azimuth=args.azimuth_samples,
                minimum_azimuthal_coverage=(
                    args.minimum_azimuthal_coverage
                ),
                maximum_angular_mode=args.maximum_angular_mode,
                robust=args.mode_residual_robust,
                percentile=args.mode_residual_percentile,
                dpi=args.dpi,
                show=args.show,
            )
            print(
                "Wrote PCA m=0-subtracted eigenimage plot to "
                f"{m0_residual_output}"
            )
        if args.plot_mpeak_residuals:
            plot_mpeak_residuals(
                results,
                labels,
                component_lists,
                ring_geometries,
                mpeak_residual_output,
                n_azimuth=args.azimuth_samples,
                minimum_azimuthal_coverage=(
                    args.minimum_azimuthal_coverage
                ),
                maximum_angular_mode=args.maximum_angular_mode,
                robust=args.mode_residual_robust,
                percentile=args.mode_residual_percentile,
                dpi=args.dpi,
                show=args.show,
            )
            print(
                "Wrote PCA dominant-mode-subtracted eigenimage plot to "
                f"{mpeak_residual_output}"
            )
        return 0

    if command == "characterize-spectra":
        from .spectral_characterization import (
            characterize_spectra,
            plot_spectral_characterization,
            write_spectral_characterization,
        )

        artifact = Path(args.artifact)
        result = read_pca_artifact(artifact)
        explicit_components = _component_values(args.components)
        if explicit_components is None:
            if args.n_components < 1:
                raise ValueError("--n-components must be positive")
            components = range(
                min(args.n_components, result.n_components)
            )
        else:
            components = explicit_components
        output = (
            Path(args.output)
            if args.output
            else _default_output(
                artifact,
                "spectral_characterization",
                ".ecsv",
            )
        )
        plot_output = (
            Path(args.plot_output)
            if args.plot_output
            else _default_output(
                artifact,
                "spectral_templates",
                ".png",
            )
        )
        if not args.overwrite:
            existing = [
                path for path in (output, plot_output) if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "Output already exists: "
                    + ", ".join(str(path) for path in existing)
                )

        table, templates = characterize_spectra(
            result,
            components=components,
            artifact=artifact,
            reference_component=args.reference_component,
            smoothing_window=args.smoothing_window,
            smoothing_order=args.smoothing_order,
            center_velocity=args.center_velocity,
        )
        write_spectral_characterization(
            table,
            output,
            overwrite=args.overwrite,
        )
        plot_spectral_characterization(
            result,
            table,
            templates,
            plot_output,
            dpi=args.dpi,
            show=args.show,
        )
        print(
            table[
                "component",
                "variance_percent",
                "parity_correlation",
                "dominant_template",
                "dominant_overlap",
            ]
        )
        print(f"Wrote spectral characterization table to {output}")
        print(f"Wrote spectral template plot to {plot_output}")
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
