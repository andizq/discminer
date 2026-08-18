# DiscMiner PCA workflow

DiscMiner can decompose stacked spectral-line cubes into principal-component
eigenimages and eigenvectors, measure their spatial and spectral structure,
and reconstruct cubes from selected components.

## Installation

The PCA workflow uses
[TurbuStat](https://turbustat.readthedocs.io/). Install DiscMiner with the PCA
optional dependencies using:

```bash
pip install "discminer[pca]"
```

## Running PCA

After removing the disc rotation component with `stackcube`, run PCA and save
the complete result:

```bash
discminer stackcube
discminer pca run cube_data_TAG_convtb_stackedcube.fits
```

The `run` command writes `pca_<cube>.fits` and a covariance plot whose axes and
central zoom are derived from the input velocity axis. The artifact stores the
decomposition, widths, covariance matrix, source velocity axis, validity mask,
reconstruction metadata, and spatial and spectral autocorrelations. All plots
can therefore be regenerated without repeating the decomposition.

## Inspecting and reconstructing components

```bash
discminer pca plot-widths pca_cube_data_TAG_convtb_stackedcube.fits
discminer pca plot-diagnostics \
    pca_cube_data_TAG_convtb_stackedcube.fits
discminer pca characterize \
    pca_cube_data_TAG_convtb_stackedcube.fits
discminer pca plot-components pca_cube_data_TAG_convtb_stackedcube.fits \
    --components 0,1,2,3,4,5
discminer pca reconstruct pca_cube_data_TAG_convtb_stackedcube.fits \
    --exclude 2 --output reconstructed_without_pc2.fits
discminer pca plot-channels reconstructed_without_pc2.fits
```

PCA component indices are zero-based. Use `discminer pca <command> -h` for the
complete set of options, including explicit distance, covariance velocity
limits, beam correction, and component selection.

## Width diagnostics

The diagnostics command plots up to the first nine components and writes
`pca_spatialwidths_<cube>.png` and `pca_spectralwidths_<cube>.png`. Use
`--n-components` to show fewer components or `--max-lag` to zoom the spectral
autocorrelation axes.

The width plot reproduces the original custom weighted fit in log space. By
default it considers the first six components resolved above one major-axis
beam FWHM and applies the original `0.2` scaling to the TurbuStat spectral
width errors. These choices can be adjusted with:

```bash
discminer pca plot-widths pca_cube_data_TAG_convtb_stackedcube.fits \
    --n-fit-components 6 --beam-multiple 1 \
    --spectral-error-scale 0.2
```

## Morphological characterization

The `characterize` command writes an ECSV table and, by default, four retained
diagnostic figures with one shared prefix. These are intrinsic measurements of
each PCA artifact; a smooth reference model is not required. Multiple
artifacts can be supplied with matching `--labels` for comparison:

```bash
discminer pca characterize pca_cube_data_TAG_convtb_stackedcube.fits \
    --plot-group all --plot-prefix pca_characterization_core
```

This writes:

- `_component_importance.png`: a two-by-two comparison of the absolute and
  PC 0-excluded renormalized variance spectra and cumulative variances.
- `_acf_morphology.png`: ACF axis ratio, ellipse residual, maximum `Q2`, and
  the radius of maximum `Q2`.
- `_eigenimage_structure.png`: `m=0` fraction, total or non-axisymmetric
  `m=2` fraction, phase coherence, winding slope, and compactness.
- `_angularmode.png`: dominant mode, total or non-axisymmetric peak-mode
  power, mode entropy, low-order modeled fraction, phase coherence, and
  physical orientation slope.

`Q4` and Euler measurements remain available in the ECSV table but are omitted
from the core figures. Use `--plot-group variance`, `--plot-group acf`,
`--plot-group eigenimage`, or `--plot-group angularmode` to write only one
figure.

By default PC 0 is omitted from the cumulative-variance curve without
renormalizing the remaining variance. Pass `--include-pc0-cumulative` to
restore the conventional cumulative curve. The right-hand panels divide the
PC 1-and-above variance fractions by their total and show both their individual
and cumulative contributions. This compares the distribution of subleading
variance independently of the fraction carried by PC 0, while the left-hand
panels preserve its absolute importance relative to the full cube.

### Eigenimage diagnostics

The `m=0` fraction measures the area-weighted power in the azimuthal mean of
each radial ring relative to the total eigenimage power. The `m=2` fraction
is stored both relative to the total eigenimage power and relative to the
remaining non-axisymmetric power while allowing its phase to vary between
radial rings.

Phase coherence measures how closely those phases follow one logarithmic
winding. The table also records the fitted phase slope against `log(radius)`.
Compactness is `4*pi*area/perimeter**2` for the strongest absolute eigenimage
excursions. It uses `abs(eigenimage)`, so it is invariant to the arbitrary sign
of a PCA component.

The angular-mode spectrum fits modes simultaneously from `m=1` through
`--maximum-angular-mode`, which defaults to 6. All modes use the same
beam-aware radial domain. The dominant-mode orientation slope divides the
coefficient phase slope by the mode number, making winding rates comparable
across modes.

Both total and non-axisymmetric normalizations are always stored in the ECSV
table. `--angular-normalization total`, the default, displays mode power
relative to the complete eigenimage. Use `--angular-normalization
nonaxisymmetric` to display mode power relative to the power remaining after
the radial-ring mean is removed. This option changes the displayed `m=2`,
peak-mode, and low-order modeled fractions; it does not allow `m=0` to become
the dominant mode and does not change phase coherence or orientation slopes.

The angular-mode table also stores `eigenimage_f_peak_fitted`, the share of
the modeled low-order power carried by the dominant mode. This differs from
the modeled fraction: the modeled fraction measures the combined power in all
modes from `m=1` through `m_max`, whereas the peak fraction measures only the
strongest of those modes. The total-power bookkeeping columns separate the
axisymmetric, modeled low-order, and unresolved or higher-order contributions.

### Disc-plane ring geometry

Eigenimage ring diagnostics use deprojected disc-plane annuli whenever a
DiscMiner `parfile.json` is available. The command first searches beside each
PCA artifact and then in the working directory. Use `--parfile` with one shared
path or one path per artifact to select files explicitly, and use
`--surface upper|lower|midplane` to choose the emitting surface.

The annuli are projected onto the image with the same inclination, position
angle, center offset, and standard emission-surface functions used by
`discminer.rail`. Pass `--circular-deproj` or its clearer alias
`--circular-sky` to restore the legacy circular rings in the sky plane.

Both geometries use radial samples spaced by one image pixel, beginning at two
pixels, and 360 uniformly spaced azimuths by default. Only complete projected
annuli are defined, and a ring must retain the fraction selected by
`--minimum-azimuthal-coverage` after applying the data mask. The ECSV table
records the geometry and sampling provenance.

ACF diagnostics remain measured in the spatial-lag plane, and compactness
remains a direct image-plane excursion-set statistic.
