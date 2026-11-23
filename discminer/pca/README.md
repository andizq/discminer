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
discminer pca plot-acf \
    pca_cube_data_TAG_convtb_stackedcube.fits
discminer pca characterize \
    pca_cube_data_TAG_convtb_stackedcube.fits
discminer pca characterize-spectra \
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

## Component and ACF diagnostics

### Standalone variance and tail diagnostics

`plot-variance` reads saved PCA artifacts directly and produces one figure with
variance and cumulative variance expressed as percentages of the full spectrum.
`--show-renormalized` adds the optional "% beyond PC 0" panels.
`--show-power` adds unnormalized per-component and cumulative-power panels:
the stored eigenvalues in squared source intensity units (for example, K²).
Use `--power-only` to show absolute power **instead of** the percentage panels.
It cannot be combined with `--show-power` or `--show-renormalized`.
It does not compute eigenimage morphology, read a parfile, or rerun PCA.

```bash
discminer pca plot-variance pca_cube_data_TAG_stackedcube.fits
discminer pca plot-variance pca_cube_data_TAG_stackedcube.fits \
    --tail-fit log-linear --tail-start 10 --tail-starts 20,30 \
    --n-components 12 --output variance_tail.png
discminer pca plot-variance pca_20ms.fits pca_100ms.fits \
    --labels "20 m/s" "100 m/s" --show-power --include-pc0-cumulative \
    --n-components 20 --output variance_and_power.png
discminer pca plot-variance pca_20ms.fits pca_100ms.fits \
    --labels "20 m/s" "100 m/s" --power-only --include-pc0-cumulative \
    --tail-fit log-linear --tail-start 50 -n 20 --output absolute_power.png
```

The default display shows nine components; `--n-components -1` shows all.
Normalization and fitting always use the complete stored eigenvalue spectrum,
regardless of the display limit. The full-spectrum cumulative panel excludes PC 0
by default while retaining the full-spectrum denominator; use
`--include-pc0-cumulative` to include it. The optional renormalized panels always
exclude PC 0 and divide by the sum of all remaining eigenvalues. With `MEANSUB=False`,
the artifact's reported variance includes uncentered power; this is recorded in
the output table metadata. Figures have no footer text. A short in-panel legend
identifies the primary baseline range and MAD envelope.

The raw cumulative-power panel also follows `--include-pc0-cumulative`;
include PC 0 and display all components (`-n -1`) to see total stored power.
Tail baselines and envelopes are shown in native units in the raw spectrum.
Power comparisons require the same source intensity units; no conversion is
performed. Use matched spatial/spectral sampling and preprocessing as well.
Raw power depends on brightness and line shape, and with `MEANSUB=False`
includes the common profile. It is not a direct measure of turbulent kinetic
energy. The ECSV diagnostics already contain the full raw `eigenvalue` column.

Tail fitting is optional (`--tail-fit none`, the default). Two models are available:

- `constant`: the median tail eigenvalue, bounded by plus/minus
  `--tail-mad` times its scaled median absolute deviation (MAD).
- `log-linear`: a robust Theil-Sen fit of `ln(eigenvalue)` against component
  index, with multiplicative bounds from the scaled MAD of the log residuals.

`--tail-start` selects the primary fit **shown as the baseline on the figure**
(default: PC 10). `--tail-starts` adds starts for sensitivity checks; those fits
are not averaged into the displayed baseline. Each fit runs through the last stored
component unless `--tail-stop` sets an **inclusive** end. Every requested range
must contain at least three components in every supplied artifact; log-linear
fits require positive eigenvalues throughout the range. `--tail-mad` defaults to
3, using the conventional MAD scale factor 1.4826022185.

When fitting a tail, the figure adds observed/primary-upper-envelope ratios and
the fraction of eligible tail choices exceeded by each component. Filled ratio
points precede the primary fitted tail; open points belong to or follow that
range and are not counted as independently tested excesses. A component is
eligible for a fit's flag only when it precedes that fit's start. Counts such as
`3/4` describe sensitivity to the chosen ranges, **not detection probability**.
The table's `above_all_eligible_tails` flag additionally requires at least three
eligible fits. Empirical MAD bounds are not confidence bands or sigma detection
levels, and a sloping tail is not independently measured background noise.

Use `--tail-diagnostics` with a tail fit to write a **separate optional figure**:

```bash
discminer pca plot-variance pca_cube_data_TAG_stackedcube.fits \
    --tail-fit log-linear --tail-starts 20,30 --tail-diagnostics \
    --output variance.png
```

This also writes `variance_tail_diagnostics.png` (or the same image format as
`--output`). It shows the full stored eigenvalue spectrum with the primary
baseline, MAD envelope and shaded fitted interval, alongside
`ln(observed / baseline)` **inside that primary interval**. Early extrapolated
PCs are excluded from the residual panel so they do not obscure tail curvature.
Zero eigenvalues, whose logarithms are undefined, are omitted from that panel.
`--n-components` affects only the main figure.

The full-spectrum panel also overlays a descriptive log-linear fit to the
**observed eigenvalues** of PCs before the primary tail that exceed its upper
MAD envelope. PC 0 is excluded. This uses the same Theil-Sen slope and
median-centered log intercept as the tail fit. The line is drawn only across
the selected ranks, with a text box reporting `ln(lambda) = a + b*PC` and
sample count `N`. Selection can contain gaps; the exact PC
list and full-precision coefficients are saved in `<figure stem>_leading.ecsv`.
The tail baseline is not subtracted from these eigenvalues. This is a
selection-dependent descriptive slope, not a calibrated signal slope or a
mechanism classification. Two selected PCs provide a two-point line
without a scatter estimate; fewer than two produce an explanatory
annotation instead of a fit. No formal coefficient uncertainties are assigned.

With several tail starts, the optional figure adds fitted slope and scaled
log-residual MAD versus tail start; constant fits instead show the median
baseline and linear MAD. A ring identifies the primary fit. Additional fits
remain sensitivity checks and do not alter the displayed primary baseline.
The diagnostic reuses the calculated tail fits and does not recompute PCA.
It supports multiple artifacts, uses the main figure's colors
and markers, and has no footer text. `--no-overwrite` protects this figure too.

Multiple artifacts can be compared without computing characterization tables:

```bash
discminer pca plot-variance pca_gi.fits pca_vsi.fits pca_mri.fits \
    --labels GI VSI MRI --tail-fit log-linear --tail-starts 10,25,45 \
    --output instability_variance.png
```

The default output is `pca_variance_<first input>.png`. A component table with
the same stem and `.ecsv` extension is always written, containing all stored
components, normalization, primary bounds, and eligible/excess counts. When
fitting, `<figure stem>_tails.ecsv` records every fit's range, coefficients, MAD
scale, and flagged components. The tables record artifact paths and settings.
`--no-overwrite` protects all requested products. Component ranks in different
PCA decompositions need not represent identical physical modes.

The `plot-components` command pairs every selected eigenimage with its
eigenvector on the artifact's physical velocity axis. PCA component signs are
arbitrary, so the relative lobes of an eigenvector are meaningful but its
overall sign is not.

The `plot-acf` command plots up to the first nine components and writes
`pca_spatialacf_<cube>.png` and `pca_spectralacf_<cube>.png`. Use
`--n-components` to show fewer components or `--max-lag` to zoom the spectral
autocorrelation axes. The former `plot-diagnostics` spelling remains available
as a compatibility alias and retains its original default output names.

### Model-independent eigenspectrum templates

The `characterize-spectra` command uses PC 0 as an empirical reference line
profile by default. It smooths that profile and constructs orthonormal
intensity, centroid, linewidth, third-derivative, and fourth-derivative
templates. It requires only the velocity axis and eigenvectors stored in the
PCA artifact; it does not read a DiscMiner parfile or any fitted moment maps.

```bash
discminer pca characterize-spectra pca_cube_data_TAG_stackedcube.fits \
    --n-components 9 --smoothing-window 11
```

The command writes `pca_spectral_characterization_<cube>.ecsv` and
`pca_spectral_templates_<cube>.png`. The figure shows the empirical templates,
the selected eigenspectra aligned to their strongest template, and the full
overlap matrix.

Parity is the normalized correlation between an eigenvector and its reflection
about the selected centre velocity,

```text
P = <e(v), e(2 v_center - v)> / (||e(v)|| ||e(2 v_center - v)||).
```

Thus `P=+1` is perfectly even, `P=-1` is perfectly odd, and values near zero
are mixed. Template overlap is the absolute cosine similarity

```text
O_kp = |<e_k, b_p>| / (||e_k|| ||b_p||),
```

where `e_k` is an eigenspectrum and `b_p` is an empirical response template.
It ranges from zero to one and uses an absolute value because PCA component
signs are arbitrary. The templates are orthogonalized in the order intensity,
centroid, linewidth, third derivative, and fourth derivative. Consequently,
the linewidth template excludes simple intensity scaling, and the higher-order
templates exclude the lower-order responses. The default uniform channel
weighting matches the Euclidean metric of the PCA decomposition.

Use `--center-velocity`, `--reference-component`, `--smoothing-window`, and
`--smoothing-order` to test the sensitivity of the result. Higher derivatives
are especially sensitive to smoothing and should be interpreted as line-shape
diagnostics rather than unique physical labels.

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

Pass `--plot-excursions` to additionally write
`_eigenimage_excursions.png`. This optional grid shows the analyzed
eigenimages with solid red positive contours, dashed blue negative contours,
and a thin black boundary around the combined absolute excursion set. Rows
correspond to input artifacts and columns to the selected PCA components. The
contours use the same threshold set by `--excursion-percentile` as the
compactness calculation and are not included by `--plot-group all`.

Pass `--plot-m0-residuals` to additionally write
`_eigenimage_m0_residuals.png`. For every selected PC, this optional figure
shows the original eigenimage, the projected azimuthal-ring mean (`m=0`), and
their residual (`m>=1`). It uses the same mask, deprojected ring geometry,
emission surface, beam-aware inner radius, azimuth sampling, and minimum ring
coverage as the tabulated angular diagnostics. Pixels outside the outermost
accepted annulus are masked.

Pass `--plot-mpeak-residuals` to write the analogous
`_eigenimage_mpeak_residuals.png` figure. Its first row is the `m=0`-subtracted
eigenimage, the middle row is the mode from `m=1` through
`--maximum-angular-mode` with the largest area-weighted power integrated over
the accepted rings, and the final row removes that mode from the
non-axisymmetric eigenimage. The selected mode is annotated in each middle-row
panel and can differ between PCs.

In both residual figures the original and removed-mode panels share a
symmetric color scale, while each residual receives its own symmetric scale
so that weak remaining structure stays visible. Scaling is robust by default
at the 99.5th percentile of absolute amplitude, matching
`plot-components --robust`. Use `--percentile` to change that percentile or
`--no-robust` to use the full finite amplitude range.

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

The optional mode-residual plots decompose eigenimages after PCA. They do not
rerun the decomposition: eigenvectors, eigenvalues, explained variances, and
component numbering therefore remain unchanged. To measure a PCA basis of a
mode-filtered cube instead, subtract the selected field channel by channel
before running PCA.

Phase coherence measures how closely those phases follow one logarithmic
winding. The table also records the fitted phase slope against `log(radius)`.
Compactness is `4*pi*area/perimeter**2` for the strongest absolute eigenimage
excursions. It uses `abs(eigenimage)`, so it is invariant to the arbitrary sign
of a PCA component. The red and blue signs in the optional excursion plot may
therefore swap between PCA analyses; their relative spatial arrangement, the
combined boundary, and compactness remain meaningful.

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

For comparison with the complete angular spectrum, the table additionally
stores `eigenimage_m_dominant_all`, selected from `m=0` through `m_max`, and
`eigenimage_f_dominant_all_total`, its area-weighted power relative to the
complete eigenimage. These all-mode columns do not replace
`eigenimage_m_peak`: the latter intentionally remains restricted to `m>=1` so
that its phase coherence and orientation slope remain well defined.

Phase coherence and physical orientation slope are conditional on the
dominant angular mode being appreciable and distinct. In the angular-mode
figure, a normal colored marker requires
`eigenimage_f_peak_total >= 0.10` and `eigenimage_mode_entropy <= 0.85`.
The orientation slope additionally requires phase coherence of at least 0.70
and at least 10 fitted rings. Markers that fail these provisional quality
checks use a translucent gray face with the dataset-colored edge. Their
measurements remain in the plot and ECSV table, but should be treated with
care. The quality check always uses the total-power peak fraction, regardless
of the displayed angular normalization.

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
