"""Interactive channel maps shared by Cube.show and Cube.show_side_by_side."""

import copy
import csv
import json
import re
from collections import Counter
from functools import partial
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backend_bases import key_press_handler
from matplotlib.colors import Normalize
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage
from matplotlib.ticker import AutoMinorLocator
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.widgets import Button, Cursor, RectangleSelector, Slider, TextBox
from astropy import units as u
from astropy.wcs import utils as wcsutils
from packaging.version import Version
import matplotlib
from .projected_maps import validate_velocity_maps


_VIEWER_STYLE = {
    'font.family': 'monospace', 'font.size': 9.,
    'axes.labelsize': 10., 'axes.labelpad': 8.,
    'axes.titlesize': 10.5, 'axes.titlepad': 9., 'axes.linewidth': 1.3,
    'xtick.labelsize': 9., 'ytick.labelsize': 9.,
    'legend.fontsize': 8.5, 'figure.titlesize': 11.,
}


class _ViewerTextBox(TextBox):
    def _resize(self, event):
        # Matplotlib 3.11.1 decorates TextBox._resize as a mouse callback,
        # but ResizeEvent has no inaxes/x/y. Only stop_typing is needed here.
        self.stop_typing()


def _viewer_cmap(name):
    cmap = plt.get_cmap(name)
    if hasattr(cmap, 'with_extremes'):
        return cmap.with_extremes(bad='0.85')
    cmap = copy.copy(cmap)
    cmap.set_bad('0.85')
    return cmap


def _compact_cube_labels(filenames, max_length=28):
    """Keep distinguishing filename tokens, with unique, readable fallbacks."""
    paths = [Path(filename) for filename in filenames]
    stems = []
    for path in paths:
        stem = path.name
        for suffix in ('.fits.gz', '.fit.gz', '.fts.gz', '.fits', '.fit', '.fts'):
            if stem.lower().endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        stems.append(stem or 'Cube')
    candidates = list(stems)
    if len(stems) > 1:
        # Preserve minus signs in numeric tags (e.g. inclination_-15).
        tokens = [re.split(r'[_\s]+|-(?!\d)', stem) for stem in stems]
        shortest = min(map(len, tokens))
        prefix = 0
        while prefix < shortest and len({parts[prefix] for parts in tokens}) == 1:
            prefix += 1
        suffix = 0
        while suffix < shortest - prefix and len({parts[-suffix - 1] for parts in tokens}) == 1:
            suffix += 1
        middle = [parts[prefix:len(parts) - suffix if suffix else None] for parts in tokens]
        if all(middle):
            candidates = []
            for parts, distinct in zip(tokens, middle):
                label = '_'.join(distinct)
                if len(label) < 3 or not any(character.isalpha() for character in label):
                    # Retain context for names differing only in a short ID.
                    distinct = parts[max(0, prefix - 1):len(parts) - suffix if suffix else None]
                    label = '_'.join(distinct)
                candidates.append(label)
        if len(set(candidates)) != len(candidates):
            parents = [path.parent.name for path in paths]
            candidates = (parents if all(parents) and len(set(parents)) == len(paths)
                          else ['Cube %d' % index for index in range(len(paths))])

    def shorten(label, length=max_length):
        if len(label) <= length:
            return label
        left = (length - 1) // 2
        return label[:left] + '…' + label[-(length - left - 1):]

    labels = [shorten(label) for label in candidates]
    counts = Counter(labels)
    for index, label in enumerate(labels):
        if counts[label] > 1:
            tag = ' [%d]' % index
            labels[index] = shorten(candidates[index], max_length - len(tag)) + tag
    if len(set(labels)) != len(labels):
        labels = ['Cube %d' % index for index in range(len(paths))]
    return labels


def pixel_at(x, y, shape, extent=None):
    """Map displayed coordinates to (row, column), or None outside the image."""
    if x is None or y is None or not np.all(np.isfinite([x, y])):
        return None
    ny, nx = shape
    if extent is None:
        extent = (-0.5, nx - 0.5, -0.5, ny - 0.5)
    left, right, bottom, top = extent
    fx, fy = (x - left) / (right - left), (y - bottom) / (top - bottom)
    if not (0 <= fx <= 1 and 0 <= fy <= 1):
        return None
    return min(int(fy * ny), ny - 1), min(int(fx * nx), nx - 1)


def box_bounds(x0, x1, y0, y1, shape, extent=None):
    """Return clipped, nonempty half-open pixel bounds in either drag direction."""
    if any(value is None for value in (x0, x1, y0, y1)):
        return None
    if not np.all(np.isfinite([x0, x1, y0, y1])) or x0 == x1 or y0 == y1:
        return None
    ny, nx = shape
    if extent is None:
        extent = (-0.5, nx - 0.5, -0.5, ny - 0.5)
    left, right, bottom, top = extent
    xx = sorted(((x0 - left) * nx / (right - left),
                 (x1 - left) * nx / (right - left)))
    yy = sorted(((y0 - bottom) * ny / (top - bottom),
                 (y1 - bottom) * ny / (top - bottom)))
    j0, j1 = np.clip([np.floor(xx[0]), np.ceil(xx[1])], 0, nx).astype(int)
    i0, i1 = np.clip([np.floor(yy[0]), np.ceil(yy[1])], 0, ny).astype(int)
    return (i0, i1, j0, j1) if i0 < i1 and j0 < j1 else None


def region_spectrum(data, bounds, stat_func=np.nanmean):
    """Reduce a region without passing nonfinite values to the statistic."""
    i0, i1, j0, j1 = bounds
    region = data[:, i0:i1, j0:j1]
    counts = np.isfinite(region).sum(axis=(1, 2))
    spectrum = np.full(len(region), np.nan)
    for index in np.flatnonzero(counts):
        plane = region[index]
        spectrum[index] = stat_func(plane[np.isfinite(plane)])
    return spectrum, counts


def validate_comparison(cubes):
    """Reject incompatible coordinates/units; warn about unequal resolution."""
    reference = cubes[0]
    velocities = np.asarray(reference.vchannels)
    if (velocities.shape != (reference.nchan,) or not np.isfinite(velocities).all()
            or (len(velocities) > 1 and not (
                np.all(np.diff(velocities) > 0) or np.all(np.diff(velocities) < 0)))):
        raise ValueError('Velocity channels must be finite and strictly monotonic.')
    for cube in cubes[1:]:
        if cube.data.shape != reference.data.shape:
            raise ValueError('Comparison cubes must have the same shape.')
        if not np.allclose(cube.vchannels, velocities, rtol=1e-9, atol=1e-10):
            raise ValueError('Comparison cubes must have matching velocity channels.')
        for key in ('SPECSYS', 'SSYSOBS'):
            first, second = reference.header.get(key), cube.header.get(key)
            if first and second and first.strip().upper() != second.strip().upper():
                raise ValueError('Comparison cubes have different %s reference frames.' % key)
        units = [item.header.get('BUNIT', '').strip() for item in (reference, cube)]
        if all(units):
            try:
                scale = u.Unit(units[1]).to(u.Unit(units[0]))
                matched = np.isclose(scale, 1., rtol=1e-12, atol=0.)
            except ValueError:
                matched = units[0] == units[1]
            if not matched:
                raise ValueError('Comparison intensity units differ; convert cubes to the same BUNIT first.')
        else:
            warnings.warn('Missing BUNIT: comparison intensity units cannot be verified.', UserWarning)
        if not np.isclose(cube.dpc.to_value(u.pc), reference.dpc.to_value(u.pc), rtol=1e-8):
            raise ValueError('Comparison cubes must use the same distance for offsets in au.')
        if reference.wcs.has_celestial and cube.wcs.has_celestial:
            ny, nx = reference.data.shape[1:]
            xx, yy = np.meshgrid(np.linspace(0, nx - 1, 3), np.linspace(0, ny - 1, 3))
            first = wcsutils.pixel_to_skycoord(xx, yy, reference.wcs.celestial)
            second = wcsutils.pixel_to_skycoord(xx, yy, cube.wcs.celestial)
            pixel_deg = np.min(wcsutils.proj_plane_pixel_scales(reference.wcs.celestial))
            if not np.all(first.separation(second).to_value(u.deg) < pixel_deg * 1e-4):
                raise ValueError('Comparison spatial WCS differs; reproject onto the same pixel grid first.')
        else:
            warnings.warn('Missing celestial WCS: comparison spatial registration cannot be verified.', UserWarning)
        beams = [item.beam for item in (reference, cube)]
        if any(beam is None for beam in beams):
            different = (beams[0] is None) != (beams[1] is None)
        else:
            sizes = [np.array([b.major.to_value(u.arcsec), b.minor.to_value(u.arcsec)]) for b in beams]
            different = not np.allclose(*sizes, rtol=1e-5, atol=0.)
            circular = all(np.isclose(size[0], size[1], rtol=1e-5) for size in sizes)
            angle = (beams[0].pa - beams[1].pa).to_value(u.deg)
            different |= not circular and abs((angle + 90) % 180 - 90) > 1e-3
        if different:
            warnings.warn('Comparison beams differ; spectra and maps have different spatial resolution.', UserWarning)


class ChannelViewer:
    """A retained viewer with navigation, selection, contrast and CSV export.

    Channel indices are zero-based. Velocities use Cube.vchannels (km/s); changing
    vel_unit only changes its displayed label. ``export()`` writes CSV spectra and
    a JSON sidecar containing pixel bounds, FITS headers and viewing parameters.
    """

    def __init__(self, cube, compare_cubes=None, side_by_side=False, extent=None,
                 chan_init=None, cube_init=0, cursor_grid=True, cmap='discminer.intensity',
                 int_unit=None, pos_unit='Offset [au]', vel_unit=r'km s$^{-1}$',
                 show_beam=True, surface_from=None, kwargs_surface=None,
                 vmin=None, vmax=None, channel_step=1, vel_fmt='.2f',
                 cube_labels=None, export_path=None, stat_func=np.nanmean,
                 velocity2d=None, kwargs_isov=None, **kwargs):
        self.cubes = [cube] + list(compare_cubes or [])
        validate_comparison(self.cubes)
        self.cube = cube
        self.side_by_side = side_by_side
        if side_by_side and len(self.cubes) != 2:
            raise ValueError('Side-by-side viewing requires exactly two cubes.')
        if chan_init is None:
            chan_init = cube.nchan // 2
        for value, limit, name in ((chan_init, cube.nchan, 'chan_init'),
                                   (cube_init, len(self.cubes), 'cube_init')):
            if int(value) != value or not 0 <= value < limit:
                raise ValueError('%s must be an in-range integer.' % name)
        if int(channel_step) != channel_step or channel_step < 1:
            raise ValueError('channel_step must be a positive integer.')
        format(float(cube.vchannels[0]), vel_fmt)
        self.channel_step, self.vel_fmt = int(channel_step), vel_fmt
        self.channel, self.cube_index = int(chan_init), int(cube_init)
        self.extent = list(extent) if extent is not None else [-cube.xsky, cube.xsky, -cube.ysky, cube.ysky]
        if (len(self.extent) != 4 or not np.isfinite(self.extent).all()
                or self.extent[0] == self.extent[1] or self.extent[2] == self.extent[3]):
            raise ValueError('extent must contain four finite values with nonzero width and height.')
        self.labels = (_compact_cube_labels([item.filename for item in self.cubes])
                       if cube_labels is None else list(cube_labels))
        if len(self.labels) != len(self.cubes):
            raise ValueError('cube_labels must contain one label per cube, including the primary cube.')
        self.int_unit = int_unit or 'Intensity [%s]' % cube.header.get('BUNIT', 'native units')
        self.vel_unit = vel_unit
        self.stat_func = stat_func
        self.spectrum_kwargs = kwargs
        self.export_path = Path(export_path) if export_path else None
        self.surface_from, self.kwargs_surface = surface_from, dict(kwargs_surface or {})
        self.velocity2d = validate_velocity_maps(velocity2d, cube.data.shape[1:])
        self.kwargs_isov = dict(kwargs_isov or {})
        if {'levels', 'extent', 'origin'} & self.kwargs_isov.keys():
            raise ValueError('kwargs_isov styles contours; levels, extent and origin follow the channel map.')
        self._isov_enabled, self._isov_artists = False, []
        self.selections, self._selection_artists = [], []
        self._surface_artists, self._connections, self._selectors = [], [], []
        self.widgets, self.cursors = {}, []
        self._custom_interactive = None
        self.mode = 'point'
        lo, hi = self._limits([self.cubes[0].data[self.channel]])
        lo, hi = lo if vmin is None else float(vmin), hi if vmax is None else float(vmax)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('Color limits must be finite and vmin < vmax.')
        self.norm = Normalize(lo, hi)
        cmap_obj = _viewer_cmap(cmap)
        self._initial = dict(channel=self.channel, cube=self.cube_index,
                             clim=(lo, hi), cmap=cmap_obj.name)
        self._initial_interactive = cube.interactive
        self._initial_custom_options = dict(kwargs)
        if stat_func is not np.nanmean:
            self._initial_custom_options['stat_func'] = stat_func
        # Use the package style locally so opening a viewer never changes the
        # user's plotting defaults or the original viewer's appearance.
        style = Path(__file__).with_name('discminer.mplstyle')
        with plt.rc_context(fname=str(style), rc=_VIEWER_STYLE):
            self._make_figure(cmap_obj, cmap, pos_unit, cursor_grid, show_beam)
        self._restore_interaction()
        self._initial['limits'] = [(ax.get_xlim(), ax.get_ylim())
                                   for ax in self.image_axes + [self.spectrum_ax]]

    def _restore_interaction(self):
        function = self._initial_interactive
        self.set_mode('box' if function == self.cube._box else 'point')
        if function not in (self.cube._cursor, self.cube._box):
            self._custom_interactive = function(
                self.fig, [self.image_axes[0], self.spectrum_ax], extent=self.extent,
                compare_cubes=self.cubes[1:], **self._initial_custom_options)

    def _make_figure(self, cmap_obj, cmap, pos_unit, cursor_grid, show_beam):
        nimages = 2 if self.side_by_side else 1
        self.fig = plt.figure(figsize=(17 if self.side_by_side else 13, 8))
        self.fig.set_facecolor('white')
        # Extra cube keys can occupy additional rows above the spectrum.
        top = .81 - max(0, len(self.cubes) - 3) * .018
        top = max(.66, top)
        grid = self.fig.add_gridspec(1, nimages + 1, left=.065, right=.925,
                                     bottom=.35, top=top, wspace=.32)
        self.image_axes = [self.fig.add_subplot(grid[0, index]) for index in range(nimages)]
        self.spectrum_ax = self.fig.add_subplot(grid[0, -1])
        self.images = []
        self._show_beam = show_beam
        self._beam_artists = []
        for index, ax in enumerate(self.image_axes):
            ci = index if self.side_by_side else self.cube_index
            self.images.append(ax.imshow(self.cubes[ci].data[self.channel], origin='lower',
                                         extent=self.extent, cmap=cmap_obj, norm=self.norm))
            ax.set_xlabel(pos_unit)
            if index == 0:
                ax.set_ylabel(pos_unit)
            self._style_axes(ax)
            if show_beam and self.cubes[ci].beam is not None:
                before = set(ax.get_children())
                self.cubes[ci].plot_beam(ax)
                self._beam_artists.extend(child for child in ax.get_children() if child not in before)
            if cursor_grid:
                self.cursors.append(Cursor(ax, useblit=True, color='lime', linewidth=.9))
        self.colorbar = self.fig.colorbar(self.images[0], ax=self.image_axes, fraction=.028, pad=.025)
        self.colorbar.set_label(self.int_unit, fontsize=9, labelpad=7)
        self.colorbar.outline.set_linewidth(1.)
        self.colorbar.ax.tick_params(labelsize=8, width=1., length=3.)
        # Use the map's active rectangle, rather than its GridSpec allocation:
        # imshow's equal aspect can shorten the panel vertically or horizontally.
        self.colorbar.ax.set_box_aspect(None)
        self.colorbar.ax.set_aspect('auto')
        self.colorbar.ax.set_axes_locator(self._colorbar_bounds)
        velocities = self.cube.vchannels
        padding = max(abs(velocities[-1] - velocities[0]) * .03, .05)
        self.spectrum_ax.set(xlabel='l.o.s. velocity [%s]' % self.vel_unit, ylabel=self.int_unit,
                             xlim=(velocities[0] - np.sign(velocities[-1] - velocities[0] or 1) * padding,
                                   velocities[-1] + np.sign(velocities[-1] - velocities[0] or 1) * padding))
        self._style_axes(self.spectrum_ax)
        self.spectrum_ax.yaxis.set_label_position('right')
        self.spectrum_ax.yaxis.tick_right()
        self.spectrum_ax.tick_params(axis='y', which='both', left=True, labelleft=False, labelright=True)
        self.spectrum_ax.set_ylim(*self._limits([item.data for item in self.cubes], full=True))
        self.spectrum_ax.margins(y=.12)
        self.current_chan = self.spectrum_ax.axvline(velocities[self.channel], color='0.3', lw=1.2, ls=':')
        self._make_spectrum_key()
        self.channel_title = self.fig.text(.065, .962, '', fontsize=11, color='#263448', va='center')
        self.status = self.fig.text(.05, .038, '', fontsize=8, color='#526174', va='center')
        self._make_controls(cmap)
        self._hover_text = self.fig.text(.05, .014, '', fontsize=8, color='#637084', va='center')
        footer_clip = TransformedBbox(Bbox.from_bounds(.045, 0., .79, .063), self.fig.transFigure)
        for text in (self.status, self._hover_text):
            text.set_clip_box(footer_clip)
            text.set_clip_on(True)
        selector_props = ('props' if Version(matplotlib.__version__) >= Version('3.5') else 'rectprops')
        for ax in self.image_axes:
            selector = RectangleSelector(ax, self._on_box, button=[1],
                                         **{selector_props: dict(facecolor='gold', edgecolor='0.2', alpha=.25)})
            selector.set_active(False)
            self._selectors.append(selector)
        self._connect('button_press_event', self._on_click)
        self._connect('scroll_event', self._on_scroll)
        # F and P are Matplotlib shortcuts for fullscreen/pan. Reserve the
        # viewer keys locally, while keeping the other toolbar shortcuts.
        manager = self.fig.canvas.manager
        default_key = getattr(manager, 'key_press_handler_id', None)
        if default_key is not None:
            self.fig.canvas.mpl_disconnect(default_key)
            self._connect('key_press_event', self._toolbar_keypress)
        self._connect('key_press_event', self._on_key)
        self._connect('motion_notify_event', self._on_hover)
        self._connect('close_event', self._on_close)
        # Retain callbacks/widgets even when callers ignore show()'s return value.
        self.fig._discminer_viewer = self
        self._update_channel(self.channel)
        self.set_mode('point')

    @staticmethod
    def _style_axes(ax):
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.locator_params(axis='both', nbins=5)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.tick_params(which='major', width=1.2, length=4, labelsize=9, pad=5)
        ax.tick_params(which='minor', width=.9, length=2.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

    def _colorbar_bounds(self, ax, renderer):
        panel = self.image_axes[-1].get_position()
        # Physical width/gap stay consistent across one- and two-map layouts.
        figure_width = self.fig.get_figwidth()
        return Bbox.from_bounds(panel.x1 + .12 / figure_width, panel.y0,
                                .10 / figure_width, panel.height)

    def _shade_style(self, index):
        count = len(self.cubes) - 1
        alpha = .24 if count == 1 else float(np.linspace(.28, .11, count)[index - 1])
        # A second cue distinguishes overlapping comparison cubes without
        # introducing extra lines or extra selection-legend entries.
        hatch = ('', '///', chr(92) * 3, '...', 'xxx')[(index - 1) % 5]
        return dict(alpha=alpha, hatch=hatch)

    def _make_spectrum_key(self):
        handles = [Line2D([], [], color='#465164', lw=1.65)]
        labels = ['Line: ' + self.labels[0]]
        for index, label in enumerate(self.labels[1:], 1):
            style = self._shade_style(index)
            handles.append(patches.Patch(facecolor='#465164', edgecolor='#465164', **style))
            labels.append('Shade: ' + label)
        self.spectrum_key = Legend(
            self.spectrum_ax, handles, labels, loc='lower left', bbox_to_anchor=(0., 1.025),
            ncol=2 if len(handles) == 2 else 1, frameon=False, handlelength=1.5,
            handletextpad=.6, columnspacing=1.1, borderaxespad=0., borderpad=0.,
            labelspacing=.35, prop=dict(family='monospace', size=8.3))
        self.spectrum_ax.add_artist(self.spectrum_key)

    @staticmethod
    def _limits(arrays, full=False):
        bounds = []
        for array in arrays:
            array = np.asarray(array)
            # Bound the cost for large cubes, and avoid stacking comparison cubes.
            sample = array.flat[::max(1, (array.size + 99999) // 100000)]
            sample = sample[np.isfinite(sample)]
            if sample.size:
                bounds.append(np.percentile(sample, [0, 100] if full else [1, 99.9]))
        if not bounds:
            return -1., 1.
        lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
        if hi == lo:
            delta = max(abs(lo) * .05, 1.)
            lo, hi = lo - delta, hi + delta
        if full:
            span = hi - lo
            lo, hi = lo - .05 * span, hi + .12 * span
        return float(lo), float(hi)

    def _connect(self, name, callback):
        self._connections.append(self.fig.canvas.mpl_connect(name, callback))

    def _make_controls(self, cmap):
        self.control_groups, self._hints, self.button_icons = {}, {}, {}
        map_positions = [ax.get_position() for ax in self.image_axes]
        left = min(position.x0 for position in map_positions)
        right = max(position.x1 for position in map_positions)
        ax = self.fig.add_axes([left, .902, right - left, .018])
        self.slider = Slider(ax, 'Channel', 0, max(1, self.cube.nchan - 1),
                             valinit=self.channel, valstep=1,
                             valfmt='%d/' + str(self.cube.nchan - 1),
                             #facecolor='#7b9fb9', 
                             facecolor='#FFB000',
                             #edgecolor='#557d9a', 
                             edgecolor='#000000',
                             linewidth=.65,
                             initcolor='none', track_color='#e9eff5',
                             handle_style=dict(facecolor='#f7fafc', edgecolor='#557d9a',
                                               edgewidth=1.05, size=7.5))
        self.slider.set_active(self.cube.nchan > 1)
        self.slider.on_changed(self._update_channel)
        self._style_slider(self.slider)
        self.slider.ax.set_axes_locator(self._slider_bounds)
        self.widgets['channel_slider'] = self.slider
        self._hints[ax] = 'Wheel / Left/Right: change channel. F: first channel. L: last channel.'
        if len(self.cubes) > 1 and not self.side_by_side:
            ax = self.fig.add_axes([left, .867, right - left, .014])
            slider = Slider(ax, 'Cube', 0, len(self.cubes) - 1, valinit=self.cube_index,
                            valstep=1, valfmt='%d', facecolor='#DC267F',
                            edgecolor='#000000', linewidth=.65, initcolor='none',
                            track_color='#e9eff5',
                            handle_style=dict(facecolor='#f7fafc', edgecolor='#7e96aa',
                                              edgewidth=1.05, size=6.5))
            slider.on_changed(self._update_cube)
            self._style_slider(slider)
            slider.ax.set_axes_locator(self._slider_bounds)
            self.widgets['cube_slider'] = slider
            self._hints[ax] = 'Up/Down: choose the cube shown in the channel map; spectra still compare all cubes.'
        if self.side_by_side:
            hint = 'Wheel / arrows: channel   Shift+arrow: 10 steps'
        else:
            hint = 'Wheel / Left/Right: channel   Shift+Left/Right: 10 steps'
            if len(self.cubes) > 1:
                hint += '   Up/Down: cube'
        hint += '   F/L: first/last   P/B: point/box'
        self.shortcut_help = self.fig.text(.96, .965, hint, ha='right', va='center',
                                          fontsize=8, color='black')
        if not self.side_by_side:
            # Follow the actual n/N text width, including wide channel indices.
            self.channel_title.remove()
            self.channel_title = self.slider.ax.annotate(
                '', xy=(1., .5), xycoords=self.slider.valtext,
                xytext=(12., 0.), textcoords='offset points',
                ha='left', va='center', fontsize=10.5, color='#263448',
                annotation_clip=False)
        groups = [('selection', 'SELECT', (.045, .07, .355, .17)),
                  ('contrast', 'CONTRAST', (.415, .07, .305, .17)),
                  ('navigation', 'NAVIGATE', (.735, .07, .145, .17)),
                  ('export', 'SAVE', (.895, .07, .07, .17))]
        for name, title, bounds in groups:
            self._control_card(name, title, bounds)
        actions = [
            ('point', 'Point', lambda event: self.set_mode('point'), 'button_cursor.jpeg',
             'Select a single pixel. Shortcut: P.'),
            ('box', 'Box', lambda event: self.set_mode('box'), 'button_box.png',
             'Drag in either direction to average finite pixels in a box. Shortcut: B.'),
            ('surface', 'Surf', self._toggle_surface, 'button_surface.png',
             'Toggle the supplied emission surface.' if self.surface_from is not None
             else 'Surface overlay requires surface_from when opening the viewer.'),
            ('isov', 'isov', self._toggle_isov, 'isovelocity',
             'Toggle isovelocity contours at the current channel velocity: upper solid, lower dashed.'
             if self.velocity2d is not None else 'Isovelocity contours require velocity2d when opening the viewer.'),
            ('path', 'Path', self._open_path, 'button_path.png',
             'Open the original path extractor in a separate window.'),
            ('clear', 'Clear', lambda event: self.clear(), 'button_trash.jpg',
             'Clear selected spectra and regions; keep channel, zoom and contrast.'),
        ]
        width = (.355 - .026 - (len(actions) - 1) * .005) / len(actions)
        for index, (name, label, callback, icon, hint) in enumerate(actions):
            enabled = ((name != 'surface' or self.surface_from is not None)
                       and (name != 'isov' or self.velocity2d is not None))
            button = self._button(name, label, [.058 + index * (width + .005), .14, width, .052],
                                  callback, icon=icon, hint=hint, enabled=enabled)
            button.label.set_fontsize(7.5)
        self.fig.text(.058, .099, 'P: point   B: box', fontsize=8, color='#738094')
        self._button('auto', 'Auto', [.428, .173, .128, .029], lambda event: self.autoscale(),
                     hint='Use the current channel 1st / 99.9th percentiles for image contrast.')
        self._button('symmetric', 'Symmetric', [.568, .173, .138, .029], lambda event: self.symmetric(),
                     hint='Set equal negative and positive color limits about zero.')
        fields = [
            ('vmin', 'Min', [.462, .13, .082, .028], format(self.norm.vmin, '.4g'), 'Image color minimum.'),
            ('vmax', 'Max', [.605, .13, .10, .028], format(self.norm.vmax, '.4g'), 'Image color maximum.'),
            ('cmap', 'Cmap', [.48, .087, .225, .028], cmap,
             'Colormaps: discminer.intensity, discminer.velocity, discminer.residuals.intensity; add _r to reverse. Matplotlib / CMasher names also work.'),
            ('channel', 'Channel', [.805, .169, .061, .032], str(self.channel), 'Enter a zero-based channel index.'),
            ('velocity', 'Velocity', [.805, .111, .061, .032],
             format(self.cube.vchannels[self.channel], self.vel_fmt), 'Enter a velocity in km/s; select the nearest channel.'),
        ]
        for name, label, rect, initial, hint in fields:
            box = _ViewerTextBox(self.fig.add_axes(rect), label, initial=initial,
                                 color='white', hovercolor='#eef4fa', label_pad=.05)
            box.label.set_fontsize(8)
            box.label.set_color('#526174')
            box.text_disp.set_fontsize(8.5)
            for spine in box.ax.spines.values():
                spine.set(color='#c5ced9', linewidth=.7)
            box.on_submit(lambda value, name=name: self._submit(name, value))
            self.widgets[name] = box
            self._hints[box.ax] = hint
        self._button('export', 'Export', [.904, .14, .052, .052], self._export_clicked,
                     hint='Write selected spectra to CSV and their metadata to JSON.')
        self._button('reset', 'Reset all', [.865, .014, .10, .035], lambda event: self.reset(),
                     hint='Clear selections and overlays; restore all opening view settings.')

    @staticmethod
    def _style_slider(slider):
        slider.ax.set_facecolor('#f5f7fa')
        slider.track.set_edgecolor('#c5d2df')
        slider.track.set_linewidth(.65)
        slider.label.set_fontsize(8.5)
        slider.label.set_color('#526174')
        slider.valtext.set_fontsize(8.5)
        slider.valtext.set_color('#263448')

    def _slider_bounds(self, ax, renderer):
        positions = [panel.get_position() for panel in self.image_axes]
        left = min(position.x0 for position in positions)
        right = max(position.x1 for position in positions)
        y, height = (.902, .018) if ax is self.slider.ax else (.867, .014)
        return Bbox.from_bounds(left, y, right - left, height)

    def _control_card(self, name, title, bounds):
        left, bottom, width, height = bounds
        card = patches.FancyBboxPatch(
            (left, bottom), width, height, boxstyle='round,pad=0.004,rounding_size=0.008',
            transform=self.fig.transFigure, facecolor='#f5f7fa', edgecolor='#dce2ea',
            linewidth=.7, zorder=-1)
        self.fig.add_artist(card)
        self.control_groups[name] = card
        self.fig.text(left + .013, bottom + height - .027, title,
                      fontsize=8, weight='bold', color='#627187')

    def _button(self, name, label, rect, callback, icon=None, hint='', enabled=True):
        button = Button(self.fig.add_axes(rect), label, color='white', hovercolor='#e8f1f9')
        button.label.set_fontsize(8)
        button.label.set_color('#33445a' if enabled else '#9ca7b5')
        for spine in button.ax.spines.values():
            spine.set(color='#c5ced9', linewidth=.7)
        if icon:
            if icon == 'isovelocity':
                # A small vector contour icon stays sharp at every figure DPI.
                drawing = DrawingArea(14, 14, 0, 0)
                t = np.linspace(-1., 1., 40)
                for offset, style in [(-3., '-'), (0., '-'), (3., '--')]:
                    drawing.add_artist(Line2D(7 + offset + 1.7 * np.sin(np.pi * t), 7 + 6 * t,
                                             color='#33445a', lw=1., ls=style,
                                             alpha=1. if enabled else .3))
            else:
                image = plt.imread(Path(__file__).parent.parent / 'icons' / icon)
                drawing = OffsetImage(image, zoom=14. / max(image.shape[:2]), alpha=1. if enabled else .3)
            artist = AnnotationBbox(
                drawing,
                (.18, .5), xycoords='axes fraction', frameon=False, pad=0.)
            button.ax.add_artist(artist)
            button.label.set_x(.64)
            self.button_icons[name] = artist
        button.on_clicked(callback)
        button.set_active(enabled)
        self.widgets[name] = button
        self._hints[button.ax] = hint
        return button

    def _on_hover(self, event):
        text = self._hints.get(event.inaxes, '')
        for label, cube in zip(self.spectrum_key.get_texts(), self.cubes):
            if label.contains(event)[0]:
                text = str(cube.filename)
                break
        if self._hover_text.get_text() != text:
            self._hover_text.set_text(text)
            self.fig.canvas.draw_idle()

    def _message(self, text):
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    def _submit(self, name, value):
        try:
            if name in ('vmin', 'vmax'):
                self.set_clim(**{name: float(value)})
            elif name == 'cmap':
                if value.startswith('cmr.'):
                    import cmasher  # noqa: F401; registers colormaps
                cmap = _viewer_cmap(value)
                for image in self.images:
                    image.set_cmap(cmap)
            elif name == 'channel':
                channel = float(value)
                if int(channel) != channel or not 0 <= channel < self.cube.nchan:
                    raise ValueError('Channel must be an in-range integer.')
                self.set_channel(int(channel))
            elif name == 'velocity':
                velocity = float(value)
                if not np.isfinite(velocity):
                    raise ValueError('Velocity must be finite.')
                self.set_channel(int(np.argmin(abs(self.cube.vchannels - velocity))))
            self._message('Updated %s.' % name)
        except (ValueError, ImportError, OverflowError) as exc:
            self._message(('Invalid %s: %s' % (name, exc))[:180])

    def _set_text(self, name, value):
        box = self.widgets[name]
        if box.capturekeystrokes:
            previous = box.eventson
            box.eventson = False
            box.set_val(value)
            box.eventson = previous
        else:
            # TextBox.set_val synchronously redraws the entire figure to place
            # its editing cursor. Browsing channels only needs a text update;
            # the caller schedules one draw_idle for the complete frame.
            box.text_disp.set_text(str(value))
            box.cursor.set_visible(False)

    def _update_channel(self, value):
        self.channel = int(value)
        for index, image in enumerate(self.images):
            ci = index if self.side_by_side else self.cube_index
            image.set_data(self.cubes[ci].data[self.channel])
            self.image_axes[index].set_title(self.labels[ci], fontsize=10.5, fontfamily='monospace', pad=9)
            self._hints[self.image_axes[index]] = str(self.cubes[ci].filename)
        velocity = self.cube.vchannels[self.channel]
        self.current_chan.set_xdata([velocity, velocity])
        self.channel_title.set_text('v = %s %s' % (format(velocity, self.vel_fmt), self.vel_unit))
        self._set_text('channel', str(self.channel))
        self._set_text('velocity', format(velocity, self.vel_fmt))
        if self._isov_enabled:
            self._draw_isov()
        self.fig.canvas.draw_idle()

    def set_channel(self, channel):
        """Select a channel, clamping integer indices to the available range."""
        channel = int(np.clip(channel, 0, self.cube.nchan - 1))
        if channel != self.channel:
            self.slider.set_val(channel)

    def _update_cube(self, value):
        self.cube_index = int(value)
        for artist in self._beam_artists:
            artist.remove()
        self._beam_artists.clear()
        if self._show_beam and self.cubes[self.cube_index].beam is not None:
            ax = self.image_axes[0]
            before = set(ax.get_children())
            self.cubes[self.cube_index].plot_beam(ax)
            self._beam_artists.extend(child for child in ax.get_children() if child not in before)
        self._update_channel(self.channel)

    def _editing(self):
        return any(isinstance(widget, TextBox) and widget.capturekeystrokes for widget in self.widgets.values())

    def _on_scroll(self, event):
        if event.inaxes in self.image_axes + [self.spectrum_ax] and not self._editing():
            step = np.sign(event.step)
            self.set_channel(self.channel + int(step) * self.channel_step)

    def _on_key(self, event):
        if self._editing():
            return
        key = event.key or ''
        multiplier = 10 if key.startswith('shift+') else 1
        key = key.replace('shift+', '')
        if key in ('up', 'down') and not self.side_by_side:
            slider = self.widgets.get('cube_slider')
            if slider is not None:
                direction = 1 if key == 'up' else -1
                index = int(np.clip(self.cube_index + direction, 0, len(self.cubes) - 1))
                if index != self.cube_index:
                    slider.set_val(index)
        elif key in ('right', 'up', 'left', 'down'):
            direction = 1 if key in ('right', 'up') else -1
            self.set_channel(self.channel + direction * multiplier * self.channel_step)
        elif key.lower() in ('f', 'l'):
            self.set_channel(0 if key.lower() == 'f' else self.cube.nchan - 1)
        elif key.lower() in ('p', 'b'):
            self.set_mode('point' if key.lower() == 'p' else 'box')

    def _toolbar_keypress(self, event):
        key = (event.key or '').lower().replace('shift+', '')
        if key in ('f', 'l', 'p', 'b', 'left', 'right', 'up', 'down') or self._editing():
            return
        key_press_handler(event, self.fig.canvas, self.fig.canvas.toolbar)

    def _toolbar_busy(self):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        return bool(toolbar is not None and toolbar.mode)

    def _on_click(self, event):
        if (event.button != 1 or event.inaxes not in self.image_axes or self.mode != 'point'
                or self._custom_interactive is not None or self._toolbar_busy()):
            return
        pixel = pixel_at(event.xdata, event.ydata, self.cube.data.shape[1:], self.extent)
        if pixel is not None:
            i, j = pixel
            self.select((i, i + 1, j, j + 1), kind='pixel')

    def _on_box(self, press, release):
        if self.mode != 'box' or self._toolbar_busy():
            return
        bounds = box_bounds(press.xdata, release.xdata, press.ydata, release.ydata,
                            self.cube.data.shape[1:], self.extent)
        if bounds is not None:
            self.select(bounds, kind='box')

    def _disconnect_custom(self):
        if self._custom_interactive is not None:
            if isinstance(self._custom_interactive, int):
                self.fig.canvas.mpl_disconnect(self._custom_interactive)
            elif hasattr(self._custom_interactive, 'disconnect_events'):
                self._custom_interactive.disconnect_events()
                self._custom_interactive.set_active(False)
            self._custom_interactive = None

    def set_mode(self, mode):
        if mode not in ('point', 'box'):
            raise ValueError('Mode must be point or box.')
        self._disconnect_custom()
        self.mode = mode
        for selector in self._selectors:
            selector.set_active(mode == 'box')
            selector.set_visible(False)
        for name in ('point', 'box'):
            button = self.widgets[name]
            button.color = '#dcebf7' if name == mode else 'white'
            button.ax.set_facecolor(button.color)
        self._message('%s selection active.' % mode.capitalize())

    def select(self, bounds, kind='box'):
        """Plot all cubes in a pixel region; return its retained selection record."""
        i0, i1, j0, j1 = bounds
        ny, nx = self.cube.data.shape[1:]
        if not (0 <= i0 < i1 <= ny and 0 <= j0 < j1 <= nx):
            raise ValueError('Selection must have nonempty in-range pixel bounds.')
        spectra, counts = zip(*(region_spectrum(item.data, bounds, self.stat_func) for item in self.cubes))
        if not any(np.isfinite(spectrum).any() for spectrum in spectra):
            self._message('Selection contains no finite data; nothing was added.')
            return None
        selection_id = len(self.selections) + 1
        color = plt.get_cmap('tab10')((selection_id - 1) % 10)
        options = dict(where='mid', linewidth=1.65, color=color, linestyle='-',
                       label=str(selection_id), zorder=3)
        options.update(self.spectrum_kwargs)
        # One solid spectrum and one number per selected region. The cube key
        # above the panel explains the comparison fills independently.
        options['label'] = str(selection_id)
        primary = self.spectrum_ax.step(self.cube.vchannels, spectra[0], **options)
        self._selection_artists.extend(primary)
        color = primary[0].get_color()
        for index, spectrum in enumerate(spectra[1:], 1):
            shade = self.spectrum_ax.fill_between(
                self.cube.vchannels, 0., spectrum, step='mid',
                facecolor=color, edgecolor=color, linewidth=.4,
                label='_nolegend_', zorder=2, **self._shade_style(index))
            self._selection_artists.append(shade)
        left, right, bottom, top = self.extent
        x, y = left + j0 * (right - left) / nx, bottom + i0 * (top - bottom) / ny
        width, height = (j1 - j0) * (right - left) / nx, (i1 - i0) * (top - bottom) / ny
        for ax in self.image_axes:
            rect = patches.Rectangle((x, y), width, height, edgecolor=color, facecolor='none', lw=1.8)
            ax.add_patch(rect)
            label = ax.text(x, y + height, str(selection_id), color=color,
                            fontsize=9, fontfamily='monospace')
            self._selection_artists.extend([rect, label])
        record = dict(id=selection_id, kind=kind, bounds=list(map(int, bounds)),
                      spectra=list(spectra), finite_pixels=list(counts))
        self.selections.append(record)
        self.spectrum_ax.relim()
        # Matplotlib's relim ignores filled collections; include model values
        # explicitly so high or negative comparison spectra are never clipped.
        for selection in self.selections:
            for spectrum in selection['spectra']:
                finite = np.isfinite(spectrum)
                self.spectrum_ax.update_datalim(
                    np.column_stack([self.cube.vchannels[finite], spectrum[finite]]))
        if len(self.cubes) > 1:
            self.spectrum_ax.update_datalim([[self.cube.vchannels[0], 0.]])
        self.spectrum_ax.set_autoscaley_on(True)
        self.spectrum_ax.autoscale_view(scalex=False, scaley=True)
        self.spectrum_ax.legend(
            frameon=False, handlelength=1.6, handletextpad=.5, labelspacing=.35,
            prop=dict(family='monospace', size=8.5), ncol=min(4, len(self.selections)))
        self._message('Selection %d: rows %d:%d, columns %d:%d; %s.' % (
            selection_id, i0, i1, j0, j1, getattr(self.stat_func, '__name__', 'custom statistic')))
        return record

    def clear(self):
        """Remove selections in place, retaining channel, contrast and image zoom."""
        for artist in self._selection_artists:
            artist.remove()
        self._selection_artists.clear()
        self.selections.clear()
        legend = self.spectrum_ax.get_legend()
        if legend is not None:
            legend.remove()
        for selector in self._selectors:
            selector.set_visible(False)
        self._message('Selections cleared.')

    def reset(self):
        """Clear selections/overlays and restore the viewer's opening settings."""
        for widget in self.widgets.values():
            if isinstance(widget, TextBox) and widget.capturekeystrokes:
                previous = widget.eventson
                widget.eventson = False
                widget.stop_typing()
                widget.eventson = previous
        toolbar = self.fig.canvas.toolbar
        if toolbar is not None:
            mode = str(toolbar.mode).lower()
            if 'pan' in mode:
                toolbar.pan()
            elif 'zoom' in mode:
                toolbar.zoom()
        self.clear()
        for artist in self._surface_artists:
            artist.remove()
        self._surface_artists.clear()
        self._isov_enabled = False
        self._remove_isov()
        for name in ('surface', 'isov'):
            self.widgets[name].color = 'white'
            self.widgets[name].ax.set_facecolor('white')
        self.set_channel(self._initial['channel'])
        if 'cube_slider' in self.widgets:
            self.widgets['cube_slider'].set_val(self._initial['cube'])
        self.set_clim(*self._initial['clim'])
        for image in self.images:
            image.set_cmap(_viewer_cmap(self._initial['cmap']))
        self._set_text('cmap', self._initial['cmap'])
        # Also replace any invalid or unfinished channel/velocity text.
        self._update_channel(self._initial['channel'])
        self._restore_interaction()
        self.spectrum_ax.relim()
        for ax, (xlim, ylim) in zip(self.image_axes + [self.spectrum_ax], self._initial['limits']):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        for cursor in self.cursors:
            cursor.lineh.set_visible(False)
            cursor.linev.set_visible(False)
        if toolbar is not None:
            toolbar.update()
            toolbar.push_current()
        self._hover_text.set_text('')
        self._message('Viewer reset to its opening defaults.')

    def set_clim(self, vmin=None, vmax=None):
        lo = self.norm.vmin if vmin is None else float(vmin)
        hi = self.norm.vmax if vmax is None else float(vmax)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError('Require finite Min < Max.')
        # Assign both limits atomically: a colorbar can otherwise adjust them
        # while the first limit temporarily crosses the previous second limit.
        with self.norm.callbacks.blocked():
            self.norm.vmin, self.norm.vmax = lo, hi
        for image in self.images:
            image.changed()
        self._set_text('vmin', format(lo, '.4g'))
        self._set_text('vmax', format(hi, '.4g'))
        self.fig.canvas.draw_idle()

    def autoscale(self):
        """Use 1st/99.9th percentiles of the displayed channel(s)."""
        self.set_clim(*self._limits([image.get_array() for image in self.images]))
        self._message('Color limits use the current channel percentiles; spectrum limits are independent.')

    def symmetric(self):
        limit = max(abs(self.norm.vmin), abs(self.norm.vmax)) or 1.
        self.set_clim(-limit, limit)
        self._message('Color limits are symmetric about zero.')

    def _toggle_surface(self, event):
        if self._surface_artists:
            for artist in self._surface_artists:
                artist.remove()
            self._surface_artists.clear()
        else:
            for ax in self.image_axes:
                before = set(ax.get_children())
                self.cube._surface(ax, self.surface_from, **self.kwargs_surface)
                self._surface_artists.extend(child for child in ax.get_children() if child not in before)
        button = self.widgets['surface']
        button.color = '#dcebf7' if self._surface_artists else 'white'
        button.ax.set_facecolor(button.color)
        self.fig.canvas.draw_idle()

    def _remove_isov(self):
        for artist in self._isov_artists:
            artist.remove()
        self._isov_artists.clear()

    def _draw_isov(self):
        self._remove_isov()
        if not self._isov_enabled:
            return
        velocity = self.cube.vchannels[self.channel]
        for ax in self.image_axes:
            for side, style in [('upper', '-'), ('lower', '--')]:
                values = self.velocity2d.get(side)
                if values is None or not values.count():
                    continue
                # Keep the toggle enabled even where no contour crosses the map.
                if not values.min() < velocity < values.max():
                    continue
                options = dict(colors='k', linewidths=1.0, linestyles=style, zorder=3)
                options.update(self.kwargs_isov)
                contour = ax.contour(values, levels=[velocity], extent=self.extent,
                                     origin='lower', **options)
                if hasattr(contour, 'remove'):
                    self._isov_artists.append(contour)
                else:  # Matplotlib before 3.8 uses separate collection artists.
                    self._isov_artists.extend(contour.collections)

    def _toggle_isov(self, event):
        if self.velocity2d is None:
            return
        self._isov_enabled = not self._isov_enabled
        self._draw_isov()
        button = self.widgets['isov']
        button.color = '#dcebf7' if self._isov_enabled else 'white'
        button.ax.set_facecolor(button.color)
        self._message('Isovelocity contours %s: upper solid, lower dashed.' %
                      ('enabled' if self._isov_enabled else 'disabled'))

    def _open_path(self, event):
        # The existing path extractor remains a separate window; retain this viewer.
        return_options = {}
        if self.velocity2d is not None:
            return_options['_return_to'] = partial(self.cube.show, velocity2d=self.velocity2d,
                                                  kwargs_isov=self.kwargs_isov)
        self.cube._show_path(extent=self.extent, chan_init=self.channel,
                             cube_init=self.cube_index, compare_cubes=self.cubes[1:],
                             cmap=self.images[0].get_cmap().name, int_unit=self.int_unit,
                             vel_unit=self.vel_unit, surface_from=self.surface_from,
                             kwargs_surface=self.kwargs_surface,
                             vmin=self.norm.vmin, vmax=self.norm.vmax, **return_options)

    def export(self, path=None):
        """Write selected spectra and provenance; return (CSV path, JSON path)."""
        from .._version import __version__

        if not self.selections:
            raise ValueError('Select at least one pixel or box before exporting.')
        if path is None and self.export_path is None:
            raise ValueError('Pass export_path to show(), or a path to viewer.export().')
        path = Path(path) if path is not None else self.export_path
        path = path.expanduser().with_suffix('.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['selection', 'cube_index', 'cube', 'channel', 'velocity_kms',
                             'intensity', 'intensity_unit', 'finite_pixels'])
            for selection in self.selections:
                for index, spectrum in enumerate(selection['spectra']):
                    for channel, (velocity, intensity) in enumerate(zip(self.cube.vchannels, spectrum)):
                        writer.writerow([selection['id'], index, self.labels[index], channel,
                                         velocity, intensity, self.cubes[index].header.get('BUNIT', ''),
                                         selection['finite_pixels'][index][channel]])
        metadata = dict(
            format_version=1, channel=self.channel, cube_index=self.cube_index,
            software=dict(discminer=__version__, matplotlib=matplotlib.__version__, numpy=np.__version__),
            extent=self.extent, cmap=self.images[0].get_cmap().name,
            clim=[self.norm.vmin, self.norm.vmax], velocity_unit='km/s',
            statistic=getattr(self.stat_func, '__name__', repr(self.stat_func)),
            pixel_bounds_convention='zero-based half-open [row_start, row_stop, column_start, column_stop]',
            selections=[{key: value for key, value in selection.items()
                         if key not in ('spectra', 'finite_pixels')} for selection in self.selections],
            cubes=[dict(label=label, filename=str(item.filename), distance_pc=item.dpc.to_value(u.pc),
                        beam=None if item.beam is None else dict(
                            major_arcsec=item.beam.major.to_value(u.arcsec),
                            minor_arcsec=item.beam.minor.to_value(u.arcsec),
                            pa_deg=item.beam.pa.to_value(u.deg)),
                        fits_header=item.header.tostring(sep='\n', padding=False))
                   for label, item in zip(self.labels, self.cubes)])
        sidecar = path.with_suffix('.json')
        sidecar.write_text(json.dumps(metadata, indent=2) + '\n')
        return path, sidecar

    def _export_clicked(self, event):
        try:
            csv_path, _ = self.export()
            self._message('Exported %s' % csv_path)
        except (ValueError, OSError) as exc:
            self._message(str(exc))

    def _on_close(self, event):
        self._disconnect_custom()
        for widget in self.widgets.values():
            if isinstance(widget, TextBox) and widget.capturekeystrokes:
                previous = widget.eventson
                widget.eventson = False
                widget.stop_typing()
                widget.eventson = previous
            widget.disconnect_events()
        self.slider.disconnect_events()
        for cursor in self.cursors:
            cursor.disconnect_events()
        for selector in self._selectors:
            selector.disconnect_events()
        for cid in self._connections:
            self.fig.canvas.mpl_disconnect(cid)
        self._connections.clear()
