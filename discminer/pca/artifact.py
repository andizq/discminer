"""FITS serialization and reconstruction for discminer PCA results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits


ARTIFACT_VERSION = 1


def _as_float_array(values):
    return np.asarray(values, dtype=float)


@dataclass
class PCAResult:
    """Complete PCA decomposition and its source-cube metadata."""

    source_header: fits.Header
    source_path: str
    distance_pc: float
    mean_sub: bool
    selected_components: int
    eigenimages: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    covariance: np.ndarray
    velocity: np.ndarray
    channel_mean: np.ndarray
    spatial_width: np.ndarray
    spatial_width_error: np.ndarray
    spectral_width: np.ndarray
    spectral_width_error: np.ndarray
    spatial_autocorrelation: np.ndarray
    spectral_autocorrelation: np.ndarray
    valid_mask: np.ndarray
    outer_radius_au: Optional[float] = None
    eigen_cut_method: str = "components"
    min_eigenvalue: Optional[float] = None
    spatial_method: str = "contour"
    spectral_method: str = "walk-down"
    beam_correct: bool = True

    def __post_init__(self):
        self.source_header = self.source_header.copy()
        self.source_path = str(self.source_path)
        self.distance_pc = float(self.distance_pc)
        self.mean_sub = bool(self.mean_sub)
        self.selected_components = int(self.selected_components)

        for name in (
            "eigenimages",
            "eigenvectors",
            "eigenvalues",
            "covariance",
            "velocity",
            "channel_mean",
            "spatial_width",
            "spatial_width_error",
            "spectral_width",
            "spectral_width_error",
            "spatial_autocorrelation",
            "spectral_autocorrelation",
        ):
            setattr(self, name, _as_float_array(getattr(self, name)))

        self.valid_mask = np.asarray(self.valid_mask, dtype=bool)
        if self.outer_radius_au is not None:
            self.outer_radius_au = float(self.outer_radius_au)
        if self.min_eigenvalue is not None:
            self.min_eigenvalue = float(self.min_eigenvalue)

        self._validate()

    @property
    def n_components(self):
        return self.eigenimages.shape[0]

    @property
    def n_channels(self):
        return self.velocity.size

    @property
    def variance_fraction(self):
        total = np.sum(self.eigenvalues)
        if not np.isfinite(total) or total == 0:
            return np.full(self.eigenvalues.shape, np.nan)
        return self.eigenvalues / total

    @property
    def cumulative_variance(self):
        return np.cumsum(self.variance_fraction)

    def _validate(self):
        if self.eigenimages.ndim != 3:
            raise ValueError(
                "eigenimages must have shape (n_components, ny, nx)"
            )

        n_components, ny, nx = self.eigenimages.shape
        n_channels = self.velocity.size

        if self.eigenvectors.shape != (n_channels, n_components):
            raise ValueError(
                "eigenvectors must have shape (n_channels, n_components)"
            )
        if self.eigenvalues.shape != (n_components,):
            raise ValueError("eigenvalues must have one value per component")
        if self.covariance.shape != (n_channels, n_channels):
            raise ValueError(
                "covariance must have shape (n_channels, n_channels)"
            )
        if self.channel_mean.shape != (n_channels,):
            raise ValueError("channel_mean must have one value per channel")
        if self.valid_mask.shape != (n_channels, ny, nx):
            raise ValueError(
                "valid_mask must match the reconstructed cube shape"
            )
        if self.spatial_autocorrelation.shape != (
            self.selected_components,
            ny,
            nx,
        ):
            raise ValueError(
                "spatial_autocorrelation must have shape "
                "(selected_components, ny, nx)"
            )
        if self.spectral_autocorrelation.shape != (
            n_channels,
            self.selected_components,
        ):
            raise ValueError(
                "spectral_autocorrelation must have shape "
                "(n_channels, selected_components)"
            )
        if not 1 <= self.selected_components <= n_components:
            raise ValueError(
                "selected_components must be between 1 and n_components"
            )

        for name in (
            "spatial_width",
            "spatial_width_error",
            "spectral_width",
            "spectral_width_error",
        ):
            if getattr(self, name).shape != (n_components,):
                raise ValueError(f"{name} must have one value per component")

    def component_mask(
        self,
        include: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        exclude_tail: Optional[int] = None,
    ):
        """Return the selected-component mask using zero-based indices."""

        keep = np.ones(self.n_components, dtype=bool)

        if include is not None:
            include = self._validate_component_indices(include, "include")
            keep[:] = False
            keep[include] = True

        if exclude is not None:
            exclude = self._validate_component_indices(exclude, "exclude")
            keep[exclude] = False

        if exclude_tail is not None:
            if not 0 <= exclude_tail <= self.n_components:
                raise IndexError(
                    "exclude_tail must be between 0 and the number of "
                    "components"
                )
            keep[exclude_tail:] = False

        return keep

    def _validate_component_indices(self, indices, argument):
        indices = np.asarray(list(indices), dtype=int)
        if indices.ndim != 1:
            raise ValueError(f"{argument} must be a one-dimensional sequence")
        if np.any(indices < 0) or np.any(indices >= self.n_components):
            raise IndexError(
                f"{argument} contains a component outside the available "
                f"range 0 to {self.n_components - 1}"
            )
        return indices

    def reconstruct(
        self,
        include: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        exclude_tail: Optional[int] = None,
    ):
        """Reconstruct the source cube from a selected component set."""

        keep = self.component_mask(
            include=include,
            exclude=exclude,
            exclude_tail=exclude_tail,
        )
        ny, nx = self.eigenimages.shape[1:]

        if np.any(keep):
            component_images = self.eigenimages[keep].reshape(
                np.count_nonzero(keep), ny * nx
            )
            reconstructed = self.eigenvectors[:, keep] @ component_images
        else:
            reconstructed = np.zeros((self.n_channels, ny * nx), dtype=float)

        reconstructed += self.channel_mean[:, np.newaxis]
        reconstructed = reconstructed.reshape(self.valid_mask.shape)
        reconstructed[~self.valid_mask] = np.nan
        return reconstructed


def _source_header_hdu(header):
    cards = np.asarray([card.image for card in header.cards], dtype="S80")
    column = fits.Column(name="CARD", format="80A", array=cards)
    return fits.BinTableHDU.from_columns([column], name="SOURCE_HEADER")


def _read_source_header(hdu):
    cards = []
    for value in hdu.data["CARD"]:
        if isinstance(value, bytes):
            value = value.decode("ascii")
        cards.append(str(value).ljust(80)[:80])
    return fits.Header.fromstring("".join(cards), sep="")


def _eigenimage_header(source_header):
    header = source_header.copy()

    for key in (
        "NAXIS3",
        "CTYPE3",
        "CUNIT3",
        "CRPIX3",
        "CRVAL3",
        "CDELT3",
        "CD3_3",
        "PC3_3",
        "RESTFRQ",
        "RESTFREQ",
    ):
        header.remove(key, ignore_missing=True, remove_all=True)

    header["CTYPE3"] = "PCA-COMP"
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = 0.0
    header["CDELT3"] = 1.0
    header["BTYPE"] = "PCA eigenimage"
    return header


def write_pca_artifact(result, filename, overwrite=True):
    """Write a :class:`PCAResult` to one multi-extension FITS file."""

    filename = Path(filename)
    primary_header = fits.Header()
    primary_header["PCAFORM"] = ARTIFACT_VERSION
    primary_header["NCOMP"] = result.n_components
    primary_header["NSELECT"] = result.selected_components
    primary_header["MEANSUB"] = result.mean_sub
    primary_header["DISTPC"] = result.distance_pc
    primary_header["SRCFILE"] = result.source_path
    primary_header["ECUTMETH"] = result.eigen_cut_method
    primary_header["SPATMETH"] = result.spatial_method
    primary_header["SPECMETH"] = result.spectral_method
    primary_header["BEAMCORR"] = result.beam_correct
    if result.min_eigenvalue is not None:
        primary_header["MINEIG"] = result.min_eigenvalue
    if result.outer_radius_au is not None:
        primary_header["ROUTAU"] = result.outer_radius_au

    components = np.arange(result.n_components, dtype=np.int32)
    table_columns = [
        fits.Column(name="COMPONENT", format="J", array=components),
        fits.Column(
            name="EIGENVALUE", format="D", array=result.eigenvalues
        ),
        fits.Column(
            name="VARIANCE", format="D", array=result.variance_fraction
        ),
        fits.Column(
            name="CUMVAR", format="D", array=result.cumulative_variance
        ),
        fits.Column(
            name="SPATIAL_WIDTH",
            format="D",
            unit="au",
            array=result.spatial_width,
        ),
        fits.Column(
            name="SPATIAL_ERROR",
            format="D",
            unit="au",
            array=result.spatial_width_error,
        ),
        fits.Column(
            name="SPECTRAL_WIDTH",
            format="D",
            unit="km/s",
            array=result.spectral_width,
        ),
        fits.Column(
            name="SPECTRAL_ERROR",
            format="D",
            unit="km/s",
            array=result.spectral_width_error,
        ),
    ]

    velocity_columns = [
        fits.Column(
            name="CHANNEL",
            format="J",
            array=np.arange(result.n_channels, dtype=np.int32),
        ),
        fits.Column(
            name="VELOCITY",
            format="D",
            unit="km/s",
            array=result.velocity,
        ),
    ]

    hdus = [
        fits.PrimaryHDU(header=primary_header),
        fits.ImageHDU(
            data=result.eigenimages,
            header=_eigenimage_header(result.source_header),
            name="EIGENIMAGES",
        ),
        fits.ImageHDU(data=result.eigenvectors, name="EIGENVECTORS"),
        fits.BinTableHDU.from_columns(table_columns, name="EIGENVALUES"),
        fits.ImageHDU(data=result.covariance, name="COVARIANCE"),
        fits.BinTableHDU.from_columns(velocity_columns, name="VELOCITY"),
        fits.ImageHDU(data=result.channel_mean, name="MEAN"),
        fits.CompImageHDU(
            data=result.valid_mask.astype(np.uint8), name="VALIDMASK"
        ),
        _source_header_hdu(result.source_header),
    ]
    hdus.append(
        fits.ImageHDU(
            data=result.spatial_autocorrelation,
            name="SPATIAL_ACF",
        )
    )
    hdus.append(
        fits.ImageHDU(
            data=result.spectral_autocorrelation,
            name="SPECTRAL_ACF",
        )
    )
    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)
    return filename


def read_pca_artifact(filename):
    """Read a complete PCA result from a multi-extension FITS artifact."""

    filename = Path(filename)
    with fits.open(filename, memmap=False) as hdul:
        primary = hdul[0].header
        version = primary.get("PCAFORM")
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported PCA artifact version {version!r}; "
                f"expected {ARTIFACT_VERSION}"
            )

        summary = hdul["EIGENVALUES"].data
        velocity_table = hdul["VELOCITY"].data

        return PCAResult(
            source_header=_read_source_header(hdul["SOURCE_HEADER"]),
            source_path=primary.get("SRCFILE", ""),
            distance_pc=primary["DISTPC"],
            mean_sub=primary["MEANSUB"],
            selected_components=primary["NSELECT"],
            eigenimages=np.array(hdul["EIGENIMAGES"].data, dtype=float),
            eigenvectors=np.array(hdul["EIGENVECTORS"].data, dtype=float),
            eigenvalues=np.array(summary["EIGENVALUE"], dtype=float),
            covariance=np.array(hdul["COVARIANCE"].data, dtype=float),
            velocity=np.array(velocity_table["VELOCITY"], dtype=float),
            channel_mean=np.array(hdul["MEAN"].data, dtype=float),
            spatial_width=np.array(summary["SPATIAL_WIDTH"], dtype=float),
            spatial_width_error=np.array(
                summary["SPATIAL_ERROR"], dtype=float
            ),
            spectral_width=np.array(
                summary["SPECTRAL_WIDTH"], dtype=float
            ),
            spectral_width_error=np.array(
                summary["SPECTRAL_ERROR"], dtype=float
            ),
            valid_mask=np.array(hdul["VALIDMASK"].data, dtype=bool),
            spatial_autocorrelation=np.array(
                hdul["SPATIAL_ACF"].data,
                dtype=float,
            ),
            spectral_autocorrelation=np.array(
                hdul["SPECTRAL_ACF"].data,
                dtype=float,
            ),
            outer_radius_au=primary.get("ROUTAU"),
            eigen_cut_method=primary.get("ECUTMETH", "components"),
            min_eigenvalue=primary.get("MINEIG"),
            spatial_method=primary.get("SPATMETH", "contour"),
            spectral_method=primary.get("SPECMETH", "walk-down"),
            beam_correct=primary.get("BEAMCORR", True),
        )


def write_reconstructed_cube(
    result,
    filename,
    include=None,
    exclude=None,
    exclude_tail=None,
    overwrite=True,
):
    """Reconstruct selected components and write a standard FITS cube."""

    data = result.reconstruct(
        include=include,
        exclude=exclude,
        exclude_tail=exclude_tail,
    )
    header = result.source_header.copy()
    header.add_history("Reconstructed from a discminer PCA artifact")
    if include is not None:
        header.add_history(f"Included PCA components: {list(include)}")
    if exclude is not None:
        header.add_history(f"Excluded PCA components: {list(exclude)}")
    if exclude_tail is not None:
        header.add_history(
            f"Excluded PCA components from {exclude_tail} onward"
        )
    fits.writeto(filename, data, header=header, overwrite=overwrite)
    return Path(filename)
