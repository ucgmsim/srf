"""An SRF's records: plane headers, point columns and slip-rate pulses.

Every value is in the units the file stores, which are CGS for the slip and the rock.
Slip is in centimetres, slip rate and shear speed in centimetres per second, area
in square centimetres and density in grams per cubic centimetre. Depths, lengths and
widths are in kilometres. Angles and positions are in degrees, and times in seconds.

The point columns and pulses are single precision, as the file holds them. Each
record checks its arrays when it's built and keeps read-only views of them. A view
doesn't copy, so an array the caller still holds and writes to changes the record
too.
"""

import dataclasses
import itertools
from collections.abc import Iterable, Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from srf._core import SrfError

POINT_COLUMNS = (
    "lon",
    "lat",
    "dep",
    "stk",
    "dip",
    "area",
    "tinit",
    "dt",
    "rake",
    "slip1",
)
"""The point columns every SRF has, in the file's order."""


def _column(name: str, values: ArrayLike, dtype: DTypeLike) -> np.ndarray:
    """A read-only, contiguous, one-dimensional view of ``values`` as ``dtype``."""
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.ndim != 1:
        raise SrfError(f"{name} must be one-dimensional, not shape {array.shape}")
    view = array.view()
    view.flags.writeable = False
    return view


def _index_column(name: str, values: ArrayLike) -> np.ndarray:
    """``values`` as read-only non-negative integers the extension can take."""
    array = np.asarray(values)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise SrfError(f"{name} must hold integers, not {array.dtype}")
    if array.size and array.min() < 0:
        raise SrfError(f"{name} must not be negative")
    return _column(name, array, np.uintp)


@dataclasses.dataclass(frozen=True)
class Plane:
    """One plane's header.

    The header locates and sizes the plane. ``POINTS`` holds the geometry, and
    readers such as EMOD3D read nothing else, so a header need not describe its
    points exactly.

    Parameters
    ----------
    elon, elat : float
        Longitude and latitude of the centre of the plane's top edge, in degrees.
    nstk, ndip : int
        Points along strike and down dip.
    len, wid : float
        Length along strike and width down dip, in kilometres.
    stk, dip : float
        Strike and dip, in degrees.
    dtop : float
        Depth of the top edge, in kilometres.
    shyp, dhyp : float
        The hypocentre along strike from the top edge's centre and down dip from the
        top edge, in kilometres.
    """

    elon: float
    elat: float
    nstk: int
    ndip: int
    len: float
    wid: float
    stk: float
    dip: float
    dtop: float
    shyp: float
    dhyp: float

    def __post_init__(self) -> None:
        """Check the fields and coerce them to the file's types."""
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name in ("nstk", "ndip"):
                if isinstance(value, bool) or int(value) != value or value < 0:
                    raise SrfError(
                        f"{field.name} must be a non-negative integer, not {value!r}"
                    )
                object.__setattr__(self, field.name, int(value))
            else:
                object.__setattr__(self, field.name, float(value))

    @property
    def points(self) -> int:
        """int: How many points the plane has, ``nstk * ndip``."""
        return self.nstk * self.ndip


@dataclasses.dataclass(frozen=True, eq=False)
class Points:
    """Every point's columns, one array each, in file order.

    A plane's points run along strike first, then down dip.

    Parameters
    ----------
    lon, lat : np.ndarray
        Position in degrees.
    dep : np.ndarray
        Depth in kilometres.
    stk, dip : np.ndarray
        Strike and dip in degrees.
    area : np.ndarray
        Area in square centimetres.
    tinit : np.ndarray
        When the point starts to slip, in seconds.
    dt : np.ndarray
        The slip-rate sample interval in seconds.
    rake : np.ndarray
        Rake in degrees.
    slip1 : np.ndarray
        The slip, in centimetres.
    vs : np.ndarray, optional
        Shear speed in centimetres per second. Version 2 only.
    density : np.ndarray, optional
        Density in grams per cubic centimetre. Version 2 only.

    Raises
    ------
    SrfError
        If a column isn't one-dimensional, the columns differ in length, or the
        points have only one of ``vs`` and ``density``.
    """

    lon: NDArray[np.float32]
    lat: NDArray[np.float32]
    dep: NDArray[np.float32]
    stk: NDArray[np.float32]
    dip: NDArray[np.float32]
    area: NDArray[np.float32]
    tinit: NDArray[np.float32]
    dt: NDArray[np.float32]
    rake: NDArray[np.float32]
    slip1: NDArray[np.float32]
    vs: NDArray[np.float32] | None = None
    density: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        """Check the fields and coerce them to the file's types."""
        if (self.vs is None) != (self.density is None):
            raise SrfError("vs and density must both be given (version 2) or neither")
        lengths = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None:
                column = _column(field.name, value, np.float32)
                object.__setattr__(self, field.name, column)
                lengths[field.name] = column.size
        if len(set(lengths.values())) > 1:
            raise SrfError(f"The point columns differ in length: {lengths}")

    def __len__(self) -> int:
        """How many points there are."""
        return self.lon.size

    @property
    def version(self) -> int:
        """int: The SRF version these columns make, 2 with the rock and 1 without."""
        return 1 if self.vs is None else 2

    def __getitem__(self, index: slice) -> Self:
        """A run of points, as views of these columns.

        Parameters
        ----------
        index : slice
            The points to take.

        Returns
        -------
        Points
            The points ``index`` selects.
        """
        columns = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is not None:
                columns[field.name] = value[index]
        return dataclasses.replace(self, **columns)


@dataclasses.dataclass(frozen=True, eq=False)
class SlipRate:
    """Every point's slip-rate pulse, as the rows of a compressed sparse row matrix.

    Row ``k`` is point ``k``'s pulse, in centimetres per second. Column ``j`` is the
    sample at ``tinit + j * dt``. Each pulse counts from its own onset rather than
    from the start of the rupture.

    Parameters
    ----------
    offsets : np.ndarray
        ``n_points + 1`` offsets into ``values``. Row ``k`` is
        ``values[offsets[k]:offsets[k + 1]]``.
    values : np.ndarray
        Every row's stored samples, end to end.
    columns : np.ndarray, optional
        Each stored sample's column. Without it every row is dense from column
        zero, which is how an SRF stores a pulse and how :func:`srf.read` returns
        one. With it, a row may skip columns, as a ``scipy.sparse`` matrix that
        dropped its zeros does, and the writer fills those in with zeros.

    Raises
    ------
    SrfError
        If ``offsets`` doesn't start at zero, decreases, or doesn't end at the
        number of values, or the columns are negative, differ in number from the
        values, or don't strictly increase along a row.
    """

    offsets: NDArray[np.uintp]
    values: NDArray[np.float32]
    columns: NDArray[np.uintp] | None = None

    def __post_init__(self) -> None:
        """Check the fields and coerce them to the file's types."""
        offsets = _index_column("offsets", self.offsets)
        values = _column("values", self.values, np.float32)
        if (
            offsets.size == 0
            or offsets[0] != 0
            or np.any(np.diff(offsets.astype(np.int64)) < 0)
            or offsets[-1] != values.size
        ):
            raise SrfError(
                "offsets must start at 0, never decrease, and end at the number of "
                f"values ({values.size})"
            )
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "values", values)
        if self.columns is not None:
            columns = _index_column("columns", self.columns)
            if columns.size != values.size:
                raise SrfError(
                    f"There are {columns.size} columns for {values.size} values"
                )
            steps = np.diff(columns.astype(np.int64))
            # A step from one row's last sample to the next row's first is a new row,
            # not a step along one.
            starts = offsets[1:-1].astype(np.int64)
            starts = starts[(starts > 0) & (starts < values.size)]
            along = np.ones(steps.size, dtype=bool)
            along[starts - 1] = False
            if np.any(steps[along] <= 0):
                raise SrfError("The columns must strictly increase along each row")
            object.__setattr__(self, "columns", columns)

    @classmethod
    def from_pulses(cls, pulses: Iterable[ArrayLike]) -> Self:
        """The matrix whose rows are ``pulses``, each dense from its onset.

        Parameters
        ----------
        pulses : iterable of array_like
            Each point's slip rate, in centimetres per second.

        Returns
        -------
        SlipRate
            The pulses as rows.
        """
        rows = [_column("pulse", pulse, np.float32) for pulse in pulses]
        offsets = np.fromiter(
            itertools.accumulate((row.size for row in rows), initial=0),
            dtype=np.uintp,
        )
        values = np.concatenate(rows) if rows else np.empty(0, dtype=np.float32)
        return cls(offsets=offsets, values=values)

    def __len__(self) -> int:
        """How many rows, one per point, there are."""
        return self.offsets.size - 1

    @property
    def samples(self) -> NDArray[np.int64]:
        """NDArray[np.int64]: How many samples each pulse spans in the file."""
        if self.columns is None:
            return np.diff(self.offsets.astype(np.int64))
        ends = self.offsets[1:].astype(np.int64)
        counts = np.zeros(len(self), dtype=np.int64)
        stored = ends > self.offsets[:-1]
        counts[stored] = self.columns[ends[stored] - 1].astype(np.int64) + 1
        return counts

    def pulse(self, point: int) -> NDArray[np.float32]:
        """One point's pulse, dense from its onset.

        Parameters
        ----------
        point : int
            The point's index.

        Returns
        -------
        np.ndarray
            The slip rate at each sample, in centimetres per second.
        """
        start, stop = int(self.offsets[point]), int(self.offsets[point + 1])
        if self.columns is None:
            return self.values[start:stop]
        dense = np.zeros(int(self.samples[point]), dtype=np.float32)
        dense[self.columns[start:stop]] = self.values[start:stop]
        return dense

    def __getitem__(self, index: slice) -> Self:
        """A run of rows, with their offsets re-based to zero.

        Parameters
        ----------
        index : slice
            The rows to take. The step must be 1.

        Returns
        -------
        SlipRate
            The rows ``index`` selects.

        Raises
        ------
        ValueError
            If the slice has a step other than 1.
        """
        start, stop, step = index.indices(len(self))
        if step != 1:
            raise ValueError("Rows can only be taken as a contiguous run")
        stop = max(start, stop)
        offsets = self.offsets[start : stop + 1]
        first, last = int(offsets[0]), int(offsets[-1])
        return type(self)(
            offsets=offsets - offsets[0],
            values=self.values[first:last],
            columns=None if self.columns is None else self.columns[first:last],
        )


@dataclasses.dataclass(frozen=True, eq=False)
class Srf:
    """A whole SRF: its planes, then their points and pulses in plane order.

    Parameters
    ----------
    planes : sequence of Plane
        The plane headers.
    points : Points
        Every plane's points, the first plane's first.
    slip_rate : SlipRate
        One pulse per point.

    Raises
    ------
    SrfError
        If the planes, the points and the pulses disagree on how many points there
        are.
    """

    planes: Sequence[Plane]
    points: Points
    slip_rate: SlipRate

    def __post_init__(self) -> None:
        """Check the fields and coerce them to the file's types."""
        object.__setattr__(self, "planes", tuple(self.planes))
        declared = sum(plane.points for plane in self.planes)
        if not declared == len(self.points) == len(self.slip_rate):
            raise SrfError(
                f"The planes declare {declared} points, but there are "
                f"{len(self.points)} points and {len(self.slip_rate)} pulses"
            )

    @property
    def version(self) -> int:
        """int: The SRF version, 2 if the points record the rock and 1 if not."""
        return self.points.version

    @property
    def rise(self) -> NDArray[np.float64]:
        """NDArray[np.float64]: Each point's rise time in seconds, samples times dt."""
        return self.slip_rate.samples * self.points.dt.astype(np.float64)

    @property
    def plane_points(self) -> tuple[slice, ...]:
        """tuple of slice: Each plane's run of points and pulses."""
        edges = itertools.accumulate((plane.points for plane in self.planes), initial=0)
        return tuple(slice(a, b) for a, b in itertools.pairwise(edges))


__all__ = ["POINT_COLUMNS", "Plane", "Points", "SlipRate", "Srf", "SrfError"]
