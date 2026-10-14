"""Reading and writing SRF files, over the Rust codec in ``srf._core``.

The reader takes versions 1.0 and 2.0, and :func:`write` writes either whole. The
streaming :class:`Writer` writes version 2.0 only. The ``slip1`` component is the
only one they handle, and the reader refuses a file with a ``slip2`` or ``slip3``
pulse. Both read and write the header's comments, the ``#`` lines between the version
line and the plane headers.
"""

import dataclasses
import mmap
from collections.abc import Buffer, Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

from srf import _core
from srf.types import Plane, Points, SlipRate, Srf, SrfError, check_comment

type FilePath = str | Path


def _from_core(parsed: _core.PySrfFile) -> Srf:
    names = [field.name for field in dataclasses.fields(Plane)]
    metadata = parsed.metadata
    return Srf(
        planes=[
            Plane(**{name: getattr(plane, name) for name in names})
            for plane in parsed.planes
        ],
        points=Points(
            **{
                field.name: getattr(metadata, field.name)
                for field in dataclasses.fields(Points)
            }
        ),
        slip_rate=SlipRate(offsets=parsed.slipt1.row_ptr, values=parsed.slipt1.data),
        comments=parsed.comments,
    )


def _core_planes(planes: Sequence[Plane]) -> list[_core.PySrfPlane]:
    return [_core.PySrfPlane(**dataclasses.asdict(plane)) for plane in planes]


def _core_points(points: Points) -> _core.PySrfMetadata:
    return _core.PySrfMetadata(
        **{
            field.name: getattr(points, field.name)
            for field in dataclasses.fields(Points)
        }
    )


def _core_slip_rate(slip_rate: SlipRate) -> _core.PyCsrMatrix:
    return _core.PyCsrMatrix(
        row_ptr=slip_rate.offsets, data=slip_rate.values, indices=slip_rate.columns
    )


def _parse[T](source: FilePath | Buffer, parse: Callable[[Buffer], T]) -> T:
    """Run a parser over an SRF's bytes, memory-mapping a file rather than reading it.

    Raises
    ------
    SrfError
        If the parser refuses the bytes, or the file is empty.
    OSError
        If opening or mapping the file fails.
    """
    if not isinstance(source, (str, Path)):
        return parse(source)
    path = Path(source)
    try:
        if path.stat().st_size == 0:
            raise SrfError("The file is empty")
        with (
            path.open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            if hasattr(mapped, "madvise"):
                mapped.madvise(mmap.MADV_SEQUENTIAL)
            return parse(mapped)
    except SrfError as error:
        raise SrfError(f"{path}: {error}") from error


def read(source: FilePath | Buffer) -> Srf:
    """Read an SRF.

    Parameters
    ----------
    source : str, Path or Buffer
        A path to the file, or its bytes. A file is memory-mapped rather than read
        into memory first.

    Returns
    -------
    Srf
        The file's comments, planes, points and pulses.

    Raises
    ------
    SrfError
        If the source isn't an SRF this reader supports.
    OSError
        If opening or mapping the file fails.
    """
    return _from_core(_parse(source, _core.parse_srf))


def read_comments(source: FilePath | Buffer) -> tuple[str, ...]:
    """Read an SRF's header comments, and nothing after them.

    The read stops at the first line after the version line that isn't a comment,
    so it touches a few kilobytes of a file of any size. It doesn't check the rest
    of the file.

    Parameters
    ----------
    source : str, Path or Buffer
        A path to the file, or its bytes.

    Returns
    -------
    tuple of str
        Each comment's text without its leading ``# ``, as :attr:`Srf.comments` has
        them.

    Raises
    ------
    SrfError
        If the source doesn't start with an SRF version line.
    OSError
        If opening or mapping the file fails.
    """
    return tuple(_parse(source, _core.parse_comments))


def write(path: FilePath, srf: Srf) -> None:
    """Write a whole SRF, at the version its points make.

    Parameters
    ----------
    path : str or Path
        Where to write the file.
    srf : Srf
        What to write.

    Raises
    ------
    OSError
        If creating or writing the file fails.
    """
    _core.write_srf(
        _core.PySrfFile(
            _core_planes(srf.planes),
            _core_points(srf.points),
            _core_slip_rate(srf.slip_rate),
            list(srf.comments),
        ),
        str(path),
    )


class Writer:
    """Write a version 2.0 SRF a run of points at a time.

    The header goes out when the writer opens. Each :meth:`write` adds the next
    points in file order, and a run may start or end anywhere in a plane.
    :meth:`close` checks that the writer wrote every point the planes declare. Used as a
    context manager, it closes on a clean exit and on an exception leaves the file
    unfinished, so that it doesn't read as an SRF.

    Parameters
    ----------
    path : str or Path
        Where to write the file.
    planes : sequence of Plane
        Every plane's header.
    comments : sequence of str, optional
        The header's comment lines, each one's text without its leading ``# ``.

    Raises
    ------
    SrfError
        If a comment can't be one line of the header. The file isn't created.
    OSError
        If creating the file or writing the header fails.
    """

    def __init__(
        self, path: FilePath, planes: Sequence[Plane], comments: Sequence[str] = ()
    ) -> None:
        """Open the file and write the header."""
        for index, comment in enumerate(comments):
            check_comment(index, comment)
        self._writer = _core.SrfWriter(str(path), _core_planes(planes), list(comments))

    def write(self, points: Points, slip_rate: SlipRate) -> None:
        """Write the next points and their pulses.

        Parameters
        ----------
        points : Points
            The points, with ``vs`` and ``density``.
        slip_rate : SlipRate
            One pulse per point.

        Raises
        ------
        SrfError
            If the points have no ``vs`` and ``density``, the pulses and the points
            differ in number, or the points run past what the planes declare.
        ValueError
            If the writer has closed.
        OSError
            If writing fails.
        """
        self._writer.write(_core_points(points), _core_slip_rate(slip_rate))

    def close(self) -> None:
        """Finish the file. Closing a closed writer does nothing.

        Raises
        ------
        SrfError
            If the writer wrote fewer points than the planes declare.
        OSError
            If writing fails.
        """
        self._writer.close()

    def __enter__(self) -> Self:
        """Use the writer as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the writer, or abandon the file if the block raised."""
        self._writer.__exit__(exc_type, exc_value, traceback)


__all__ = ["Writer", "read", "read_comments", "write"]
