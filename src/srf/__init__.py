"""Read and write Standard Rupture Format (SRF) files."""

from srf.io import Writer, read, read_comments, write
from srf.types import (
    MAX_COMMENT_BYTES,
    POINT_COLUMNS,
    Plane,
    Points,
    SlipRate,
    Srf,
    SrfError,
)

__all__ = [
    "MAX_COMMENT_BYTES",
    "POINT_COLUMNS",
    "Plane",
    "Points",
    "SlipRate",
    "Srf",
    "SrfError",
    "Writer",
    "read",
    "read_comments",
    "write",
]
