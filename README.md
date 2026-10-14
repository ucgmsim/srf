# `srf`

Read and write Standard Rupture Format (SRF) files, versions 1.0 and 2.0. The
parser and writer are Rust. The Python API consists of frozen dataclasses over NumPy
arrays, and NumPy is its only dependency.

## Install

```sh
pip install ucgmsim-srf
```

Building from source needs a Rust toolchain.

## Use

```python
import srf

rupture = srf.read("rupture.srf")
rupture.planes  # tuple of Plane headers
rupture.points.slip1  # slip at each point, in cm
rupture.slip_rate.pulse(0)  # point 0's slip rate in cm/s, from its onset
rupture.rise  # rise times in seconds

srf.write("copy.srf", rupture)
```

A large version 2.0 file can go out a run of points at a time:

```python
with srf.Writer("rupture.srf", planes) as writer:
    for points, slip_rate in runs:
        writer.write(points, slip_rate)
```

Values keep the file's own units, so slip and rock are in centimetre-gram-second
units. The records check their arrays when built and refuse an inconsistent SRF
with `srf.SrfError`, a `ValueError`. The reader and writers handle the `slip1`
component alone.

## Development

```sh
uv sync --all-extras --dev
uvx lefthook install    # the CI checks, as git hooks
vale sync               # the prose styles the pre-push hook needs
uv run pytest tests
cargo test
```
