"""Writing SRFs: whole files, and a stream of points written a run at a time."""

import dataclasses
import itertools

import numpy as np
import pytest

import srf
from tests.conftest import SRF_DIR


def assert_same(left: srf.Srf, right: srf.Srf) -> None:
    assert left.planes == right.planes
    for field in dataclasses.fields(srf.Points):
        a, b = getattr(left.points, field.name), getattr(right.points, field.name)
        if a is None or b is None:
            assert a is b, field.name
        else:
            np.testing.assert_array_equal(a, b, err_msg=field.name)
    assert len(left.slip_rate) == len(right.slip_rate)
    for k in range(len(left.slip_rate)):
        np.testing.assert_array_equal(left.slip_rate.pulse(k), right.slip_rate.pulse(k))


def with_rock(srf_file: srf.Srf) -> srf.Srf:
    """A version 1.0 SRF made version 2.0, with a made-up rock at each point."""
    count = len(srf_file.points)
    points = dataclasses.replace(
        srf_file.points,
        vs=np.linspace(2.0e5, 4.0e5, count),
        density=np.linspace(2.4, 2.8, count),
    )
    return dataclasses.replace(srf_file, points=points)


@pytest.fixture(scope="module")
def darfield_v2(darfield, tmp_path_factory):
    """The Darfield planes as version 2.0, and that SRF written whole."""
    path = tmp_path_factory.mktemp("srf") / "darfield.srf"
    srf_file = with_rock(darfield)
    srf.write(path, srf_file)
    return path, srf_file


def test_version_1_round_trips(christchurch, tmp_path):
    path = tmp_path / "christchurch.srf"
    srf.write(path, christchurch)
    assert path.read_text().startswith("1.0\n")
    assert_same(srf.read(path), christchurch)


def test_version_2_round_trips(tmp_path):
    original = srf.read(SRF_DIR / "point_source_v2.srf")
    path = tmp_path / "point_source_v2.srf"
    srf.write(str(path), original)
    assert_same(srf.read(path), original)


def test_version_2_writes_a_points_block_per_plane(darfield_v2):
    path, srf_file = darfield_v2
    text = path.read_text()
    assert text.startswith("2.0\n")
    assert text.count("POINTS ") == len(srf_file.planes) == 7
    assert_same(srf.read(path), srf_file)


def test_skipped_columns_are_written_as_zeros(tmp_path):
    """A row without its zeros stored, as `scipy.sparse` leaves one, keeps its shape.

    source_modelling's writer dropped the columns, so such a row came back shorter,
    with every later sample moved earlier.
    """
    original = srf.read(SRF_DIR / "point_source_v2.srf")
    sparse = srf.SlipRate(
        offsets=np.array([0, 1, 3]),
        values=np.array([5.0, 1.0, 6.0]),
        columns=np.array([1, 0, 2]),
    )
    path = tmp_path / "sparse.srf"
    srf.write(path, dataclasses.replace(original, slip_rate=sparse))
    reread = srf.read(path).slip_rate
    assert reread.pulse(0).tolist() == [0.0, 5.0]
    assert reread.pulse(1).tolist() == [1.0, 0.0, 6.0]


def test_a_stream_in_any_runs_writes_the_same_file(darfield_v2, tmp_path):
    path, srf_file = darfield_v2
    count = len(srf_file.points)
    # Runs of 997 points mostly start and end mid-plane, and some span a boundary.
    edges = [*range(0, count, 997), count]
    streamed = tmp_path / "streamed.srf"
    with srf.Writer(streamed, srf_file.planes) as writer:
        for start, stop in itertools.pairwise(edges):
            run = slice(start, stop)
            writer.write(srf_file.points[run], srf_file.slip_rate[run])
    assert streamed.read_bytes() == path.read_bytes()


def test_a_stream_refuses_points_the_planes_do_not_declare(darfield_v2, tmp_path):
    _, srf_file = darfield_v2
    writer = srf.Writer(tmp_path / "long.srf", srf_file.planes[:1])
    with pytest.raises(srf.SrfError, match="exceed the"):
        writer.write(srf_file.points, srf_file.slip_rate)


def test_a_stream_refuses_to_close_short(darfield_v2, tmp_path):
    _, srf_file = darfield_v2
    writer = srf.Writer(tmp_path / "short.srf", srf_file.planes)
    writer.write(srf_file.points[:3], srf_file.slip_rate[:3])
    with pytest.raises(srf.SrfError, match=r"Points written \(3\)"):
        writer.close()
    with pytest.raises(ValueError, match="is closed"):
        writer.write(srf_file.points[3:4], srf_file.slip_rate[3:4])
    writer.close()


def test_a_stream_needs_one_pulse_per_point(darfield_v2, tmp_path):
    _, srf_file = darfield_v2
    writer = srf.Writer(tmp_path / "rows.srf", srf_file.planes)
    with pytest.raises(srf.SrfError, match="slipt1 has 2 rows but the metadata has 3"):
        writer.write(srf_file.points[:3], srf_file.slip_rate[:2])
    writer.write(srf_file.points, srf_file.slip_rate)
    writer.close()


def test_a_stream_needs_the_rock(christchurch, tmp_path):
    writer = srf.Writer(tmp_path / "v1.srf", christchurch.planes)
    with pytest.raises(srf.SrfError, match="vs and density are required"):
        writer.write(christchurch.points, christchurch.slip_rate)


def test_an_exception_leaves_the_stream_unfinished(darfield_v2, tmp_path):
    _, srf_file = darfield_v2
    path = tmp_path / "interrupted.srf"
    with pytest.raises(RuntimeError), srf.Writer(path, srf_file.planes) as writer:
        writer.write(srf_file.points[:3], srf_file.slip_rate[:3])
        raise RuntimeError
    # Closing the file as whole would have raised for the missing points instead.
    with pytest.raises(srf.SrfError):
        srf.read(path)


def test_an_unwritable_path_is_an_os_error(christchurch, tmp_path):
    with pytest.raises(OSError):
        srf.write(tmp_path / "missing" / "christchurch.srf", christchurch)
    with pytest.raises(OSError):
        srf.Writer(tmp_path / "missing" / "christchurch.srf", christchurch.planes)
