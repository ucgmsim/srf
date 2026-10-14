"""What the records accept and refuse, and that whatever they accept round trips."""

import dataclasses

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import srf

FLOATS = st.floats(width=32, allow_nan=False, allow_infinity=False)


def plane(nstk: int = 1, ndip: int = 1) -> srf.Plane:
    return srf.Plane(172.0, -43.0, nstk, ndip, 1.0, 1.0, 45.0, 80.0, 0.0, 0.0, 0.5)


def points(count: int, *, rock: bool = True) -> srf.Points:
    columns = {name: np.arange(count, dtype=np.float32) for name in srf.POINT_COLUMNS}
    if rock:
        columns["vs"] = np.full(count, 3.5e5, dtype=np.float32)
        columns["density"] = np.full(count, 2.7, dtype=np.float32)
    return srf.Points(**columns)


def test_a_plane_coerces_its_fields():
    # NumPy scalars and integral floats, as a caller reading a table would pass.
    header = srf.Plane(
        np.float32(172.5),  # ty: ignore[invalid-argument-type]
        -43,
        np.int64(2),  # ty: ignore[invalid-argument-type]
        3.0,  # ty: ignore[invalid-argument-type]
        1,
        1,
        45,
        80,
        0,
        0,
        0,
    )
    assert type(header.elon) is float
    assert type(header.nstk) is int
    assert header.ndip == 3
    assert header.points == 6


@pytest.mark.parametrize("nstk", [-1, 1.5, True])
def test_a_plane_refuses_a_bad_count(nstk):
    with pytest.raises(srf.SrfError, match="nstk must be a non-negative integer"):
        plane(nstk=nstk)


def test_points_are_read_only_single_precision_views():
    lon = np.zeros(3, dtype=np.float32)
    columns = points(3)
    assert columns.lon.dtype == np.float32
    assert not columns.lon.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        columns.lon[0] = 1.0
    shared = dataclasses.replace(columns, lon=lon)
    assert np.shares_memory(shared.lon, lon)
    assert lon.flags.writeable


def test_points_refuse_ragged_columns():
    with pytest.raises(srf.SrfError, match="differ in length"):
        dataclasses.replace(points(3), rake=np.zeros(2))


def test_points_refuse_a_column_of_the_wrong_shape():
    with pytest.raises(srf.SrfError, match="rake must be one-dimensional"):
        dataclasses.replace(points(3), rake=np.zeros((3, 1)))


def test_points_need_both_or_neither_of_the_rock():
    with pytest.raises(srf.SrfError, match="vs and density"):
        dataclasses.replace(points(3), vs=None)
    assert points(3, rock=False).version == 1
    assert points(3).version == 2


@pytest.mark.parametrize(
    ("offsets", "values"),
    [([], []), ([1, 2], [0.0, 0.0]), ([0, 2, 1], [0.0, 0.0]), ([0, 1], [0.0, 0.0])],
)
def test_a_slip_rate_refuses_bad_offsets(offsets, values):
    with pytest.raises(srf.SrfError, match="offsets must start at 0"):
        srf.SlipRate(offsets=np.array(offsets), values=np.array(values))


def test_a_slip_rate_refuses_offsets_that_are_not_counts():
    with pytest.raises(srf.SrfError, match="must not be negative"):
        srf.SlipRate(offsets=np.array([0, -1]), values=np.array([]))
    with pytest.raises(srf.SrfError, match="must hold integers"):
        srf.SlipRate(offsets=np.array([0.0, 1.0]), values=np.array([1.0]))


@pytest.mark.parametrize("columns", [[0, 0], [1, 0], [0]])
def test_a_slip_rate_refuses_bad_columns(columns):
    with pytest.raises(srf.SrfError, match="columns"):
        srf.SlipRate(
            offsets=np.array([0, 2]),
            values=np.array([1.0, 2.0]),
            columns=np.array(columns),
        )


def test_columns_restart_on_each_row():
    rate = srf.SlipRate(
        offsets=np.array([0, 2, 2, 3]),
        values=np.array([1, 2, 3]),
        columns=np.array([1, 3, 0]),
    )
    assert rate.samples.tolist() == [4, 0, 1]
    assert rate.pulse(0).tolist() == [0.0, 1.0, 0.0, 2.0]
    assert rate.pulse(1).tolist() == []
    assert rate[1:].offsets.tolist() == [0, 0, 1]
    tail = rate[1:]
    assert tail.columns is not None
    assert tail.columns.tolist() == [0]


def test_rows_are_taken_as_a_run():
    rate = srf.SlipRate.from_pulses([[1.0], [2.0, 3.0], []])
    assert rate[1:].values.tolist() == [2.0, 3.0]
    assert len(rate[2:1]) == 0
    with pytest.raises(ValueError, match="contiguous run"):
        rate[::2]


def test_an_empty_slip_rate():
    rate = srf.SlipRate.from_pulses([])
    assert len(rate) == 0
    assert rate.samples.tolist() == []


def test_the_planes_points_and_pulses_must_agree():
    with pytest.raises(srf.SrfError, match="declare 2 points"):
        srf.Srf([plane(2, 1)], points(3), srf.SlipRate.from_pulses([[]] * 3))
    with pytest.raises(srf.SrfError, match="2 pulses"):
        srf.Srf([plane(3, 1)], points(3), srf.SlipRate.from_pulses([[]] * 2))


@st.composite
def srfs(draw: st.DrawFn) -> srf.Srf:
    counts = st.integers(min_value=0, max_value=3)
    planes = [
        srf.Plane(
            *draw(st.tuples(FLOATS, FLOATS)),
            *draw(st.tuples(counts, counts)),
            *draw(st.tuples(*[FLOATS] * 7)),
        )
        for _ in range(draw(st.integers(min_value=1, max_value=3)))
    ]
    count = sum(header.points for header in planes)
    column = hnp.arrays(np.float32, count, elements=FLOATS)
    rock = draw(st.booleans())
    names = [*srf.POINT_COLUMNS, *(["vs", "density"] if rock else [])]
    columns = {name: draw(column) for name in names}
    pulses = [
        draw(hnp.arrays(np.float32, st.integers(0, 5), elements=FLOATS))
        for _ in range(count)
    ]
    return srf.Srf(planes, srf.Points(**columns), srf.SlipRate.from_pulses(pulses))


@settings(max_examples=50, deadline=None)
@given(srfs())
def test_whatever_the_records_accept_round_trips(tmp_path_factory, srf_file):
    path = tmp_path_factory.mktemp("roundtrip") / "srf.srf"
    srf.write(path, srf_file)
    reread = srf.read(path)
    assert reread.planes == srf_file.planes
    assert reread.version == srf_file.version
    for field in dataclasses.fields(srf.Points):
        expected = getattr(srf_file.points, field.name)
        if expected is not None:
            np.testing.assert_array_equal(getattr(reread.points, field.name), expected)
    np.testing.assert_array_equal(reread.slip_rate.offsets, srf_file.slip_rate.offsets)
    np.testing.assert_array_equal(reread.slip_rate.values, srf_file.slip_rate.values)
