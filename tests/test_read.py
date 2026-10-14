"""Reading real and hand-written SRFs, adapted from source_modelling's tests."""

import dataclasses

import numpy as np
import pytest

import srf
from tests.conftest import SRF_DIR, read_fixture


def point(srf_file: srf.Srf, index: int) -> dict[str, float]:
    points = srf_file.points
    values = {
        field.name: float(column[index])
        for field in dataclasses.fields(points)
        if (column := getattr(points, field.name)) is not None
    }
    values["rise"] = float(srf_file.rise[index])
    return values


def test_christchurch(christchurch):
    assert christchurch.version == 1
    assert len(christchurch.planes) == 1
    assert len(christchurch.points) == 14400
    assert dataclasses.asdict(christchurch.planes[0]) == pytest.approx(
        {
            "elon": 172.6966,
            "elat": -43.5446,
            "nstk": 160,
            "ndip": 90,
            "len": 16.00,
            "wid": 9.00,
            "stk": 59,
            "dip": 69,
            "dtop": 0.63,
            "shyp": -2.00,
            "dhyp": 6.00,
        }
    )
    # The local strike and dip match the header's.
    assert (christchurch.points.dip == 69).all()
    assert (christchurch.points.stk == 59).all()
    assert christchurch.points.tinit.min() == 0.0
    assert (christchurch.points.dt == np.float32(2.5e-02)).all()
    assert len(christchurch.slip_rate) == len(christchurch.points)

    assert point(christchurch, 0) == pytest.approx(
        {
            "lon": 172.6127,
            "lat": -43.5821,
            "dep": 0.6767,
            "stk": 59,
            "dip": 69,
            "area": 1.0e08,
            "tinit": 5.7029,
            "dt": 2.5e-02,
            "rake": 102,
            "slip1": 17.49,
            "rise": 0.3,
        }
    )
    np.testing.assert_array_equal(
        christchurch.slip_rate.pulse(0)[:12],
        np.array(
            [
                0.00000e00,
                2.07568e02,
                2.42313e02,
                5.90245e01,
                4.89368e01,
                4.26333e01,
                3.50411e01,
                2.68253e01,
                1.87057e01,
                1.13937e01,
                5.52983e00,
                1.62786e00,
            ],
            dtype=np.float32,
        ),
    )
    # The last pulse too, since the parser reads the file as one stream.
    np.testing.assert_array_equal(
        christchurch.slip_rate.pulse(len(christchurch.points) - 1)[:7],
        np.array(
            [
                0.00000e00,
                3.97588e02,
                7.47954e01,
                6.18204e01,
                4.37692e01,
                2.48125e01,
                9.33055e00,
            ],
            dtype=np.float32,
        ),
    )


def test_darfield(darfield):
    expected = [
        (172.133408, -43.550999, 50, 90, 10.0, 18.0, 40, 75, 1.0, 1.0, 10.0),
        (172.003906, -43.568298, 60, 90, 12.0, 18.0, 121, 105, 0.0, 6.0, 6.0),
        (172.194901, -43.588299, 100, 90, 20.0, 18.0, 87, 85, 0.0, -10.0, 6.0),
        (172.379898, -43.571301, 70, 90, 14.0, 18.0, 87, 85, 0.0, -7.0, 6.0),
        (171.944305, -43.578400, 35, 90, 7.0, 18.0, 216, 50, 0.0, 3.5, 6.0),
        (172.309799, -43.549900, 55, 90, 11.0, 18.0, 40, 80, 0.0, -5.5, 6.0),
        (172.182205, -43.508999, 40, 90, 8.0, 18.0, 150, 54, 0.0, 4.0, 6.0),
    ]
    assert [dataclasses.astuple(plane) for plane in darfield.planes] == [
        pytest.approx(header) for header in expected
    ]
    for plane, points in zip(darfield.planes, darfield.plane_points, strict=True):
        assert points.stop - points.start == plane.nstk * plane.ndip
        assert (darfield.points.dip[points] == plane.dip).all()
        assert (darfield.points.stk[points] == plane.stk).all()
    assert darfield.plane_points[-1].stop == len(darfield.points)


@pytest.mark.parametrize(
    "name",
    ["empty.srf", "bad_int.srf", "bad_float.srf", "no_points.srf", "bad_plane.srf"],
)
def test_junk_is_refused(name):
    with pytest.raises(srf.SrfError, match=name):
        srf.read(SRF_DIR / name)


def test_junk_bytes_are_refused():
    with pytest.raises(srf.SrfError, match="empty"):
        srf.read(b"")
    with pytest.raises(srf.SrfError, match="unknown SRF version"):
        srf.read(b"garbage\n")


def test_a_hand_written_version_2():
    """Every value in the two-point file, checked by eye against it."""
    srf_v2 = srf.read(SRF_DIR / "point_source_v2.srf")
    assert srf_v2.version == 2
    assert len(srf_v2.points) == 2
    assert srf_v2.points.vs is not None
    assert srf_v2.points.density is not None
    assert srf_v2.points.vs.tolist() == pytest.approx([3.5e5, 3.6e5])
    assert srf_v2.points.density.tolist() == pytest.approx([2.7, 2.8])
    assert point(srf_v2, 0) == pytest.approx(
        {
            "lon": 172.0,
            "lat": -43.0,
            "dep": 0.5,
            "stk": 45,
            "dip": 80,
            "area": 1.0e10,
            "tinit": 0.0,
            "dt": 0.1,
            "vs": 3.5e5,
            "density": 2.7,
            "rake": 90,
            "slip1": 10.0,
            "rise": 0.2,
        }
    )
    # Each pulse counts from its own onset, so both start at column zero.
    assert srf_v2.slip_rate.offsets.tolist() == [0, 2, 4]
    assert srf_v2.slip_rate.values.tolist() == [0.0, 5.0, 0.0, 6.0]
    assert srf_v2.slip_rate.columns is None


def test_a_genslip_version_2():
    """A real genslip file, with comment lines after the version and 2601 points.

    The values come from the first, middle and last points. The parser
    reads the file as one stream, so a correct last point means it stayed aligned
    through every block before it.
    """
    real = read_fixture("test_v2.srf.gz")
    assert real.version == 2
    # genslip writes the command that made the file, and its seeds, as comments.
    assert real.comments[0].startswith("Command: /home/jake/src/EMOD3D/tools/genslip")
    assert real.comments[1:4] == (
        "",
        "starting_seed= 7252193",
        "ending_seed= 1320619617",
    )
    assert dataclasses.asdict(real.planes[0]) == pytest.approx(
        {
            "elon": 176.514603,
            "elat": -38.006092,
            "nstk": 51,
            "ndip": 51,
            "len": 5.0699,
            "wid": 5.0699,
            "stk": 240,
            "dip": 88,
            "dtop": 0.0,
            "shyp": 0.0,
            "dhyp": 2.5350,
        }
    )
    assert len(real.points) == 2601
    common = {"stk": 240, "dip": 88, "area": 9.88234e07, "dt": 5.0e-03}
    assert point(real, 0) == pytest.approx(
        common
        | {
            "lon": 176.539108,
            "lat": -37.994919,
            "dep": 4.96747e-02,
            "tinit": 5.815377,
            "vs": 3.8e04,
            "density": 1.81,
            "rake": -16,
            "slip1": 94.2758,
            "rise": 64 * 5.0e-03,
        }
    )
    assert point(real, 1300) == pytest.approx(
        common
        | {
            "lon": 176.514099,
            "lat": -38.005402,
            "dep": 2.53341,
            "tinit": 9.254894e-02,
            "vs": 2.28e05,
            "density": 2.40,
            "rake": -15,
            "slip1": 49.4048,
            "rise": 59 * 5.0e-03,
        }
    )
    assert point(real, 2600) == pytest.approx(
        common
        | {
            "lon": 176.489090,
            "lat": -38.015869,
            "dep": 5.01714,
            "tinit": 2.567641,
            "vs": 3.6e05,
            "density": 2.72,
            "rake": -7,
            "slip1": 15.8105,
            "rise": 3 * 5.0e-03,
        }
    )
    assert real.slip_rate.samples[0] == 64
    assert real.slip_rate.pulse(0)[:3] == pytest.approx([0.0, 9.69786, 21.3934])
    assert real.slip_rate.samples[-1] == 3
    assert real.slip_rate.pulse(2600) == pytest.approx([0.0, 3.16209e03, 0.0])


def test_version_1_has_no_rock(christchurch):
    assert christchurch.points.vs is None
    assert christchurch.points.density is None


def test_an_unknown_version_is_refused():
    text = (
        "9.0\n"
        "PLANE 1\n"
        "  172.0  -43.0   1   1   1.0   1.0\n"
        "  45   80   0.0   0.0   0.5\n"
        "POINTS 1\n"
        "  172.0  -43.0   0.5   45   80   1.0e10   0.0   0.1\n"
        "  90   10.0   1   0.0   0   0.0   0\n"
        "  0.0\n"
    )
    with pytest.raises(srf.SrfError, match="unknown SRF version: 9.0"):
        srf.read(text.encode())


def test_a_missing_file_is_an_os_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        srf.read(tmp_path / "missing.srf")
