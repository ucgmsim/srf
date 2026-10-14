import gzip
from pathlib import Path

import pytest

import srf

SRF_DIR = Path(__file__).parent / "srfs"


def read_fixture(name: str) -> srf.Srf:
    path = SRF_DIR / name
    if path.suffix == ".gz":
        return srf.read(gzip.decompress(path.read_bytes()))
    return srf.read(path)


@pytest.fixture(scope="session")
def christchurch() -> srf.Srf:
    """A version 1.0 SRF of the 2011 Christchurch earthquake: one plane."""
    return read_fixture("christchurch.srf.gz")


@pytest.fixture(scope="session")
def darfield() -> srf.Srf:
    """A version 1.0 SRF of the 2010 Darfield earthquake: seven planes."""
    return read_fixture("darfield.srf.gz")
