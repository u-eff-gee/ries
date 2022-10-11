import numpy as np

from ries.constituents.iupac_isotopic_compositions.isotopic_compositions import (
    isotopic_compositions,
)


def test_isotopic_compositions():
    # Check whether all isotopic abundances are correctly normalized.
    # This is a simple test to ensure that all data have been transferred correctly.
    for element in isotopic_compositions:
        norm = 0.0
        for A in isotopic_compositions[element]:
            norm += isotopic_compositions[element][A]
        assert np.isclose(
            norm, 1.0, rtol=1e-9
        )  # A relative tolerance of 1e-9 is more precise than any of the experimental values. Any discrepancy is due to numerical inaccuracies of python floats.
