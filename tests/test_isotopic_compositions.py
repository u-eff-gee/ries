import numpy as np

from ries.constituents.iupac_isotopic_compositions.isotopic_compositions import (
    isotopic_compositions,
)


def test_isotopic_compositions():
    # Check whether all isotopic abundances are correctly normalized.
    # This is a simple test to ensure that all data have been transferred correctly.
    for Z in isotopic_compositions:
        if len(isotopic_compositions[Z]):
            norm = 0.0
            for A in isotopic_compositions[Z]:
                norm += isotopic_compositions[Z][A]
            assert np.isclose(
                norm, 1.0, rtol=1e-9
            )  # A relative tolerance of 1e-9 is more precise than any of the experimental values. Any discrepancy is due to numerical inaccuracies of python floats.

    # Count the number of quasi-monoisotopic [1] elements.
    # The result should be :
    #
    #  1. Be
    #  2. F
    #  3. Na
    #  4. Al
    #  5. P
    #  6. Sc
    #  7. V
    #  8. Mn
    #  9. Co
    # 10. As
    # 11. Nb
    # 12. Rh
    # 13. I
    # 14. Cs
    # 15. Pr
    # 16. Tb
    # 17. Ho
    # 18. Tm
    # 19. Au
    # 20. Bi
    # 21. Pa
    #
    # [1] Some elements occur naturally with a significant abundance of a radioactive isotope due
    # to various reasons, mostly extremely long half lives.
    n_monoisotopic = 0
    for Z in isotopic_compositions:
        if len(isotopic_compositions[Z]) == 1:
            n_monoisotopic += 1
    assert n_monoisotopic == 21
