# This file is part of ries.

# ries is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# ries is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with ries.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pytest
from scipy.constants import physical_constants

from ries.constituents.element import natural_elements
from ries.nonresonant.xrmac import load_xrmac_data, xrmac_cm2_per_g, xrmac_fm2_per_atom


def test_xrmac():
    with pytest.warns(UserWarning):
        load_xrmac_data()
    xrmac_Pb_1MeV = 7.1e-2  # Rounded NIST value
    xrmac_test = (
        xrmac_Pb_1MeV
        * 1e26
        * natural_elements["Pb"].amu
        * physical_constants["atomic mass constant"][0]
        * 1e3
    )
    # Use a generous tolerance to be able to cover the Compton-scattering default values
    # and the more realistic NIST data.
    # Even at 1 MeV, where Compton scattering contributes most to the attenuation by lead,
    # the Compton approximation is about 30% off.
    assert np.isclose(xrmac_cm2_per_g["Pb"](1.0), xrmac_Pb_1MeV, rtol=3e-1)
    assert np.isclose(xrmac_fm2_per_atom["Pb"](1.0), xrmac_test, rtol=3e-1)
