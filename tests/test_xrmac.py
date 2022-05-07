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
from scipy.constants import physical_constants

from ries.constituents.element import natural_elements
from ries.nonresonant.xrmac import xrmac_cm2_per_g, xrmac_fm2_per_atom


def test_xrmac():
    xrmac_Pb_1MeV = 7.1e-2  # Rounded value
    xrmac_test = (
        xrmac_Pb_1MeV
        * 1e26
        * natural_elements["Pb"].amu
        * physical_constants["atomic mass constant"][0]
        * 1e3
    )
    assert np.isclose(xrmac_cm2_per_g["Pb"](1.0), xrmac_Pb_1MeV, 1e-2)
    assert np.isclose(xrmac_fm2_per_atom["Pb"](1.0), xrmac_test, 1e-2)
