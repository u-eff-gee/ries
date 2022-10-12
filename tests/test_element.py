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

from pathlib import Path

from ries.constituents.element import X_from_Z, Z_from_X, natural_elements
from ries.constituents.iupac_isotopic_compositions.isotopic_compositions import (
    isotopic_compositions,
)
from ries.constituents.ame_2020_mass_data_reader import AME2020MassDataReader


def test_element():
    ame2020_reader = AME2020MassDataReader(
        Path(__file__).parent.absolute()
        / "../ries/constituents/ame2020_masses/mass_1.mas20"
    )
    ame_masses = ame2020_reader.read_mass_data()

    assert natural_elements[82].Z == 82
    assert (
        len(natural_elements[Z_from_X["Pb"]].isotopes) == 4
    )  # Besides testing the correct number of isotopes, this line obtains the proton number of lead from its element symbol.
    assert natural_elements[82].amu() == (
        isotopic_compositions[82][204] * ame_masses[82][204]
        + isotopic_compositions[82][206] * ame_masses[82][206]
        + isotopic_compositions[82][207] * ame_masses[82][207]
        + isotopic_compositions[82][208] * ame_masses[82][208]
    )
    assert (
        natural_elements[82].isotopes[204].abundance == isotopic_compositions[82][204]
    )
    assert natural_elements[82].density == 11.35

    # Test the two limiting cases of the element-symbol dictionaries.
    assert Z_from_X["n"] == 0
    assert X_from_Z[118] == "Og"
    # Test consistency.
    assert Z_from_X[X_from_Z[50]] == 50
