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

from ries.constituents.element import natural_elements
from ries.constituents.iupac_isotopic_compositions.isotopic_compositions import isotopic_compositions
from ries.constituents.natural_element_data import AME2020MassDataReader

def test_element():
    ame2020_reader = AME2020MassDataReader(
        Path(__file__).parent.absolute()
        / "../ries/constituents/ame2020_masses/mass_1.mas20"
    )
    ame_masses = ame2020_reader.read_mass_data()

    assert natural_elements["Pb"].Z == 82
    assert len(natural_elements["Pb"].isotopes) == 4
    assert natural_elements["Pb"].amu == (
        isotopic_compositions["Pb"][204]*ame_masses[82][204]
        +isotopic_compositions["Pb"][206]*ame_masses[82][206]
        +isotopic_compositions["Pb"][207]*ame_masses[82][207]
        +isotopic_compositions["Pb"][208]*ame_masses[82][208]
    )
    assert natural_elements["Pb"].abundances[204] == isotopic_compositions["Pb"][204]
    assert natural_elements["Pb"].density == 11.35
