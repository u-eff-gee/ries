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

from ries.constituents.element import natural_elements


def test_element():
    assert natural_elements["Pb"].Z == 82
    assert len(natural_elements["Pb"].isotopes) == 4
    assert natural_elements["Pb"].amu == (207.216908063)
    assert natural_elements["Pb"].density == 11.35

    # Test some special cases.

    # In the NIST data file, the carbon isotopes 12C, 13C, and 14C are listed.
    # No abundance is given for 14C, since it is radioactive.
    # Test that the 14C abundance is initialized to zero.
    assert natural_elements["C"].amu == 12.010735896735248

    # Beryllium has only a single stable isotope, 9Be.
    # Its abundance is given as "1" without an error.
    # Check that this is parsed correctly.
    assert natural_elements["Be"].abundances["9Be"] == 1.0

    # Technetium has no stable isotopes, but three are listed in the NIST data file.
    # By default, their abundances should be initialized to 1/3.
    assert natural_elements["Tc"].abundances["97Tc"] == 1.0 / 3.0
