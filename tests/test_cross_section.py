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

import pytest

import numpy as np
from scipy.constants import physical_constants

from ries.cross_section import (
    CrossSection,
    ConstantCrossSection,
)
from ries.constituents.element import natural_elements
from ries.nonresonant.xrmac import load_xrmac_data, xrmac_fm2_per_atom
from ries.resonance.voigt import Voigt
from ries.resonance.debye_model import (
    effective_temperature_debye_approximation,
    load_room_temperature_T_D_data,
    room_temperature_T_D,
)

from .boron import B11


class TestCrossSection:
    def test_abstract(self):
        # The abstract CrossSection class requires the users to implement the functions
        # CrossSection.__call__() and CrossSection.equidistant_probability_grid().
        cross_section = CrossSection()

        with pytest.raises(NotImplementedError):
            cross_section(1.0)
        with pytest.raises(NotImplementedError):
            cross_section.equidistant_probability_grid([0.0, 1.0], 10)

    def test_algebra(self):

        cm_to_fm = 1e13
        kg_to_g = 1e3
        with pytest.warns(UserWarning):
            load_xrmac_data()
        with pytest.warns(UserWarning):
            load_room_temperature_T_D_data()
        # Create array of all 11B ground-state transitions.
        ground_state_resonances = [
            Voigt(
                B11.ground_state,
                B11.excited_states[excited_state],
                B11.amu,
                effective_temperature_debye_approximation(
                    293.0, room_temperature_T_D["B"]
                ),
            )
            for excited_state in B11.excited_states
        ]

        # Test CrossSection.__add__()
        photoabsorption_cross_section = (
            ground_state_resonances[0] + ground_state_resonances[1]
        )
        single_cross_section = ground_state_resonances[0](
            ground_state_resonances[0].resonance_energy
        )
        sum_cross_section = photoabsorption_cross_section(
            ground_state_resonances[0].resonance_energy
        )
        assert np.isclose(single_cross_section, sum_cross_section, rtol=1e-5)

        # Test CrossSection.__mul__() and CrossSection.__rmul__()
        photoabsorption_cross_section = (
            ground_state_resonances[0] + 0.0 * ground_state_resonances[1]
        )
        single_cross_section = ground_state_resonances[0](
            ground_state_resonances[0].resonance_energy
        )
        sum_cross_section = photoabsorption_cross_section(
            ground_state_resonances[0].resonance_energy
        )
        assert np.isclose(single_cross_section, sum_cross_section, rtol=1e-5)

        # Test CrossSection.__add__() using the convenient sum() function.
        # CrossSection.radd() are CrossSectionWeightedSum().add() are tested implicitly,
        # because sum() adds all terms to an integer value of 0, and because any sum of cross
        # sections yields a CrossSectionWeightedSum object.
        photoabsorption_cross_section = sum(ground_state_resonances)

        for resonance in ground_state_resonances:
            energy = resonance.equidistant_probability_grid(0.9, 10)
            single_cross_section = resonance(energy)
            sum_cross_section = photoabsorption_cross_section(energy)

            assert np.allclose(single_cross_section, sum_cross_section, rtol=1e-3)

        # Test CrossSectionWeightedSum.__add__() and CrossSectionWeightedSum.__mul__().
        photoabsorption_cross_section = (
            photoabsorption_cross_section + 0.5 * photoabsorption_cross_section
        )

        for resonance in ground_state_resonances:
            energy = resonance.equidistant_probability_grid(0.9, 10)
            single_cross_section = resonance(energy)
            sum_cross_section = photoabsorption_cross_section(energy)

            assert np.allclose(1.5 * single_cross_section, sum_cross_section, rtol=1e-3)

        # Test CrossSectionWeightedSum.__radd__().
        photoabsorption_cross_section = 0.0 + (
            ground_state_resonances[0] + ground_state_resonances[1]
        )
        single_cross_section = ground_state_resonances[0](
            ground_state_resonances[0].resonance_energy
        )
        sum_cross_section = photoabsorption_cross_section(
            ground_state_resonances[0].resonance_energy
        )
        assert np.isclose(single_cross_section, sum_cross_section, rtol=1e-5)

        # Test additivity with a derived class for a nonresonant cross section.
        photoabsorption_cross_section = (
            photoabsorption_cross_section + xrmac_fm2_per_atom[5]
        )

        photoabsorption_cross_section_value = (
            5.9e-2  # Rounded NIST value
            * cm_to_fm**2
            * natural_elements[5].amu()
            * physical_constants["atomic mass constant"][0]
            * kg_to_g
        )
        # Use a generous tolerance to be able to cover the Compton-scattering default values
        # and the more realistic NIST data.
        # See also test/test_xrmac.py.
        assert np.isclose(
            photoabsorption_cross_section(1.0),
            photoabsorption_cross_section_value,
            rtol=1e-1,
        )

    def test_grid(self):
        with pytest.warns(UserWarning):
            load_xrmac_data()
        assert np.allclose(
            xrmac_fm2_per_atom[5].equidistant_energy_grid((0.0, 1.0), 3),
            np.array([0.0, 0.5, 1.0]),
            rtol=1e-3,
        )

        cross_section = ConstantCrossSection(1.0)
        assert np.allclose(
            cross_section.equidistant_probability_grid((0.0, 1.0), 3),
            np.array([0.0, 0.5, 1.0]),
            rtol=1e-6,
        )

        cross_section = cross_section + cross_section
        assert np.allclose(
            cross_section.equidistant_probability_grid((0.0, 1.0), 3),
            np.array([0.0, 0.5, 1.0]),
            rtol=1e-6,
        )

        assert np.allclose(
            cross_section.equidistant_energy_grid((0.0, 1.0), 3),
            np.array([0.0, 0.5, 1.0]),
            rtol=1e-6,
        )
