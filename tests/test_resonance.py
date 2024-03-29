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

from ries.constituents.state import GroundState, State
from ries.resonance.breit_wigner import BreitWigner
from ries.resonance.resonance import Resonance

from .boron import B11


def test_resonance():
    cs = Resonance(
        initial_state=B11.ground_state, intermediate_state=B11.excited_states["5/2^-_1"]
    )

    assert (
        cs(B11.excited_states["5/2^-_1"].excitation_energy)
        == cs.energy_integrated_cross_section
    )
    assert cs(0.0, input_is_absolute_energy=False) == cs.energy_integrated_cross_section
    assert (
        cs.resonance_energy
        == cs.intermediate_state.excitation_energy - cs.initial_state.excitation_energy
    )
    assert cs.final_state_branching_ratio == 1.0
    assert cs.statistical_factor == 1.5
    assert np.allclose(
        cs.equidistant_energy_grid(0.5, 3),
        np.array(
            [
                B11.excited_states["5/2^-_1"].excitation_energy - 0.25,
                B11.excited_states["5/2^-_1"].excitation_energy,
                B11.excited_states["5/2^-_1"].excitation_energy + 0.25,
            ]
        ),
        rtol=1e-5,
    )
    assert np.allclose(
        cs.equidistant_energy_grid(
            (
                B11.excited_states["5/2^-_1"].excitation_energy - 0.25,
                B11.excited_states["5/2^-_1"].excitation_energy + 0.25,
            ),
            3,
        ),
        np.array(
            [
                B11.excited_states["5/2^-_1"].excitation_energy - 0.25,
                B11.excited_states["5/2^-_1"].excitation_energy,
                B11.excited_states["5/2^-_1"].excitation_energy + 0.25,
            ]
        ),
        rtol=1e-5,
    )
    assert np.allclose(
        cs.equidistant_probability_grid(0.5, 3),
        np.array(
            [
                B11.excited_states["5/2^-_1"].excitation_energy - 0.25,
                B11.excited_states["5/2^-_1"].excitation_energy,
                B11.excited_states["5/2^-_1"].excitation_energy + 0.25,
            ]
        ),
        rtol=1e-5,
    )
    assert np.allclose(
        cs.equidistant_probability_grid(
            (
                B11.excited_states["5/2^-_1"].excitation_energy - 0.25,
                B11.excited_states["5/2^-_1"].excitation_energy + 0.25,
            ),
            3,
        ),
        np.array(
            [
                B11.excited_states["5/2^-_1"].excitation_energy - 0.25,
                B11.excited_states["5/2^-_1"].excitation_energy,
                B11.excited_states["5/2^-_1"].excitation_energy + 0.25,
            ]
        ),
        rtol=1e-5,
    )

    cs = Resonance(
        initial_state=B11.ground_state,
        intermediate_state=B11.excited_states["3/2^-_2"],
        final_state=B11.excited_states["1/2^-_1"],
    )

    assert (
        cs.resonance_energy
        == cs.intermediate_state.excitation_energy - cs.initial_state.excitation_energy
    )
    assert cs.final_state_branching_ratio == 0.144
    assert cs.statistical_factor == 1.0

    energy_integrated_cross_section = (
        np.pi
        * np.pi
        * physical_constants["reduced Planck constant times c in MeV fm"][0] ** 2
        / (5.02030**2)
        * (3.0 + 1.0)
        / (3.0 + 1.0)
        * 0.856
        * 1.97e-6
        * 0.144
    )
    assert cs.energy_integrated_cross_section == energy_integrated_cross_section


def test_warnings():

    cs = BreitWigner(GroundState("0", 0, 1), State("2", 2, 1, 1e-3, {"0": 1.0}))

    with pytest.warns(UserWarning) as record:
        cs.coverage_interval(0.9)

    assert len(record) == 1
    assert "Unphysical negative" in str(record[0].message)

    with pytest.warns(UserWarning) as record:
        cs.coverage_interval(1.0)

    assert len(record) == 2
    assert "Unphysical negative" in str(record[0].message)
    assert "Infinite" in str(record[1].message)
