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

r"""Debye model

The Debye model (see, e.g., Ref. :cite:`LandauLifshitz1980`) is an approximation for the specific \
heat of a harmonic solid, which assumes that the frequency distribution 
:math:`P^\mathrm{Debye} \left( \omega \right)` of the oscillators is 
given by:

.. math:: P^\mathrm{Debye} \left( \omega \right) = \frac{9 N k_B \omega^2}{\omega_\mathrm{D}} ~~~ \left( \omega \leq \omega_\mathrm{D}\right)

Here, :math:`N` is the number of particles in the solid, :math:`k_B` is the Boltzmann constant, 
and :math:`\omega_\mathrm{D}` is a cutoff frequency that is specific for the material of interest.
The Debye model interpolates between the :math:`T^3` law at low temperatures to the Dulong-Petit 
law at high temperatures.

The quantity

.. math:: T_\mathrm{D} = \hbar \omega_\mathrm{D}

is called the Debye temperature.
In a solid for which the Debye model is a good approximation, it can be shown :cite:`Lamb1939` 
that the particles move with a Maxwell-Boltzmann velocity distribution at an effective temperature

.. math:: T_\mathrm{eff} = 3 T \left( \frac{T}{T_\mathrm{D}} \right) \int_0^{\frac{T_\mathrm{D}}{T}} t^3 \left( \frac{1}{e^t - 1} + \frac{1}{2} \right) \mathrm{d}t

which is larger than the thermodynamic temperature :math:`T`.

Given values of :math:`T` and :math:`T_\mathrm{D}`, this module provides a function that solves 
the integral above numerically.

In addition, a dictionary of Debye temperatures at room temperature (`room_temperature_T_D`) 
from an online compilation 
:cite:`KnowledgeDoor2020` is provided.
The keys of the dictionary are the respective element symbols.
For example, for the Debye temperature of boron, one would type:

::

    from ries.resonance.debye_model import room_temperature_T_D
    
    print(room_temperature_T_D['B'])
"""

from warnings import warn

import numpy as np
from scipy.integrate import quad

from ries.constituents.element import X_from_Z

def effective_temperature_debye_approximation(T, T_D):
    """Debye model for the effective temperature

    Given the thermodynamic temperature and the Debye temperature, this function calculates the
    effective temperature of the material.
    The defining integral is solved numerically using `scipy.integrate.quad`.

    Parameters:

    - `T`, float or array_like, thermodynamic temperature in K.
    - `T_D`, float or array_like, Debye temperature in K.

    Returns:

    - float or array_like, effective temperature in K.

    Exceptions:

    - `ZeroDivisionError`, if a value of exactly 0 is entered for the thermodynamic temperature.
    """
    return (
        3.0
        * (T / T_D) ** 3
        * T
        * quad(lambda t: t ** 3 * (1.0 / (np.exp(t) - 1.0) + 0.5), 0.0, T_D / T)[0]
    )


effective_temperature_debye_approximation = np.vectorize(
    effective_temperature_debye_approximation
)

room_temperature_T_D = {}

def load_room_temperature_T_D_data():
    warn("At the moment, all isotopes in the room_temperature_T_D dictionary are "
         "initialized with an arbitrarily chosen Debye temperature of 500 K.")
    for Z in range(1,100):
        room_temperature_T_D[X_from_Z[Z]] = 500.