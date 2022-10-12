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


class Isotope:
    """Class representing an isotope

    Attributes
    ----------
    Z: int
        Proton number
    A: int
        Mass number
    amu: float
        Mass in atomic mass units (u)
    abundance: float
        Isotopic abundance as a fraction (default: 1.0)
    X: str
        Element symbol (default: None, i.e. initialize from Z)
    ground_state: State object
        Ground state of the isotope (default: None)
    excited_states: dict str: State
        Dictionary of the excited states of the isotope with arbitrary strings as keys (default: None)
    """

    def __init__(
        self, Z, A, amu, abundance=0.0, X=None, ground_state=None, excited_states=None
    ):
        """Initialization

        Parameters
        ----------
        Z: int
            Proton number
        A: int
            Mass number
        amu: float
            Mass in atomic mass units (u)
        abundance: float
            Isotopic abundance as a fraction (default: 0.0)
        X: str
            Element symbol (default: None, i.e. initialize from Z)
        ground_state: State object
            Ground state of the isotope (default: None)
        excited_states: dict str: State
            Dictionary of the excited states of the isotope with arbitrary strings as keys (default: None)
        """
        self.Z = Z
        self.A = A
        self.amu = amu
        self.abundance = abundance
        self.X = X
        self.ground_state = ground_state
        self.excited_states = excited_states
