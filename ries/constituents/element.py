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

from ries.constituents.geant4_densities.densities import densities
from ries.constituents.ame_2020_mass_data_reader import AME2020MassDataReader
from ries.constituents.isotope import Isotope
from ries.constituents.iupac_isotopic_compositions.isotopic_compositions import (
    isotopic_compositions,
)


class Element:
    """Class representing a chemical element

    Attributes
    ----------
    Z: int
        Proton number
    X: str
        Element symbol (default: None, i.e. initialize from Z)
    isotopes: dict int: Isotope
        Dictionary of the isotopic composition of the element with the mass numbers as keys (default: None)
    density: float
        Density of the element in arbitrary units (default: 0.0).
    """

    def __init__(self, Z, X, isotopes, density=0.0):
        """Initialization

        Parameters
        ----------
        Z: int
            Proton number
        X: str
            Element symbol (default: None, i.e. initialize from Z)
        isotopes: dict int: Isotope
            Dictionary of the isotopic composition of the element with the mass numbers as keys (default: None)
        density: float
            Density of the element in arbitrary units (default: 0.0).
        """
        self.Z = Z
        self.X = X
        self.isotopes = isotopes
        self.density = density

    def amu(self):
        """Calculate element mass from the isotopic composition

        Returns
        -------
        float
            Element mass in atomic mass units (u)
        """
        return sum(
            [self.isotopes[A].amu * self.isotopes[A].abundance for A in self.isotopes]
        )


def create_natural_element_dictionary():
    ame2020_mass_data_reader = AME2020MassDataReader(
        Path(__file__).parent.absolute() / "ame2020_masses/mass_1.mas20"
    )
    ame_masses = ame2020_mass_data_reader.read_mass_data()
    X_from_Z, Z_from_X = ame2020_mass_data_reader.read_element_symbols()

    natural_elements = {}
    for Z in X_from_Z:
        abundances = {}
        isotopes = {}
        if Z in isotopic_compositions:
            for A in isotopic_compositions[Z]:
                isotopes[A] = Isotope(
                    Z=Z,
                    A=A,
                    amu=ame_masses[Z][A],
                    abundance=isotopic_compositions[Z][A],
                    X=X_from_Z[Z],
                )
            natural_elements[Z] = Element(
                Z=Z,
                X=X_from_Z[Z],
                isotopes=isotopes,
                density=densities[Z] if Z in densities else None,
            )
    return (X_from_Z, Z_from_X, natural_elements)


X_from_Z, Z_from_X, natural_elements = create_natural_element_dictionary()
