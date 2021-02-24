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

r"""
The National Institute of Standards and Technology (NIST) provides a list of
'Atomic Weights and Isotopic Compositions with Relative Atomic Masses' :cite:`Coursey2015`
which includes most of the known chemical elements.
Except for the element hydrogen, where the isotope tritium is included as well, the list contains
all naturally occurring (i.e. :math:`x \left( ^A\mathrm{X} \right) \neq 0`) isotopes of
'stable elements' (i.e. elements with at least one stable isotope).
For unstable elements like technetium, which cannot be found in nature, the authors of
Ref. :cite:`Coursey2015` appear to have selected the isotopes with the longest half-lives.

The code in the `NISTElementDataReader` class of this module reads the 'Linearized ASCII Output' 
from the NIST web page and creates a list of `Isotope` objects for use in the `ries` code.
A copy of the data file is distributed in the `ries` repository and loaded in the `element`
module into a dictionary called `natural_elements`.

For macroscopic samples, one is often also interested in the densities of the elements.
Compiling such a list is difficult, because chemical elements show allotropy, i.e. the same
element may exist in different forms (for example, carbon may be graphite or a diamond). 
The particle simulation toolkit Geant4 :cite:`Agostinelli2003` :cite:`Allison2006` 
:cite:`Allison2016` does provide such a list of densities of all natural elements at standard 
conditions.
Since no obvious references were found, it is difficult to decide how the Geant4 authors were 
able to decide on one density per each element, so these data should be used for rough estimates only.

The code in the `Geant4DensityDataReader` class reads data that were extracted from the source 
code of Geant4.
The data are distributed in the `ries` repository and loaded into a dictionary called 
`natural_element_densities`.
If possible, the density is also added to the `natural_elements` list above.
"""

from ries.constituents.isotope import Isotope


class NISTElementDataReader:
    """Class to parse a list of chemical elements by NIST in the 'Linearized ASCII Output' format

    The data file consists of multiline paragraphs for each isotope.
    Each line has a syntax like

    ::

        PROPERTY = VALUE

    or

    ::

        PROPERTY = VALUE(UNCERTAINTY)

    The `NISTElementDataReader` provides methods to find all isotopes of a given element in the list
    and read some of their properties.

    Attributes:

    - `element_data_file_name`, str, name of the ASCII file.
    - `*_prefix`, str, prefixes of the type 'PROPERTY = ' that identify the lines to be read by the
      `NISTElementDataReader`.
    """

    def __init__(self, element_data_file_name):
        self.element_data_file_name = element_data_file_name

        self.Z_prefix = "Atomic Number = "
        self.A_prefix = "Mass Number = "
        self.X_prefix = "Atomic Symbol = "
        self.amu_prefix = "Relative Atomic Mass = "
        self.abundance_prefix = "Isotopic Composition = "

    def read_nist_element_data(self, Z):
        """Find all isotopes and abundances for a given element.

        This function loops over the data file, finds all paragraphs for a given Z, and reads the isotope
        data.
        In the following, let :math:`n_A(Z)` denote the number of isotopes per element in the data file.

        Parameters:

        - `Z`, int, proton number of the element of interest.

        Returns:

        - (n,1) array of float, list of isotopic abundances.
        - (n,1) array of `Isotope` objects, list of isotopes.
        """
        abundances = {}
        isotopes = {}
        with open(self.element_data_file_name, "r") as file:
            for line in file:
                if self.Z_prefix in line:
                    current_Z = self.read_nist_element_property(
                        line, self.Z_prefix, int
                    )
                    if current_Z < Z:
                        continue
                    if current_Z > Z:
                        break
                elif self.X_prefix in line and current_Z == Z:
                    X = self.read_nist_element_property(line, self.X_prefix, str)
                elif self.A_prefix in line and current_Z == Z:
                    A = self.read_nist_element_property(line, self.A_prefix, int)
                elif self.amu_prefix in line and current_Z == Z:
                    amu = self.read_nist_element_property_with_uncertainty(
                        line, self.amu_prefix
                    )
                elif self.abundance_prefix in line and current_Z == Z:
                    abundance = self.read_nist_element_property_with_uncertainty(
                        line, self.abundance_prefix, default=0.0
                    )
                    AX = "{:d}{}".format(A, X)
                    abundances[AX] = abundance
                    isotopes[AX] = Isotope(AX, amu)

        return (abundances, isotopes)

    @staticmethod
    def read_nist_element_property(line, prefix, property_type=str, default=None):
        """Read the value of a single property from a line

        Parameters:

        - `line`, str, line from which to read the value.
        - `prefix`, str, prefix that identifies the property.
        - `property`, type, type of the value.
          This information will be used to cast the value string extracted from `line` to the correct type.
        - `default`, default value to be returned if the property's value is an empty string (`''`).

        Returns:

        - value of type `property_type` or `default`.
        """
        prop = line[
            len(prefix) : -1
        ]  # Take all characters except the last two, which are the newline
        # escape sequence '\n'.
        if prop != "":
            return property_type(prop)
        return default

    def read_nist_element_property_with_uncertainty(self, line, prefix, default=None):
        """Read a floating-point value from a line that also contains an uncertainty

        Given a line like

        ::

            PROPERTY = VALUE(UNCERTAINTY)

        this function reads `VALUE` and casts it to a `float`.

        Parameters:

        - `line`, str, line from which to read the value.
        - `prefix`, str, prefix that identifies the property.
        - `property`, type, type of the value.
        - `default`, default value to be returned if the property's value is an empty string (`''`).

        Returns:

        - float, value
        """
        return self.read_nist_element_property(
            line[0 : line.find("(") + 1], prefix, float, default
        )

    def read_nist_element_symbols(self):
        """Find all element symbols in the data file and organize them in a dictionary.

        This functions loops over the data file and finds all unique element symbols.
        They will be collected in a dictionary such that the proton number is the key, and the element
        symbol the value.
        For example, let the dictionary be called `X`.
        To get a list of all element symbols for Z = 1 to Z = 10, one could do:

        ::

            [X[Z] for Z in range(1, 11)]

        Returns:

        - dictionary with element symbols as keys and proton numbers as values.
        """
        X = {}
        with open(self.element_data_file_name, "r") as file:
            for line in file:
                if self.Z_prefix in line:
                    Z = self.read_nist_element_property(line, self.Z_prefix, int)
                if self.X_prefix in line and Z not in X:
                    X[Z] = self.read_nist_element_property(line, self.X_prefix)
        return X


class Geant4DensityDataReader:
    r"""Class to parse a list of chemical-element densities

    The data file consists of one line per element.
    Each line has the structure

    ::

        ELEMENT_SYMBOL  VALUE

    where `ELEMENT_SYMBOL` is the element symbol and `VALUE` is the density in
    :math:`\mathrm{g} \mathrm{cm}^{-3}`.

    Attributes:

    - `density_data_file_name`, str, name of the data file.
    - `density_conversion`, callable, function to convert the density values (default: identity function). For example, to convert the densities from :math:`\mathrm{g} \mathrm{cm}^{-3}` to :math:`\mathrm{kg} \mathrm{m}^{-3}`, one would use:

    ::

        density_conversion = lambda rho: 1e3*rho
    """

    def __init__(self, density_data_file_name, density_conversion=lambda rho: rho):
        """Initialization

        Parameters:

        - `density_data_file_name`, str, name of the data file.
        - `density_conversion`, callable, function to convert the density values (default: identity function).
        """
        self.density_data_file_name = density_data_file_name
        self.density_conversion = density_conversion

    def read_density_data(self, X):
        """Read density of a single element from data file

        This method loops through the lines in the file given by `density_data_file_name` until
        it finds the required element symbol in the first two characters of the line.
        If found, the rest of the line is cast to a `float`, converted using the user-defined
        `density_conversion`, and returned.
        If the loop ends, but no data have been found, `None` is returned.

        Parameters:

        - `X`, str, element symbol.

        Returns:

        - float, density of the element, converted by the user-defined density-conversion function. If no conversion was given, the value is in g/cm^3. If the given element symbol is not found, None is returned.
        """

        with open(self.density_data_file_name, "r") as file:
            for line in file:
                if X in line[0:2]:
                    return self.density_conversion(float(line[2:]))

            return None
