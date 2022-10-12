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

class AME2020MassDataReader:
    """Class to parse the Atomic Mass Evaluation - AME 2020 data

    The Atomic Mass Evaluation (AME) :cite:`Huang2021` :cite:`Wang2021` is an evaluated database of
    atomic masses by the International Atomic Energy Agency.
    The committee publishes a structured text file with detailed instructions on how to parse it.
    Based on the instructions, this class extracts all information from the file that is needed to 
    uniquely identify isotopes, plus their nuclear masses in atomic mass units.

    At present and to the knowledge of the author, only the `mass_1.mas20` file from the 2020 
    evaluation, which is distributed along with the `ries` source code, can be parsed 
    (there was a format change compared to the 2016 format).

    Attributes
    ----------
    ame2020_file: str
        Name the AME 2020 data file, possibly including the file-system path.
    """

    def __init__(self, ame2020_file):
        """"""
        self.ame2020_file = ame2020_file

    def read_mass_data(self):
        """Create a dictionary of atomic masses by reading the AME 2020 file

        This function reads the entire file and creates a nested dictionary with the proton number Z and
        the mass number A as keys, and the atomic masses as values.

        Returns
        -------
        dict of dict's
            Nested dictionary which returns the mass of a specific isotope in
            atomic mass units (u, not micro-u as in the data file)when given a proton number Z and
            mass number A.

        Example
        -------
        >>> ame2020_mass_data_reader = AME2020MassDataReader("mass_1.mas20")
        >>> ame2020_masses = ame2020_mass_data_reader.read_mass_data()
        >>> ame2020_masses[6][12]
        12.0
        """
        ame2020_masses = {}
        with open(self.ame2020_file) as file:
            for n_line, line in enumerate(file):
                if n_line > 35: # Skip header.
                    Z = int(line[9:14].strip())
                    A = int(line[14:19].strip())
                    mass = float(line[106:125].strip().replace(" ", "").replace("#", "."))*1e-6

                    if Z not in ame2020_masses:
                        ame2020_masses[Z] = {}
                    ame2020_masses[Z][A] = mass

        return ame2020_masses

    def read_element_symbols(self):
        """Create dictionaries to relate element symbols and proton numbers from the AME2020 file

        The AME2020 file is a conveniently available list of known isotopes, so it should 
        definitely contain all chemical-element symbols.
        
        Returns
        -------
        X_from_Z: dict
            Dictionary with proton numbers (int) as keys and element symbols (str) as values.
        Z_from_X: dict
            Dictionary with element symbols (str) as keys and protons numbers (int) as values.

        Examples
        --------
        >>> ame2020_mass_data_reader = AME2020MassDataReader("mass_1.mas20")
        >>> X_from_Z, Z_from_X = ame2020_mass_data_reader.read_element_symbols()
        >>> X_from_Z[6]
        "C"
        >>> Z_from_X["Au"]
        79
        """
        X_from_Z = {}
        Z_from_X = {}

        with open(self.ame2020_file) as file:
            for n_line, line in enumerate(file):
                if n_line > 35: # Skip header.
                    proton_number = int(line[9:14].strip())
                    element_symbol = line[20:22].strip()

                    if proton_number not in X_from_Z:
                        X_from_Z[proton_number] = element_symbol
                    if element_symbol not in Z_from_X:
                        Z_from_X[element_symbol] = proton_number

        return (X_from_Z, Z_from_X)