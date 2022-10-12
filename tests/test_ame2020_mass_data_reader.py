from pathlib import Path

from ries.constituents.ame_2020_mass_data_reader import AME2020MassDataReader

class TestAME2020MassDataReader:
    def test_ame2020_mass_data_reader(self):
        ame2020_reader = AME2020MassDataReader(
            Path(__file__).parent.absolute()
            / "../ries/constituents/ame2020_masses/mass_1.mas20"
        )
        ame2020_masses = ame2020_reader.read_mass_data()

        # Test the reference mass.
        assert ame2020_masses[6][12] == 12.0
        # Test a mass where A has a single digit.
        assert ame2020_masses[0][1] == 1.00866491590
        # Test a nontrivial mass where A has two digits.
        assert ame2020_masses[14][37] == 36.992945191
        # Test a mass where A has three digits.
        assert ame2020_masses[64][153] == 152.921756945
        # Test an estimated mass, i.e. an entry that has a '#' instead of a decimal point.
        # At the same time, test the last entry of the file to ensure it has been parsed completely.
        assert ame2020_masses[118][295] == 295.216178

    def test_element_symbol_reader(self):
        ame2020_reader = AME2020MassDataReader(
            Path(__file__).parent.absolute()
            / "../ries/constituents/ame2020_masses/mass_1.mas20"
        )
        X, Z = ame2020_reader.read_element_symbols()
        
        # Test element symbol with a single letter.
        assert X[6] == "C"
        # Test element symbol with two letters.
        assert X[79] == "Au"

        # Test the inverse
        assert Z["C"] == 6
        assert Z["Au"] == 79
        assert Z["Og"] == 118