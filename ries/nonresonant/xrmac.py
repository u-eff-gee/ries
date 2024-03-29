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
This module reads and processes the x-ray mass attenuation coefficients (XRMAC)
:math:`\mu / \rho` of Hubbell and Seltzer :cite:`HubbellSeltzer2004`.
These coefficients give the total **nonresonant** [a]_ cross section per atom
:math:`\sigma_\mathrm{NR}`, normalized to the mass of the
chemical element (i.e. assuming a natural abundance of isotopes) :math:`m \left( X \right)`:

.. math:: \frac{\mu}{\rho} = \frac{\sigma_\mathrm{NR}}{m \left( X \right)}.

Here, :math:`\rho` denotes the density of a material in units of mass per volume.
The energy (:math:`E`)-dependent quantity :math:`\mu` has the dimension of an inverse length, and acts
as the decay constant in the Beer-Lambert law:

.. math:: \frac{I \left( E, z \right)}{I \left( E, 0 \right)} = \exp \left[ - \mu \left( E \right) z \right].

Here, :math:`I \left( E, z \right)` denotes the dependence of the intensity of a collinear
photon beam on the penetration depth :math:`z` into a material.

In Ref. :cite:`HubbellSeltzer2004`, the energy dependence of :math:`\mu / \rho` is tabulated
for discrete values of the energy.
This module reads the tabulated data, converts them to a cross section per atom using the
equation above, and interpolates them to obtain a continuous cross section.
At the moment, only the data for 'elemental media' from Ref. :cite:`HubbellSeltzer2004` are
available.
They are stored in the `xrmac_cm2_per_g` and `xrmac_fm2_per_atom` dictionaries, which uses the 
element symbol as a key.
They contain the same data, but in different units as indicated by the dictionary name.
For example, to obtain the XRMAC for lead at an energy of 1 MeV in :math:`\mathrm{cm^2} \mathrm{g^{-1}}`, use:

::

    from ries.nonresonant.xrmac import xrmac

    print(xrmac_cm2_per_g['Pb'](1.))

.. [a] Hubbell and Seltzer call it the 'total cross section per atom', since it does not
    include nuclear excitations.
    In the scope of the `ries` code, which is focused on those nuclear resonances, it can be treated
    as the sum of all nonresonant contributions.
    For a more detailed listing of the processes that are included in :math:`\mu / \rho`, please
    refer to Ref. :cite:`HubbellSeltzer2004`.
"""

from pathlib import Path
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import physical_constants

from ries.constituents.element import natural_elements, X_from_Z
from ries.nonresonant.nonresonant import Nonresonant
from ries.nonresonant.klein_nishina import KleinNishina


class XRMAC(Nonresonant):
    r"""Nonresonant attenuation cross section based on the x-ray mass attenuation coefficients of Hubbell and Seltzer.

    Given a list of energies :math:`E_i` (:math:`0 \leq i < n`) and a list of values of the XRMAC
    :math:`\left( \mu / \rho \right)_i` at these energies, this class converts the data using
    a user-defined conversion function and interpolates them to obtain a
    function of a continuous variable.

    Since the XRMACS vary strongly with the energy, the code interpolates the base-10 logarithms
    :math:`\log_{10} \left( E_i \right)` and
    :math:`\log_{10} \left[ \left( \mu / \rho \right)_i \right]` to be able to use a
    linear interpolation function.

    This class also provides a method to read XRMAC data in the 'ASCII format' of Hubbell and
    Seltzer.

    Attributes:

    - `data`: (n,2) array or str. If an array is given, `data` is assumed to contain :math:`E_i` in
      the first column and :math:`\left( \mu / \rho \right)_i` in the second column, both of them in
      arbitrary units (the data of Hubbell and Seltzer are given in
      :math:`\mathrm{MeV}` and :math:`\mathrm{cm}^2 g^{-1}`, respectively).
      If a string is given, it is assumed to be a file that contains XRMAC data in the ASCII format of
      Hubbell and Seltzer.
    - `energy_conversion` and `xrmac_conversion`, functions to convert the
      energy- and XRMAC data (default: no conversion, i.e. functions that simply return their input).
      For example, to convert energies from MeV to eV, set

      ::

          energy_conversion = lambda energy: energy*1e6

    - `interpolation_log_log`, function which takes a base-10 logarithm of an energy and returns the
      base-10 logarithm of an XRMAC in user-defined units.
    """

    def __init__(
        self,
        data,
        energy_conversion=lambda energy: energy,
        xrmac_conversion=lambda xrmac: xrmac,
    ):
        r"""Initialization

        Parameters:

        - `data`: (n,2) array or str. If an array is given, `data` is assumed to contain :math:`E_i` in
          the first column and :math:`\left( \mu / \rho \right)_i` in the second column, both of them in
          arbitrary units (the data of Hubbell and Seltzer are given in
          :math:`\mathrm{MeV}` and :math:`\mathrm{cm}^2 g^{-1}`, respectively).
          If a string is given, it is assumed to be a file that contains XRMAC data in the ASCII format of
          Hubbell and Seltzer.
        - `energy_conversion` and `xrmac_conversion`, functions to convert the
          energy- and XRMAC data (default: no conversion, i.e. functions that simply return their input).
        """
        self.energy_conversion = energy_conversion
        self.xrmac_conversion = xrmac_conversion
        if isinstance(data, str):
            data = self.read_nist_xrmac(data)
        self.data = []
        for d in data:
            self.data.append(
                [
                    self.energy_conversion(d[0]),
                    self.xrmac_conversion(d[1]),
                    self.xrmac_conversion(d[2]),
                ]
            )
        self.data = np.array(self.data)
        self.interpolation_log_log = self.interpolate_log_log(self.data)

    def __call__(self, E):
        r"""Return XRMAC for a given energy.

        Returns 10 to the power of `self.interpolation_log_log`.

        Parameters:

        - `E`, array_like or scalar, energy in user-defined units (Hubbell and Seltzer: :math:`\mathrm{MeV}`).

        Returns:

        - array_like or scalar, XRMAC in user-defined units (Hubbell and Seltzer: :math:`\mathrm{cm}^2 g^{-1}`).
        """
        return 10 ** (self.interpolation_log_log(np.log10(E)))

    @staticmethod
    def interpolate_log_log(data):
        r"""Interpolate base-10 logarithm of data pairs.

        Given a set of pairs :math:`\left( x_i, y_i \right)` (:math:`0 \leq i < n`), this function
        linearly (default setting of scipy) interpolates
        :math:`\left[ \log_{10} \left( x_i \right), \log_{10} \left( y_i \right) \right]`
        and returns a callable function.

        Parameters:

        - `data`, (n,2) array with :math:`x_i` in the first, and :math:`y_i` in the second column.

        Returns:

        - Callable function which returns :math:`\log \left( y \right)` for a given
          value of :math:`\log \left( x \right)`.
        """
        return interp1d(
            np.log10(data[:, 0]),
            np.log10(data[:, 1]),
            bounds_error=False,
            fill_value=(np.log10(data[0][1]), np.log10(data[-1][1])),
        )

    def read_nist_xrmac(self, xrmac_file_name):
        """Function to read XRMAC data from a file with the ASCII format of Hubbell and Seltzer.

        This function can parse data files for 'elemental media' with an arbitrary number of lines.
        Each line of the original ASCII table contains data in the following format:

        ::

            line[0] # Single character which indicates an x-ray resonance or space.
            line[3:14] # Energy in MeV
            line[16:25] # XRMAC in cm**2/g
            line[27:36] # Mass-energy absorption coefficient in cm**2/g

        The data are separated by two spaces, respectively.
        Since some lighter elements do not have x-ray resonances in the given energy range, some files do
        not have the x-ray resonance column, but the format is:

        ::

            line[0:11] # Energy in MeV
            line[13:22] # XRMAC in cm**2/g
            line[24:33] # Mass-energy absorption coefficient in cm**2/g

        When reading, a file, the data are converted using `self.energy_conversion` and
        `self.xrmac_conversion`.

        Parameters:

        - `xrmax_file_name`, str, file that contains XRMAC data in the ASCII format of Hubbell and Seltzer

        Returns:

        - (n,3) array with converted energies in the first, XRMAC in the second, and mass-energy absorption
          coefficients in the third column.
        """
        data = []
        skip = 0
        # Some data files have a three-character column that indicates the label of the atomic
        # resonance.
        # Open the file a first time to find out whether this is the case.
        # If yes, skip this column when reading.
        with open(xrmac_file_name, "r") as file:
            if not file.readline()[0].isdigit():
                skip = 3
        with open(xrmac_file_name, "r") as file:
            for line in file:
                line = line[skip:-2].split(sep="  ")
                data.append(
                    [
                        float(line[0]),
                        float(line[1]),
                        float(line[2]),
                    ]
                )
        return np.array(data)


# Read the XRMAC data of Hubbell and Seltzer supplied with the `ries` repository and create the
# `xrmac_cm2_per_g` and `xrmac_fm2_per_atom` dictionaries.
# Alternatively, use the analytical expression of the Compton-scattering cross section to
# substitute nonexistent XRMAC datasets.
xrmac_data_dir = Path(__file__).parent.absolute() / "../nonresonant/nist_xrmac/"

xrmac_cm2_per_g = {}
xrmac_fm2_per_atom = {}


def load_xrmac_data():
    cm_to_fm = 1e13
    kg_to_g = 1e3
    default_data = np.zeros((100, 3))
    default_data[:, 0] = np.logspace(
        -3, np.log10(20.0), len(default_data)
    )  # Energies in MeV. Same range as the NIST datasets.
    missing_datasets = []

    for Z in range(1, 93):
        if (xrmac_data_file := xrmac_data_dir / "{:02d}.txt".format(Z)).is_file():
            xrmac_cm2_per_g[Z] = XRMAC(xrmac_data_file)
            xrmac_fm2_per_atom[Z] = XRMAC(
                str(xrmac_data_dir / "{:02d}.txt".format(Z)),
                xrmac_conversion=lambda xrmac: xrmac
                * cm_to_fm**2
                * natural_elements[Z].amu()
                * physical_constants["atomic mass constant"][0]
                * kg_to_g,
            )
        else:
            default_data[:, 1] = KleinNishina(Z)(default_data[:, 0])
            xrmac_fm2_per_atom[Z] = XRMAC(default_data)
            xrmac_cm2_per_g[Z] = XRMAC(
                default_data,
                xrmac_conversion=lambda xrmac: xrmac
                / (
                    cm_to_fm**2
                    * natural_elements[Z].amu()
                    * physical_constants["atomic mass constant"][0]
                    * kg_to_g
                ),
            )
            missing_datasets.append(Z)

    if len(missing_datasets) > 0:
        warning_message = (
            "X-ray mass attenuation coefficient (XRMAC) data for the following elements "
            "were not found in '{}' and replaced by an analytical expression for the Compton-scattering "
            "cross section:\n".format(xrmac_data_dir)
        )
        for n_element, element in enumerate(missing_datasets):
            warning_message += X_from_Z[element]
            if n_element < len(missing_datasets) - 1:
                warning_message += ", "
        warning_message += (
            "\nPlease note that the Compton-scattering cross section is only a rough "
            "approximation of the true XRMAC."
        )
        warn(warning_message)
