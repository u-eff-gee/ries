"""
Class for a photonuclear cross section with a single isolated resonance.

The cross section for a resonant reaction :math:`\\sigma_r` is assumed to consists of a localized \
'peak' with an arbitrary shape and a 'tail' that decays sufficiently fast towards low and high \
energies so that:

.. math:: \\int_0^{\\infty} \\sigma_r \\left( E \\right) \\mathrm{d} E \\equiv I_r < \\infty.

Here, the quantity :math:`I_r` is the energy-integrated cross section \
(the general form of :math:`I_r` for photonuclear reactions can be found below).
By dividing the cross section by :math:`I_r`, a function

.. math:: P_r \\left( E \\right) = \\frac{\\sigma_r \\left( E \\right)}{I_r}

is obtained, which has properties of a probability density function (PDF), in particular:

.. math:: \\int_0^\\infty P_r \\left( E \\right) \\mathrm{d} E = 1.

A typical Doppler-broadened [g]_ nuclear resonance below the neutron separation threshold has a full \
width at half maximum of few :math:`\\mathrm{eV}` :cite:`ENSDF2020`, while the best \
'general-purpose' detectors have an energy resolution on the order of few :math:`\\mathrm{keV}` \
and the typical spectral width of an artificial `monochromatic` photon beam in the \
with a tunable energy in the range of few :math:`\\mathrm{MeV}` is on the order of tens of \
:math:`\\mathrm{keV}` [h]_.
This discrepancy of about five orders of magnitude in energy scales means that the nuclear \
resonances only contribute significantly to the total cross section in a very small interval \
compared to the entire energy range of interest :math:`\\left[ E_0, E_{n-1} \\right]`.
Any numerical integration algorithm will sample the cross section at a finite number of grid \
points :math:`\\left\\{ E_i \\right\\}` (:math:`0 \leq i < n`, :math:`E_i < E_{i + 1}` for \
all :math:`i < n - 1`) to approximate the integral.
As demonstrated in one of the tests of `ries`, general-purpose adaptive algorithms like \
`scipy.integrate.quad` will not be able to close in on the resonance if its location is unknown.

A brute-force approach to solve this problem is to use a more primitive integration algorithm \
like Darboux sums (see `ries.integration.darboux`) or the trapezoidal rule on an equidistant grid, \
i.e.

.. math:: E_{i+1} - E_{i} = E_{j+1} - E_{j}

for all :math:`0 \\leq i, j < n-1`, and a large value of :math:`n`.
This method is simple to implement, but not very efficient, since it does not make use of the \
resonance character of the function.

Here, the range :math:`\\left[ E_0, E_{n-1} \\right]` is partitioned into intervals of equal \
probability instead.
This procedure is based on the inverse of the cumulative distribution function (CDF) :math:`F_P`, \
the quantile function or percent-point function (PPF) :math:`F_P^{-1}`.
In the present context, the CDF is defined as

.. math:: F_P \\left( E \\right) = P \\left( E^\\prime \\leq E \\right) = \\int_0^{E} P \\left( E^\\prime \\right) \\mathrm{d} E^\\prime,

and it gives the probability of finding a value of :math:`E^\\prime` less or equal to :math:`E`.
Consequently, the PPF :math:`F_P^{-1} \\left[ F_P \\left( E \\right) \\right]` returns the \
energy that corresponds to this probability.
An equal-probability partition with :math:`n` grid points between :math:`E_0` and :math:`E_{n-1}` \
is obtained as follows:
Since the total probability inside the interval :math:`\\left[ E_0, E_{n-1} \\right]` is given by

.. math:: P \\left( E_0 \\leq E \\leq E_{n-1} \\right) = F_P \\left( E_{n-1} \\right) - F_P \\left( E_0 \\right),

the :math:`n` energies that correspond to the limits :math:`n-1` partitions with a probability

.. math:: \\frac{P \\left( E_0 \\leq E \\leq E_{n-1} \\right)}{n-1}

are given by:

.. math:: E_i = F_P^{-1} \\left\\{ F_P \\left( E_0 \\right) + \\frac{i}{n-1} \\left[ F_P \\left( E_{n-1} \\right) - F_P \\left( E_0 \\right) \\right] \\right\\}.

An equal-probability grid has the advantage that it is dense in the regions that contribute most \
to the energy-integrated cross section.

The `Resonance` class in the present module has been designed for a straighforward integration \
of the distributions in the `scipy.stats` submodule.
The continuous probability distributions in `scipy.stats` are classes that provide methods to \
evaluate their PDF, CDF, and PPF.
For example:

::

    from scipy.stats import norm
    
    print(
        norm.ppf(norm.cdf(0.)) == 0.
    )

Any user-defined probability distribution should be derived from `scipy.stats.rv_continuous`, or \
at least provide the methods `pdf`, `cdf`, and `ppf`.

It is important to note that this module is intended for the use with **isolated** resonances \
in the sense of the theory of resonance by Breit and Wigner :cite:`BreitWigner1936`.
If resonances get too close to each other, interference effects will occur that are outside the \
scope of the present code.

For an electromagnetic excitation from an isolated state 0 to an isolated state 2 by absorption \
of a photon, the cross section is proportional to \
[compare Eq. (12) in :cite:`BreitWigner1936`]:

.. math:: \\underbrace{\\left( \\frac{\\pi \\hbar c}{E}  \\right)^2}_{\\frac{\\Lambda^2}{4}} \\underbrace{\\frac{2 J_2 + 1}{2 J_0 + 1}}_{S} \\Gamma_{2 \\to 0}.

In the expression above, :math:`\\hbar c` denotes the product of the reduced Planck constant and \
the speed of light.
The cross section depends on the de-Broglie wave length :math:`\\Lambda` of the incident photon, \
which is related to its energy :math:`E`, \
a statistical factor :math:`S` that takes into account the magnetic substates, and the partial \
width for a transition between the two states [i]_.

Since the FWHM of the cross sections for isolated electromagnetic resonances is usually much \
smaller than the resonance energy :math:`E_2 - E_0`, the approximation

.. math:: E \\approx E_2 - E_0 + R

or (with an obvious definition of :math:`\\Lambda_{2 \\to 0}`)

.. math:: \\Lambda \\approx \\Lambda_{2 \\to 0}

can be used [j]_.
Here, :math:`R` is a correction term that takes into account the recoil energy transferred to the \
nucleus in the absorption process, which is also much smaller than the resonance energy.
With this approximation, the energy-integrated cross section is given by:

.. math:: I_{0 \\to 2} = \\frac{\\Lambda_{2 \\to 0}^2}{4} \\frac{2 J_2 + 1}{2 J_0 + 1} \\Gamma_{2 \\to 0}.

.. [g] See also `ries.resonance.voigt` and `ries.resonance.gauss`.
.. [h] The phrases 'general purpose' and 'tunable energy' are meant to exclude instruments \
like GAMS :cite:`Kessler2001` that can reach extremely high resolution in special cases.
.. [i] Here, it was assumed that the detailed-balance theorem holds, i.e. \
:math:`\\Gamma_{2 \\to 0} = \\Gamma_{0 \\to 2}` or 'the partial widths for excitation and \
decay are the same'. \
For electromagnetic transitions, this assumption is valid (see, e.g., Secs. X.2.E and XII.4.A in \
Ref. :cite:`BlattWeisskopf1979`).
.. [j] This approximation is so common that it is sometimes introduced tacitly. \
For example, compare Refs. :cite:`Metzger1959` and :cite:`Romig2015` to :cite:`BreitWigner1936` and :cite:`Pruet2006`.
"""

import numpy as np

from scipy.constants import physical_constants
from scipy.stats import uniform

from ries.cross_section import CrossSection
from ries.resonance.recoil import NoRecoil

class Resonance(CrossSection):
    """Class for a resonance cross section that can be modeled by a continuous probability distribution.

This class models an 'excitation cross section' :math:`\\sigma_{0 \\to 2}` for a resonant \
reaction that excites a nucleus from an initial state :math:`J_0` to \
an excited state :math:`J_2`.

Optionally, the cross section can be restricted to a certain decay channel :math:`J_1` of the \
excited state, which is denoted as :math:`\\sigma_{0 \\to 2 \\to 1}`.
In this case, the excitation cross section is multiplied by the branching ratio for the decay \
to :math:`J_1`:

.. math:: \\sigma_{0 \\to 2 \\to 1} = \\sigma_{0 \\to 2} \\frac{\\Gamma_{2 \\to 1}}{\\Gamma_2}.

Attributes:

- `*_state`, `State` objects, initial, intermediate, and final state of the reaction. \
The final state is optional and defaults to `None`.
- `resonance_energy`, float. The resonance energy is equal to the energy difference of the states \
0 and 2 up to an optional recoil correction.
- `energy_integrated_cross_section_constant`, float, the constant :math:`\\left( \\pi \\hbar c \\right)^2` in :math:`\\mathrm{MeV} \\mathrm{fm}^2`.
    """
    def __init__(self, initial_state, intermediate_state,
        final_state=None, recoil_correction=NoRecoil()):
        self.initial_state = initial_state
        self.intermediate_state = intermediate_state
        self.final_state = final_state

        self.resonance_energy = recoil_correction(self.intermediate_state.excitation_energy - self.initial_state.excitation_energy)
        self.energy_integrated_cross_section_constant = (np.pi*physical_constants['reduced Planck constant times c in MeV fm'][0])**2
        self.statistical_factor = self.get_statistical_factor()
        self.final_state_branching_ratio = self.get_final_state_branching_ratio()
        self.energy_integrated_cross_section = self.get_energy_integrated_cross_section()

        self.probability_distribution = uniform
        self.probability_distribution_parameters = (self.resonance_energy-0.5, 1.)

    def __call__(self, energy, input_is_absolute_energy=True):
        if not input_is_absolute_energy:
            energy = energy + self.resonance_energy
        return self.energy_integrated_cross_section*self.probability_distribution.pdf(energy, *self.probability_distribution_parameters)

    def coverage_interval(self, coverage):
        return self.probability_distribution.ppf(
            0.5*np.array([1. - coverage, 1.+coverage]),
            *self.probability_distribution_parameters
        )

    def equidistant_energy_grid(self, coverage_or_limits, n_points):
        if isinstance(coverage_or_limits, (int, float)):
            coverage_or_limits = self.coverage_interval(coverage_or_limits)
        return CrossSection.equidistant_energy_grid(self, coverage_or_limits, n_points)

    def equidistant_probability_grid(self, coverage_or_limits, n_points):
        if isinstance(coverage_or_limits, (int, float)):
            limits = (0.5*(1.-coverage_or_limits), 0.5*(1.+coverage_or_limits))
        else:
            limits = self.probability_distribution.cdf(coverage_or_limits, *self.probability_distribution_parameters)
        equ_dis_pro_grid = self.probability_distribution.ppf(
            np.linspace(limits[0], limits[1], n_points),
            *self.probability_distribution_parameters
        )
        # This last if clause prevents rounding errors.
        # For finite limits that are very far away from the resonance energy, 
        # probability_distribution.cdf() may return 0 or 1 instead of 0.000... or 0.999..., 
        # which would then cause probability_distribution.ppf to return -np.inf or np.inf as 
        # limits of the grid instead of the given values.
        if not isinstance(coverage_or_limits, (int, float)):
            equ_dis_pro_grid[0] = coverage_or_limits[0]
            equ_dis_pro_grid[-1] = coverage_or_limits[1]
        return equ_dis_pro_grid

    def get_energy_integrated_cross_section(self):
        return (
            self.energy_integrated_cross_section_constant
            /((self.resonance_energy)**2)
            *self.statistical_factor
            *self.intermediate_state.partial_widths[self.initial_state.J_pi]
            *self.final_state_branching_ratio
        )

    def get_final_state_branching_ratio(self):
        if self.final_state is None:
            return 1.
        return (
            self.intermediate_state.partial_widths[self.final_state.J_pi]
            /self.intermediate_state.width)

    def get_statistical_factor(self):
        return (
            (self.intermediate_state.two_J+1.)
            /(self.initial_state.two_J+1.)
        )