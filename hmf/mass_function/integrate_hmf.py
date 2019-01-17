"""
A supporting module that provides a routine to integrate the differential hmf in a robust manner.
"""
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
import numpy as np
import scipy.integrate as intg
import warnings


def hmf_integral_gtm(m, dndm=None, dndlnm=None, hmf_object=None, upper_integral=None, mass_density=False):
    """
    Cumulatively integrate dn/dm.

    Parameters
    ----------
    M : array_like
        Array of masses.

    dndm : array_like
        Array of dn/dm (corresponding to M)

    mass_density : bool, `False`
        Whether to calculate mass density (or number density).

    Returns
    -------
    ngtm : array_like
        Cumulative integral of dndm.

    Examples
    --------
    Using a simple power-law mass function:

    >>> import numpy as np
    >>> m = np.logspace(10,18,500)
    >>> dndm = m**-2
    >>> ngtm = hmf_integral_gtm(m,dndm)
    >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True

    The function always integrates to m=1e18, and extrapolates with a spline
    if data not provided:

    >>> m = np.logspace(10,12,500)
    >>> dndm = m**-2
    >>> ngtm = hmf_integral_gtm(m,dndm)
    >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True
    
    """
    # Eliminate NaN's
    if dndm is not None:
        warnings.warn("passing dndm is deprecated, please pass dndlnm. This will be removed in v4")
        dndlnm = m * dndm

    if dndlnm is None or len(m) != len(dndlnm):
        raise ValueError("dndlnm must be given, and of the same size as m")

    if len(m) < 4:
        raise ValueError("Not enough values in mass function to perform integral.")

    if mass_density:
        # Integrate m * dndlnm, rather than just dndlnm
        dndlnm *= m

    # Calculate the cumulative integral (backwards) of [m*]dndlnm
    cum_integral = np.concatenate((intg.cumtrapz(dndlnm[::-1], dx=np.log(m[1]/m[0]))[::-1], np.zeros(1)))

    if upper_integral is not None:
        return cum_integral + upper_integral
    elif hmf_object is not None:
        upper_m, upper_integral = calc_required_maximum_mass(hmf_object, mass_density)
        return cum_integral + upper_integral
    else:
        return cum_integral


def calc_required_maximum_mass(hmf_obj, mass_density=False):
    """
    Given a specification for a mass function, and a maximum desired mass
    scale, calculate the maximum mass scale required to ensure a converged
    upper integral.

    Parameters
    ----------
    hmf_obj: :class:~`hmf.Massfunction` instance
        A mass function instance that holds all the options and parameters
        for which the calculation should be performed.
    mass_density: bool, optional
        Whether to calculate the mass for a mass density integral (otherwise
        just a number density integral).

    Returns
    -------
    float: the maximum mass required.
    float: the integral up to the maximum mass (from the original upper mass).

    Notes
    -----
    The mass grid on which the result is calculated maintains the intrinsic
    mass resolution of the input hmf object.
    """
    # If the original maximum dndm was zero, we cannot do any better, so exit.
    if hmf_obj.dndm[-1] == 0:
        return np.log10(hmf_obj.m.max()), 0

    hmf_obj.Mmin = np.log10(hmf_obj.m.max())

    # this choice of Mmax is conservative.. inefficient but will always cover
    # the range of mass required to converge.
    hmf_obj.Mmax = max(18, hmf_obj.Mmax + 3)

    integ = hmf_obj.dndlnm
    if mass_density:
        integ *= hmf_obj.m

    cumsum = intg.cumtrapz(integ, dx=np.log(hmf_obj.m[1]) - np.log(hmf_obj.m[0]))

    ind = np.argwhere(1 - cumsum/cumsum[-1] < 1e-4)[0][0]

    # If it only converges on the last index (where it *must* "converge" to exactly zero), then
    # we warn the user, and report how far the second-last index is from the last, then return
    # the last index.
    if ind == len(cumsum) - 1:
        warnings.warn("The upper integral only has estimated convergence level of ", 1 - cumsum[-2]/cumsum[-1])

    return np.log10(hmf_obj.m[ind]), cumsum[ind]