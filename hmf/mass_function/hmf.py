'''
The primary module for user-interaction with the :mod:`hmf` package.

The module contains a single class, `MassFunction`, which wraps almost all the
functionality of :mod:`hmf` in an easy-to-use way.
'''

import copy
import warnings

import numpy as np
from numpy import issubclass_
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import minimize

from . import fitting_functions as ff
from .integrate_hmf import hmf_integral_gtm as int_gtm, calc_required_maximum_mass
from .._internals._cache import parameter, cached_quantity
from .._internals._framework import get_model_
from ..density_field import transfer
from ..density_field.filters import TopHat, Filter
from ..halos.mass_definitions import MassDefinition as md


class MassFunction(transfer.Transfer):
    """
    An object containing all relevant quantities for the mass function.

    The purpose of this class is to calculate many quantities associated with
    the dark matter halo mass function (HMF). The class is initialized to form a
    cosmology and takes in various options as to how to calculate all
    further quantities.

    All required outputs are provided as ``@property`` attributes for ease of
    access.

    Contains an update() method which can be passed arguments to update, in the
    most optimal manner. All output quantities are calculated only when needed
    (but stored after first calculation for quick access).

    In addition to the parameters directly passed to this class, others are available
    which are passed on to its superclass. To read a standard documented list of (all) parameters,
    use ``MassFunction.parameter_info()``. If you want to just see the plain list of available parameters,
    use ``MassFunction.get_all_parameters()``.To see the actual defaults for each parameter, use
    ``MassFunction.get_all_parameter_defaults()``.

    Examples
    --------
    Since all parameters have reasonable defaults, the most obvious thing to do is

    >>> h = MassFunction()
    >>> h.dndm

    Many different parameters may be passed, both models and parameters of those models. For instance:

    >>> h = MassFunction(z=1.0,Mmin=8,hmf_model="SMT")
    >>> h.dndm

    Once instantiated, changing parameters should be done through the :meth:`update` method:

    >>> h.update(z=2)
    >>> h.dndm
    """

    def __init__(self, Mmin=10, Mmax=15, dlog10m=0.01, hmf_model=ff.Tinker08, hmf_params=None,
                 mdef_model=None, mdef_params=None,
                 delta_c=1.686, filter_model=TopHat, filter_params=None,
                 ensure_convergent_upper_integral=True,
                 **transfer_kwargs):
        # # Call super init MUST BE DONE FIRST.
        super(MassFunction, self).__init__(**transfer_kwargs)

        # Set all given parameters.
        self.hmf_model = hmf_model
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.dlog10m = dlog10m
        self.mdef_model = mdef_model
        self.mdef_params = mdef_params or {}
        # self.cut_fit = cut_fit
        # self.z2 = z2
        # self.nz = nz
        self.delta_c = delta_c
        self.hmf_params = hmf_params or {}
        self.filter_model = filter_model
        self.filter_params = filter_params or {}
        self.ensure_convergent_upper_integral = ensure_convergent_upper_integral

    # ===========================================================================
    # PARAMETERS
    # ===========================================================================
    @parameter("res")
    def Mmin(self, val):
        r"""
        Minimum mass at which to perform analysis [units :math:`\log_{10}M_\odot h^{-1}`].

        :type: float
        """
        return val

    @parameter("res")
    def Mmax(self, val):
        r"""
        Maximum mass at which to perform analysis [units :math:`\log_{10}M_\odot h^{-1}`].

        :type: float
        """
        return val

    @parameter("res")
    def dlog10m(self, val):
        """
        log10 interval between mass bins

        :type: float
        """
        return val

    @parameter("switch")
    def ensure_convergent_upper_integral(self, val):
        """
        Whether to ensure that upper integrals (eg. ngtm or mgtm) are converged for all mass values prescribed.

        If True (which it is by default), then :attr:`upper_integral_ngtm` is added to :attr:`ngtm`, so that all values
        within the integral can be trusted to be converged (within a relative tolerance of 10^-4).
        :attr:`upper_integral_mass_limit` holds the relevant upper mass limit at
        which the integral converges to this relative tolerance *for the masses set by the user*.

        Changing `Mmax` will change the upper mass limit, as the the integral will need to be performed further to
        converge for the highest mass.
        """
        return bool(val)

    @parameter("model")
    def filter_model(self, val):
        """
        A model for the window/filter function.

        :type: :class:`hmf.filters.Filter` subclass
        """
        if not issubclass_(val, Filter) and not isinstance(val, str):
            raise ValueError("filter must be a Filter or string, got %s" % type(val))
        elif isinstance(val, str):
            return get_model_(val, "hmf.density_field.filters")
        else:
            return val

    @parameter("param")
    def filter_params(self, val):
        """
        Model parameters for `filter_model`.

        :type: dict
        """
        return val

    @parameter("param")
    def delta_c(self, val):
        r"""
        The critical overdensity for collapse, :math:`\delta_c`.

        :type: float
        """
        try:
            val = float(val)
        except ValueError:
            raise ValueError("delta_c must be a number: ", val)

        if val <= 0:
            raise ValueError("delta_c must be > 0 (", val, ")")
        if val > 10.0:
            raise ValueError("delta_c must be < 10.0 (", val, ")")

        return val

    @parameter("model")
    def hmf_model(self, val):
        r"""
        A model to use as the fitting function :math:`f(\sigma)`

        :type: str or `hmf.fitting_functions.FittingFunction` subclass
        """
        if not issubclass_(val, ff.FittingFunction) and not isinstance(val, str):
            raise ValueError("hmf_model must be a ff.FittingFunction or string, got %s" % type(val))
        elif isinstance(val, str):
            return get_model_(val, "hmf.mass_function.fitting_functions")
        else:
            return val

    @parameter("param")
    def hmf_params(self, val):
        """
        Model parameters for `hmf_model`.

        :type: dict
        """
        return val

    @parameter("model")
    def mdef_model(self, val):
        """
        A model to use as the mass definition.

        :type: str or :class:`hmf.halos.mass_definitions.MassDefinition` subclass
        """
        if not issubclass_(val, md) and not isinstance(val, str) and val is not None:
            raise ValueError("mdef_model must be a MassDefinition or string, got %s" % type(val))
        elif isinstance(val, str):
            return get_model_(val, "hmf.halos.mass_definitions")
        else:
            return val

    @parameter("param")
    def mdef_params(self, val):
        """
        Model parameters for `mdef_model`.
        :type: dict
        """
        return val

    #
    # @parameter("param")
    # def delta_h(self, val):
    #     """
    #     The overdensity for the halo definition, with respect to :attr:`delta_wrt`
    #
    #     :type: float
    #     """
    #     try:
    #         val = float(val)
    #     except ValueError:
    #         raise ValueError("delta_halo must be a number: ", val)
    #
    #     if val <= 0:
    #         raise ValueError("delta_halo must be > 0 (", val, ")")
    #     if val > 10000:
    #         raise ValueError("delta_halo must be < 10,000 (", val, ")")
    #
    #     return val
    #
    # @parameter("switch")
    # def delta_wrt(self, val):
    #     """
    #     Defines what the overdensity of a halo is with respect to, mean density
    #     of the universe, or critical density.
    #
    #     :type: str, {"mean", "crit"}
    #     """
    #     if val not in ['mean', 'crit']:
    #         raise ValueError("delta_wrt must be either 'mean' or 'crit' (", val, ")")
    #
    #     return val

    #
    # @parameter
    # def z2(self, val):
    #     if val is None:
    #         return val
    #
    #     try:
    #         val = float(val)
    #     except ValueError:
    #         raise ValueError("z must be a number (", val, ")")
    #
    #     if val <= self.z:
    #         raise ValueError("z2 must be larger than z")
    #     else:
    #         return val
    #
    # @parameter
    # def nz(self, val):
    #     if val is None:
    #         return val
    #
    #     try:
    #         val = int(val)
    #     except ValueError:
    #         raise ValueError("nz must be an integer")
    #
    #     if val < 1:
    #         raise ValueError("nz must be >= 1")
    #     else:
    #         return val

    # @parameter
    # def cut_fit(self, val):
    #     if not isinstance(val, bool):
    #         raise ValueError("cut_fit must be a bool, " + str(val))
    #     return val

    # --------------------------------  PROPERTIES ------------------------------
    @cached_quantity
    def mean_density(self):
        """
        Mean density of universe at redshift z
        """
        return self.mean_density0 * (1 + self.z) ** 3

    @cached_quantity
    def mdef(self):
        if self.mdef_model is not None:
            return self.mdef_model(self.cosmo, self.z, **self.mdef_params)
        else:
            return None

    @cached_quantity
    def hmf(self):
        """
        Instantiated model for the hmf fitting function.
        """
        return self.hmf_model(m=self.m, nu2=self.nu, z=self.z,
                              mass_definition=self.mdef, cosmo=self.cosmo,
                              delta_c=self.delta_c, n_eff=self.n_eff,
                              **self.hmf_params)

    @cached_quantity
    def filter(self):
        """
        Instantiated model for filter/window functions.
        """
        return self.filter_model(self.k, self._unnormalised_power, **self.filter_params)

    @cached_quantity
    def m(self):
        """Masses [Msun/h]"""
        return 10 ** np.arange(self.Mmin, self.Mmax, self.dlog10m)

    # @cached_quantity
    # def delta_halo(self):
    #     """ Overdensity of a halo w.r.t mean density"""
    #     if self.delta_wrt == 'mean':
    #         return self.delta_h
    #
    #     elif self.delta_wrt == 'crit':
    #         return self.delta_h / self.cosmo.Om(self.z)

    @cached_quantity
    def _unn_sigma0(self):
        """
        Unnormalised mass variance at z=0
        """
        return self.filter.sigma(self.radii)

    @cached_quantity
    def _sigma_0(self):
        r"""
        The normalised mass variance at z=0 :math:`\sigma`
        """
        return self._normalisation * self._unn_sigma0

    @cached_quantity
    def radii(self):
        """
        The radii corresponding to the masses `m`.
        """
        return self.filter.mass_to_radius(self.m, self.mean_density0)

    @cached_quantity
    def _dlnsdlnm(self):
        r"""
        The value of :math:`\left|\frac{\d \ln \sigma}{\d \ln m}\right|`, ``len=len(m)``

        Notes
        -----

        .. math:: frac{d\ln\sigma}{d\ln m} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk

        """
        return 0.5 * self.filter.dlnss_dlnm(self.radii)

    @cached_quantity
    def sigma(self):
        """
        The mass variance at `z`, ``len=len(m)``
        """
        return self._sigma_0 * self.growth_factor

    @cached_quantity
    def nu(self):
        r"""
        The parameter :math:`\nu = \left(\frac{\delta_c}{\sigma}\right)^2`, ``len=len(m)``
        """
        return (self.delta_c / self.sigma) ** 2

    @cached_quantity
    def mass_nonlinear(self):
        """
        The nonlinear mass, nu(Mstar) = 1.
        """
        if self.nu.min() > 1 or self.nu.max() < 1:
            warnings.warn("Nonlinear mass outside mass range")
            if self.nu.min() > 1:
                startr = np.log(self.radii.min())
            else:
                startr = np.log(self.radii.max())

            model = lambda lnr: (self.filter.sigma(np.exp(lnr)) * self._normalisation * self.growth_factor
                                 - self.delta_c) ** 2

            res = minimize(model, [startr, ])

            if res.success:
                r = np.exp(res.x[0])
                return self.filter.radius_to_mass(r, self.mean_density0)
            else:
                warnings.warn("Minimization failed :(")
                return 0
        else:
            nu = spline(self.nu, self.m, k=5)
            return nu(1)

    @cached_quantity
    def lnsigma(self):
        """
        Natural log of inverse mass variance, ``len=len(m)``
        """
        return np.log(1 / self.sigma)

    @cached_quantity
    def n_eff(self):
        """
        Effective spectral index at scale of halo radius, ``len=len(m)``

        Notes
        -----
        This function, and any derived quantities, can show small non-physical
        'wiggles' at the 0.1% level, if too coarse a grid in ln(k) is used. If
        applications are sensitive at this level, please use a very fine k-space
        grid.

        Uses eq. 42 in Lukic et. al 2007.
        """
        return -3.0 * (2.0 * self._dlnsdlnm + 1.0)

    @cached_quantity
    def fsigma(self):
        """
        The multiplicity function, :math:`f(\sigma)`, for `hmf_model`. ``len=len(m)``
        """
        return self.hmf.fsigma

    @cached_quantity
    def dndm(self):
        r"""
        The number density of haloes, ``len=len(m)`` [units :math:`h^4 M_\odot^{-1} Mpc^{-3}`]
        """
        # if self.z2 is None:  # #This is normally the case
        dndm = self.fsigma * self.mean_density0 * np.abs(self._dlnsdlnm) / self.m ** 2
        if isinstance(self.hmf, ff.Behroozi):
            ngtm_tinker = self._gtm(dndm)
            dndm = self.hmf._modify_dndm(self.m, dndm, self.z, ngtm_tinker)

        # Alter the mass definition
        if self.hmf.measured_mass_definition is not None:
            if self.mdef is not None:
                mnew = self.hmf.measured_mass_definition.change_definition(self.m, self.mdef)[
                    0]  # this uses NFW, but we can change that in halomod.
                spl = spline(np.log(mnew), np.log(dndm))
                spl2 = spline(self.m, mnew)
                dndm = np.exp(spl(np.log(self.m))) / spl2.derivative()(self.m)

        # else:  # #This is for a survey-volume weighted calculation
        #     raise NotImplementedError()
        #             if self.nz is None:
        #                 self.nz = 10
        #             zedges = np.linspace(self.z, self.z2, self.nz)
        #             zcentres = (zedges[:-1] + zedges[1:]) / 2
        #             dndm = np.zeros_like(zcentres)
        #             vol = np.zeros_like(zedges)
        #             vol[0] = cp.distance.comoving_volume(self.z,
        #                                         **self.cosmolopy_dict)
        #             for i, zz in enumerate(zcentres):
        #                 self.update(z=zz)
        #                 dndm[i] = self.fsigma * self.mean_dens * np.abs(self._dlnsdlnm) / self.m ** 2
        #                 if isinstance(self.hmf_model, "ff.Behroozi"):
        #                     ngtm_tinker = self._gtm(dndm[i])
        #                     dndm[i] = self.hmf_model._modify_dndm(self.m, dndm[i], self.z, ngtm_tinker)
        #
        #                 vol[i + 1] = cp.distance.comoving_volume(z=zedges[i + 1],
        #                                                 **self.cosmolopy_dict)
        #
        #             vol = vol[1:] - vol[:-1]  # Volume in shells
        #             integrand = vol * dndm[i]
        #             numerator = intg.simps(integrand, x=zcentres)
        #             denom = intg.simps(vol, zcentres)
        #             dndm = numerator / denom

        return dndm

    @cached_quantity
    def dndlnm(self):
        r"""
        The differential mass function in terms of natural log of `m`, ``len=len(m)`` [units :math:`h^3 Mpc^{-3}`]
        """
        return self.m * self.dndm

    @cached_quantity
    def dndlog10m(self):
        r"""
        The differential mass function in terms of log of `m`, ``len=len(m)`` [units :math:`h^3 Mpc^{-3}`]
        """
        return self.m * self.dndm * np.log(10)

    @cached_quantity
    def _required_upper_mass_limits(self):
        # this is kind of bad, but we need to make sure that it knows we're using dndm here
        _ = self.dndm
        return calc_required_maximum_mass(copy.deepcopy(self), mass_density=True)

    @cached_quantity
    def upper_integral_mgtm(self):
        """
        Converged mass density integral above Mmax.

        This is added to :attr:`rho_gtm` if :attr:`ensure_convergent_upper_integral` is True, to ensure that the
        density integral at the *maximum* mass is converged.
        """
        return self._required_upper_mass_limits[1]

    @cached_quantity
    def upper_integral_mass_limit(self):
        """Upper mass limit required to have a convergent upper integral"""
        return self._required_upper_mass_limits[0]

    @cached_quantity
    def upper_integral_ngtm(self):
        """
        Converged number density integral above Mmax.

        This is added to :attr:`ngtm` if :attr:`ensure_convergent_upper_integral` is True, to ensure that the
        density integral at the *maximum* mass is converged.
        """
        _ = self.dndm
        return calc_required_maximum_mass(copy.deepcopy(self))[1]

    def _gtm(self, dndm, mass_density=False):
        """
        Calculate number or mass density above mass thresholds in `m`

        This function is here, separate from the properties, due to its need
        of being passed ``dndm` in the case of the ff.Behroozi fit only, in which
        case an infinite recursion would occur otherwise.

        Parameters
        ----------
        dndm : array_like, ``len(self.m)``
            Should usually just be exactly :attr:`dndm`, except in Behroozi fit.

        mass_density : bool, ``False``
            Whether to get the mass density, or number density.
        """
        if not self.ensure_convergent_upper_integral:
            return int_gtm(self.m, dndlnm=dndm * self.m, mass_density=mass_density)
        else:
            return int_gtm(self.m, dndlnm=dndm * self.m,
                           upper_integral=self.upper_integral_mgtm if mass_density else self.upper_integral_ngtm,
                           mass_density=mass_density)

    @cached_quantity
    def ngtm(self):
        r"""
        The cumulative mass function above `m`, ``len=len(m)`` [units :math:`h^3 Mpc^{-3}`]

        If :attr:`ensure_convergent_upper_integral` is True, `ngtm` will be evaluated as the integral
        of the current data, plus the integral of the mass function above the current mass limit,
        ensuring that all values contained in `ngtm` are accurate.

        In the case of the ff.Behroozi fit, it is impossible to auto-extend the mass
        range except by the power-law fit, thus one should be careful to supply
        appropriate mass ranges in this case.
        """
        return self._gtm(self.dndm)

    @cached_quantity
    def rho_gtm(self):
        r"""
        Mass density in haloes `>m`, ``len=len(m)`` [units :math:`M_\odot h^2 Mpc^{-3}`]

        If :attr:`ensure_convergent_upper_integral` is True, `rho_gtm` will be evaluated as the integral
        of the current data, plus the integral of the mass function above the current mass limit,
        ensuring that all values contained in `rho_gtm` are accurate.

        In the case of the ff.Behroozi fit, it is impossible to auto-extend the mass
        range except by the power-law fit, thus one should be careful to supply
        appropriate mass ranges in this case.
        """
        return self._gtm(self.dndm, mass_density=True)

    @cached_quantity
    def rho_ltm(self):
        r"""
        Mass density in haloes `<m`, ``len=len(m)`` [units :math:`M_\odot h^2 Mpc^{-3}`]

        .. note :: As of v1.6.2, this assumes that the entire mass density of
                   halos is encoded by the ``mean_density0`` parameter (ie. all
                   mass is found in halos). This is not explicitly true of all
                   fitting functions (eg. Warren), in which case the definition
                   of this property is somewhat inconsistent, but will still
                   work. Note that this means it explicitly uses :func:`rho_gtm`,
                   and all caveats for that function should be considered.
        """
        return self.mean_density0 - self.rho_gtm

    @cached_quantity
    def how_big(self):
        """
        Size of simulation volume in which to expect one halo of mass m (with 95% probability), ``len=len(m)`` [units :math:`Mpch^{-1}`]
        """
        return (0.366362 / self.ngtm) ** (1. / 3.)
