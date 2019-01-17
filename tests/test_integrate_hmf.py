import numpy as np
from mpmath import gammainc as _mp_ginc

from hmf.mass_function.integrate_hmf import hmf_integral_gtm


def _flt(a):
    try:
        return a.astype('float')
    except AttributeError:
        return float(a)


_ginc_ufunc = np.frompyfunc(lambda z, x: _mp_ginc(z, x), 2, 1)


def gammainc(z, x):
    return _flt(_ginc_ufunc(z, x))


class TestAnalyticIntegral(object):
    @staticmethod
    def tggd(m, loghs, alpha, beta):
        return beta * (m / 10 ** loghs) ** alpha * np.exp(-(m / 10 ** loghs) ** beta)

    @staticmethod
    def anl_int(m, loghs, alpha, beta):
        return 10 ** loghs * gammainc((alpha + 1) / beta, (m / 10 ** loghs) ** beta)

    def test_basic(self):
        m = np.logspace(10, 18, 500)
        dndm = self.tggd(m, 14.0, -1.9, 0.8)
        ngtm = self.anl_int(m, 14.0, -1.9, 0.8)

        full_numerical_ngtm = hmf_integral_gtm(m, dndm)

        print(ngtm[m<1e15] / full_numerical_ngtm[m<1e15])
        # Test integrating the whole integral at once, but only comparing
        # smallish values
        assert np.allclose(ngtm[m<1e15], full_numerical_ngtm[m<1e15], atol=0, rtol=0.03)

        # more like what actually happens in hmf:
        upper_integral = self.anl_int(m[m<1e15][-1], 14.0, -1.9, 0.8)

        full_numerical_ngtm = hmf_integral_gtm(m[m<1e15], dndm[m<1e15]) + upper_integral

        print(ngtm[m<1e15] / full_numerical_ngtm)
        assert np.allclose(ngtm[m < 1e15], full_numerical_ngtm, atol=0, rtol=0.03)

    def test_high_z(self):
        m = np.logspace(10, 18, 500)
        dndm = self.tggd(m, 9.0, -1.93, 0.4)
        ngtm = self.anl_int(m, 9.0, -1.93, 0.4)

        print(ngtm / hmf_integral_gtm(m, dndm))
        assert np.allclose(ngtm, hmf_integral_gtm(m, dndlnm=m * dndm), rtol=0.03)

    def test_low_mmax_z0(self):
        m = np.logspace(10,15,500)
        dndm = self.tggd(m,14.0,-1.9,0.8)
        ngtm = self.anl_int(m,14.0,-1.9,0.8)

        upper_integral = self.anl_int(1e15, 14.0, -1.9, 0.8)
        num_intg = hmf_integral_gtm(m,dndm, upper_integral=upper_integral)
        print(ngtm/num_intg)
        assert np.allclose(ngtm, num_intg, atol=0, rtol=0.03)

    def test_low_mmax_high_z(self):
        m = np.logspace(10, 15, 500)
        dndm = self.tggd(m, 9.0, -1.93, 0.4)
        ngtm = self.anl_int(m, 9.0, -1.93, 0.4)

        print(ngtm / hmf_integral_gtm(m, dndm))
        assert np.allclose(ngtm, hmf_integral_gtm(m, dndm), rtol=0.03)
