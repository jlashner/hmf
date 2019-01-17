import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from hmf.helpers.sample import sample_mf, dndm_from_sample


def test_circular():
    np.random.seed(1234)
    m, h = sample_mf(N=1e5, log_mmin=11, log_mmax=15, transfer_model="EH")

    print(m.min(), m.max())
    centres, hist = dndm_from_sample(m, 1e5 / h.ngtm[0])

    s = spline(np.log10(h.m), np.log10(h.dndm))

    print(hist, 10 ** s(centres))
    assert np.allclose(hist, 10 ** s(centres), rtol=0.05)


def test_mmax_big():
    # raises ValueError because ngtm=0 exactly at m=18
    # due to hard limit of integration in integrate_hmf.
    np.random.seed(12345)

    m, h = sample_mf(N=1e5, log_mmin=11, log_mmax=18, transfer_model="EH")

    centres, hist = dndm_from_sample(m, 1e5 / h.ngtm[0])

    s = spline(np.log10(h.m[h.dndm > 0]), np.log10(h.dndm[h.dndm > 0]))

    print(hist, 10 ** s(centres))
    assert np.allclose(hist, 10 ** s(centres), rtol=0.05)
