import unittest
import pressure as p
import temperature as t
import numpy as np
import math as math


class TestPressure(unittest.TestCase):
    def test_interp(self):
        a, b = p.interp(1950, 2014, 10, False)

        np.testing.assert_array_equal(a, [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])
        np.testing.assert_array_almost_equal(b, [-20556.62941, -20556.62941, -20556.62941, -20556.62941, -4865.50652895, 2603.76035, 3093.87558, 598.7435])



    def test_interpolate_q_total(self):
        n = int(np.ceil(2014 - 1950)/10)  # number of steps
        ts = 1950 + np.arange(n + 1)*10  # time array
        q = p.interpolate_q_total(ts)

        np.testing.assert_allclose(q, [0.770137,  27.331150, 126.954041, 210.248070, 119.987037, 116.290509, 112.031212], atol=1e-6)


    def test_analytical(self):
        a = p.analytical(t=2, q=4, a=1, b=1, c=0, p0=3)

        np.testing.assert_allclose(a, -0.4586588670535492, atol=1e-6)

    def test_ode_model(self):
        a = p.ode_model(t=4, pr=3, q=4, dqdt=1, a=1, b=0.5, c=3, p0=4)

        np.testing.assert_array_almost_equal(a, -6.5, atol=1e-6)

    def test_solve_ode(self):
        a, b = p.solve_ode(f=p.ode_model, t0= 1960, t1=2000, dt=10, x0=5, indicator="SAME", pars= [1, 0.5, 1, 4])

        np.testing.assert_allclose(a, [1960, 1970, 1980, 1990, 2000], atol=1e-6)
        np.testing.assert_allclose(b, [5, 1916.81062, 19539.0778, 169153.630, 1438870.98], atol=1e-6)


    def test_helper(self):
        a = p.helper(t=[1970, 2000], dt=15, x0= 1, indicator="same", a=0.5, b=1, c=1.5, p0=4)

        np.testing.assert_allclose(a, [1.000000e+00, 3.814192e+04, 3.772747e+06], atol=1e-3)

    def test_fit(self):
        a, _ = p.fit(t= [1960, 1970, 1980, 1990], wp=[30, 40, 50, 60], dt=10, x0=25, p0=10, flag=False)

        np.testing.assert_allclose(a, [1.138981e-10, 2.434981e-01], atol=1e-6)

class TestTemperature(unittest.TestCase):
    def test_interp(self):
        a, b = t.interp(t0=1970, t1=2030, dt=15)

        np.testing.assert_allclose(a, [1970, 1985, 2000, 2015, 2030], atol=1e-6)
        np.testing.assert_allclose(b, [147., 136., 138., 144., 144.], atol=1e-6)

    def test_analytical(self):
        a = t.analytical(t=5, q=3, at=0.5, bp=1, T0=2, Tc=1)

        np.testing.assert_allclose(a, 1.002454, atol=1e-6)

    def test_ode_model(self):
        a = t.ode_model(t= 4, temp=20, pr=10, a=0.2, b=0.5, p0=20, t0=10, tc=2)

        np.testing.assert_allclose(a, -41, atol=1e-6)

    def test_solve_ode(self):
        a, b = t.solve_ode(f=t.ode_model, t0=1960, t1=2010, dt=10, x0=4, pr=[10, 9, 8, 5, 10], pars=[0.0001, 0.5, 2, 3, 0.5])

        np.testing.assert_allclose(a, [1960, 1970, 1980, 1990,2000, 2010], atol=1e-6)
        np.testing.assert_allclose(b, [4, 11.5, 7.525000e+01, 6.171250e+02, 5.223062e+03, 4.437353e+04], atol=1e-6)

    def test_helper(self):
        a = t.helper(t=[1960, 2010], dt=10, x0=3, pr=[6, 2, 4, 5, 7], a=0.00001, b=0.5, p0=4, temp0=3, tempc=3)

        np.testing.assert_allclose(a, [3., 3., 3., 3., 3., 3.], atol=1e-6)

    def test_fit(self):
        a, _ = t.fit(t= [1950, 1960, 1970, 1980, 1990, 2000], temp=[5, 7, 9, 6, 8, 9], dt=10, x0=4, pr=[10, 73, 96, 68, 84, 93], p0=7, temp0=3, tempc=2)

        np.testing.assert_allclose(a, [ 1.000000e-06, -3.898198e-02], atol=1e-6)





if __name__ == '__main__':
    unittest.main()
