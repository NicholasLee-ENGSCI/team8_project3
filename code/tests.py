import unittest
import pressure as p
import temperature as t
import numpy as np
import math as math


def find_analytic_pressure(t, ap, bp, cp, p0):
    '''
    compares analytic solution with model

    for the analytical solution there are assumptions we make:
    - rate of change of q will be constant
    - slow drainage term c is ignored
    '''
    pAnalytic = np.zeros(len(t))
    q = p.interpolate_q_total(t)
    dqdt = np.gradient(q)
    for i in range(len(t)):
        pAnalytic[i] = p0 - (ap*q[i])*(1/bp)*(1-math.exp(-bp*t[i]))
    return pAnalytic


def find_analytic_temp(t, T0, Tcold, a, b, q):
    return


def benchmarking(t, dt, x0, indicator, ap, bp, cp, p0):

    t1, p1 = p.solve_ode(p.ode_model, t[0], t[-1], dt, x0, 'SAME', pars=[ap,bp,cp, p0])
    p2 = find_analytic_pressure(t, ap, bp, cp, p0)
    #     # store 1/h in the t array
    #     t_values.append(1 / h)
    #     # store the pressure at year 2014 in the y-array
    #     p_values.append(p1[-1])
    return p1, p2

class TestPressure(unittest.TestCase):
    def test_calculate_pressure(self):
        self.assertEqual(True, False)  # add assertion here

    def test_interpolate_q_total(self):
        self.assertEqual(True, False)  # add assertion here

    def test_pressure_ode_model(self):
        self.assertEqual(True, False)  # add assertion here

    def test_solve_pressure_ode(self):
        self.assertEqual(True, False)  # add assertion here

    def test_fit_pressure(self):
        self.assertEqual(True, False)  # add assertion here


class TestTemperature(unittest.TestCase):
    def test_calculate_temperature(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
