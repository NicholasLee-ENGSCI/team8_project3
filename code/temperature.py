import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as op
import pressure as p


def interp(t0, t1, dt):
    n = int(np.ceil((t1 - t0) / dt))    # number of steps
    t = t0 + np.arange(n + 1) * dt      # time array

    time, temp = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # temperature (gr_t data)
    temp_interp = np.interp(t, time, temp)

    #f, ax = plt.subplots(1)
    #ax.plot(t, temp_interp, 'bo', marker='o')

    return t, temp_interp

def analytical():
    temp = 0



def ode_model(t, temp, pr, a, b, p0, t0):
    """ Return the derivative dT/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        T : float
            Dependent variable.
        at : float
            cold water inflow parameter.
        bp : float
            cold water inflow parameter (recharge strength parameter). 
        ap : float
            cold water inflow parameter (extraction strength parameter).
        bt : float
            conduction strength parameter.
        P : float
            pressure.
        P0 : float
            initial pressure.
        Tx : float
            temperature that depends on direction of flow.
        T0 : float
            initial value of the dependent variable.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        Tx varies depending on the direction of the flow;
        Tx = T if P > P0
        Tx = Tc otherwise, where Tc is the temperature of cold water injection


        Examples:
        ---------
        >>> temperature_ode_model(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        = -3.5
    """
    # Tx is equal to T if P > P0, otherwise Tx is equal to Tcold.
    if pr > p0:
        tempx = temp
    else:
        tempx = 10  # otherwise Tx is equal to temperature of the cold water injection, 30 degrees celsius.

    # the first derivative returns dT/dt = -at*bp*(1/ap)*(Tx-T)-bt*(T-T0) where all the parameters are provided as
    # inputs
    return -a * (pr - p0) * (tempx - temp) - b * (temp - t0)


def solve_ode(f, t0, t1, dt, x0, pr, pars):
    """Solve the temperature ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t : array-like
            time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method.

        Function q(t) should be hard coded within this method. Create duplicates of
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    """

    n = int(np.ceil((t1 - t0) / dt))  # number of steps
    ts = t0 + np.arange(n + 1) * dt  # time array
    ys = 0. * ts  # array to store solution values
    ys[0] = x0  # set initial value of solution array
        

    # calculate solution values using Improved Euler
    for i in range(n):
        fk = f(ts[i], ys[i], pr[i], *pars)
        fk1 = f(ts[i] + dt, ys[i] + dt * fk, pr[i], *pars)
        ys[i + 1] = ys[i] + dt * ((fk + fk1) / 2)

    # Return both arrays contained calculated values
    return ts, ys


def helper(t, dt, x0, pr, a, b, p0, temp0):

    t0 = t[0]
    t1 = t[-1]

    return solve_ode(ode_model, t0, t1, dt, x0, pr, pars=[a, b, p0, temp0])[1]


def fit(t, temp, dt, x0, pr, p0, temp0):
    sigma = [2]*len(t)

    para, cov = op.curve_fit(lambda t, a, b: helper(t, dt, x0, pr, a, b, p0, temp0), xdata=t, ydata=temp, p0=(0.000001, 0.08), sigma=sigma)

    return para, cov
