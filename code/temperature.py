import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as op
import pressure as p


def interp(t0, t1, dt):
    """ Return the interpolated values of pressure for a given range of time.

                Parameters:
                -----------
                t0 : float
                    Initial time.
                t1 : float
                    Final time.
                dt : float
                    Time step size.

                Returns:
                --------
                t : array-like
                    array of times
                temp_interp : array-like
                    array of interpolated temperature values

        """

    n = int(np.ceil((t1 - t0) / dt))    # number of steps
    t = t0 + np.arange(n + 1) * dt      # time array

    # read and interpolate temperature values
    time, temp = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # temperature (gr_t data)
    temp_interp = np.interp(t, time, temp)

    return t, temp_interp

def analytical(t, q, at, bp, T0, Tc):
    """ Return the pressure value for a given time

        Parameters:
        -----------
        t : float
            Independent variable.
        q : float
            total extraction rate.
        at : float
            cold water inflow parameter
        bp : float
            pressure recharge strength parameter.
        T0 : float
            temperature of recharge source.
        Tc : float
            temperature of cold water inflow.

        Returns:
        T : float
            Pressure value for given time and parameters

        Notes:
            This function assumes a constant extraction rate therefore the slow drainage term is equal to 0
    """

    return Tc + (T0 - Tc)*math.exp(-1 * (at*q/bp)*(math.exp(-bp*t) + bp*t - 1))



def ode_model(t, temp, pr, a, b, p0, t0, tc):
    """ Return the derivative dT/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        temp : float
            Dependent variable.
        pr : float
            pressure.
        a : float
            cold water inflow parameter.
        b : float
            conduction parameter (recharge strength parameter).
        p0 : float
            pressure at recharge source.
        t0 : float
            initial value of the dependent variable.
        tc : float
            temperature of cold water inflow.

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
    # check for direction of flow
    if pr > p0:
        tempx = temp
    else:
        tempx = tc

    return -a * (pr - p0) * (tempx - temp) - b * (temp - t0)


def solve_ode(f, t0, t1, dt, x0, pr, pars):
    """Solve the temperature ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Time at solution start.
        t1 : float
            Time at solution end.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pr : array like
            fitted pressure values.
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

        The Temperature LPM is coupled to the pressure using the fitted pressure value
    """

    n = int(np.ceil((t1 - t0) / dt))    # number of steps
    ts = t0 + np.arange(n + 1) * dt     # time array
    ys = 0. * ts                        # array to store solution values
    ys[0] = x0                          # set initial value of solution array
        

    # calculate solution values using Improved Euler
    for i in range(n):
        fk = f(ts[i], ys[i], pr[i], *pars)
        fk1 = f(ts[i] + dt, ys[i] + dt * fk, pr[i], *pars)
        ys[i + 1] = ys[i] + dt * ((fk + fk1) / 2)

    # Return both arrays contained calculated values
    return ts, ys


def helper(t, dt, x0, pr, a, b, p0, temp0, tempc):
    """ A helper method for curve_fit.

        Parameters:
        -----------
        t : float
            Independent variable.
        dt : float
            Time step length.
        x0 : float
            Initial value of the dependent variable.
        pr : array like
            fitted pressure values.
        a : float
            cold water inflow parameter.
        b : float
            conduction parameter (recharge strength parameter).
        p0 : float
            hydrostatic pressure of recharge source.
        temp0 : float
            temperature of recharge source
        tempc : float
            temperature of cold water inflow.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        this method when used in combination with the lambda function allows us to make quick change to our other
        parameters and resolve for the a, b

        It also breaks up the time array into the correct format.

    """
    # converting time to correct format
    t0 = t[0]
    t1 = t[-1]

    return solve_ode(ode_model, t0, t1, dt, x0, pr, pars=[a, b, p0, temp0, tempc])[1]


def fit(t, temp, dt, x0, pr, p0, temp0, tempc):
    """ A helper method for curve_fit.

        Parameters:
        -----------
        t : float
            Independent variable.
        dt : float
            Time step length.
        x0 : float
            Initial value of the dependent variable.
        pr : array like
            fitted pressure values.
        a : float
            cold water inflow parameter.
        b : float
            conduction parameter (recharge strength parameter).
        p0 : float
            hydrostatic pressure of recharge source.
        temp0 : float
            temperature of recharge source
        tempc : float
            temperature of cold water inflow.

        Returns:
        --------
        ap : float
            Extraction strength parameter.
        bp : float
            Recharge strength parameter.
        cp : float
            Slow drainage strength parameter.

        Notes:
        ------
        The noise of the data is the how the seasons effects the temperature values.
        The covariance array from curve_fit can be used to model uncertainty
    """
    # uncertainty calculation for temperature
    sigma = [2]*len(t)

    para, cov = op.curve_fit(lambda t, a, b: helper(t, dt, x0, pr, a, b, p0, temp0, tempc),
                             xdata=t,
                             ydata=temp,
                             p0=(0.000001, 0.08),
                             sigma=sigma)

    return para, cov
