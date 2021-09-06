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
        tempx = 30  # otherwise Tx is equal to temperature of the cold water injection, 30 degrees celsius.

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
    para, _ = op.curve_fit(lambda t, a, b: helper(t, dt, x0, pr, a, b, p0, temp0), xdata=t, ydata=temp, p0=(0.000001, 0.08))

    print(para)  # for testing
    a = para[0]
    b = para[1]

    return a, b

def forecast(time0, t1, dt, x0, t, pr1, pr2, pr3, pr4, a, b, p0, t0):
    '''
    time0 = ...
    .
    .
    .
    t = array 
        array of time values corresponding to the extrapolated pressure values
    pr1 = array
        array of "no change" extrapolation pressure value 
    pr2 = array
        array of "no production" extrapolation pressure value 
    pr3 = array
        array of "double production" extrapolation pressure value 
    pr4 = array
        array of "half production" extrapolation pressure value 
    .
    .
    .
    '''
    # plotting format
    f, ax1 = plt.subplots(nrows=1, ncols=1)

    n = int(np.ceil((t1 - time0) / dt))  # number of steps
    ts = time0 + np.arange(n + 1) * dt  # time array

    p_no_change = np.interp(ts,t,pr1)
    p_no_prod = np.interp(ts,t,pr2)
    p_double_prod = np.interp(ts,t,pr3)
    p_half_prod = np.interp(ts,t,pr4)

    tx, t_no_change  = solve_ode(ode_model, time0, t1, dt, x0, p_no_change, pars=[a, b, p0, t0])
    t_no_prod  = solve_ode(ode_model, time0, t1, dt, x0, p_no_prod, pars=[a, b, p0, t0])[1]
    t_double_prod  = solve_ode(ode_model, time0, t1, dt, x0, p_double_prod, pars=[a, b, p0, t0])[1]
    t_half_prod  = solve_ode(ode_model, time0, t1, dt, x0, p_half_prod, pars=[a, b, p0, t0])[1]

    # plotting the different scenarios against each other
    ln1 = ax1.plot(tx, t_no_change, 'k--o', label='maintained production')
    ln2 = ax1.plot(tx, t_no_prod, 'r*', label='operation terminated')
    ln3 = ax1.plot(tx, t_double_prod, 'g.', label='production doubled')
    ln4 = ax1.plot(tx, t_half_prod, 'b-', label='production halved')

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=3)
    ax1.set_ylabel('temperature [degC]')
    ax1.set_xlabel('time [yr]')
    ax1.set_title('Temperature predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_forecast.png', dpi=300)
    return