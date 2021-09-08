import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as op


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

            pr : array-like
                array of interpolated and converted pressure values

            Notes:
            ------
            for the specfic case where dt = 1, we have approximated the uncertain data using a circle function

            **I can solve for and implement a circle function for different dt values of dt we just need to decide what
            the best value is first. difference size of dt produce different functions and coding in a circel solving function
            given 3 points is just too much effort

            Examples:
            ---------
            >>> pressure_ode_model(0, 1, 2, 3, 4, 5, 6, 7)
            = -12
    """
    n = int(np.ceil((t1 - t0) / dt))  # number of steps
    t = t0 + np.arange(n + 1) * dt  # time array

    # reading water level data from text file
    time, water_level = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=1).T  # water level (gr_p data)

    # interpolating water level values to the correct times
    water_interp = np.interp(t, time, water_level)

    # Calculation of water lvl from 1850 - 1875   **This is a bit messy for now but leave like this
    # Ratouis2017, figure 16, gives us historical water data
    # Can be approximated to a circle function (x + 2.0211)²  +  (y - 11.518)²  =  8.1733e+4
    # Calculated using http://www.1728.org/circle2.htm, parameters {0, 297.4; 30, 296.9; 69,295} respectively
    # math.sqrt(1.1631150e+6 - (i + 2.9704) ** 2) - 781.074     for 0.5 step
    #  (x + 2.0211)²  +  (y - 11.518)²  =  8.1733e+4  for 1 step?
    if (dt == 1):
        for i in range(0, 34):
            water_interp[i] = math.sqrt(8.1733e+4 - (i + 2.0211) ** 2) + 11.518

    # f, ax = plt.subplots(1)
    # ax.plot(t, water_interp, 'bo', marker='o')

    # Conversion of water level to pressure
    pr = ((water_interp - 297.4) * 997 * 9.81) + 5000

    return t, pr


def interpolate_q_total(t):
    """ interplate two extraction rates to find the total extraction rate, q.

        Parameters:
        -----------
        t : float
            Independent variable.

        Returns:
        --------
        q : float
            total extraction rate.

        Notes:
        ------
        tq1 is the total extraction rate over the rotrua region including the rhyolite formation
        tq2 is only the rhyolite formation

        there's something about the geography that were meant to realise and make an assumption
        Read ratious2017 pg 171
    """
    # read extraction rate data
    tq1, pr1 = np.genfromtxt('gr_q1.txt', delimiter=',', skip_header=1).T  # production rate 1 (gr_q1 data)
    tq2, pr2 = np.genfromtxt('gr_q2.txt', delimiter=',', skip_header=1).T  # production rate 2 (gr_q2 data)

    # we need to decide how we calculate q and what falls in our zone, I remember something in the first few lectures
    # about this, 2D blocking maybe??
    ex1 = np.interp(t, tq1, pr1)
    ex2 = np.interp(t, tq2, pr2)

    # calculation of reinjection rate
    ex_final = ex1
    ex_final[34:] = ex_final[34:] - 1500  # 1500 1985
    ex_final[41:] = ex_final[41:] - 3800  # 5300 1992
    ex_final[50:] = ex_final[50:] - 2200  # 7500 2001

    return ex_final


def ode_model(t, pr, q, dqdt, a, b, c, p0):
    """ Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        pr : float
            Dependent variable.
        q : float
            relative extraction rate.
        dqdt : float
            rate of change of relative extraction rate.
        a : float
            extraction strength parameter.
        b : float
            recharge strength parameter.
        c : float
            slow drainage strength parameter.
        p0 : float
            hydrostatic pressure value of the source of reacharge.

        Returns:
        --------
        dpdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        q = {qtotal,qrhyolite,qnotrhyolite}
        q is found by using interpolate_q_total(t):
        - 1. interpolating extraction rate 1 and extraction rate 2 to independent variables values that corresponds to input variable t.
        - 2. summing the two interpolated data.

        Examples:
        ---------
        >>> pressure_ode_model(0, 1, 2, 3, 4, 5, 6, 7)
        = -12
    """

    # the first derivative returns dP/dt = -a*q-b*(P-P0)-c*dqdt where all the parameters are provided as inputs
    return -a * q - b * (pr - p0) - c * dqdt


def solve_ode(f, t0, t1, dt, x0, indicator, pars):
    """ Solve the pressure ODE numerically.

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
        indicator: string
            string that describes the future operation of production. PLEASE REFER TO NOTES TO MORE INFORMATION.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        ts : array-like
            Independent variable solution vector.
        ys : array-like
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

        if indicator is:
            'SAME' => No extrapolation is wanted or production is maintained from year 2014; there is no change in production rate from 2014
            'STOP' => production is stopped from year 2014; production rate = 0 from q[65]
            'DOUBLE' => production is doubled from year 2014
            'HALF' => production is halved from year 2014
    """

    n = int(np.ceil((t1 - t0) / dt))  # number of steps
    ts = t0 + np.arange(n + 1) * dt  # time array
    ys = 0. * ts  # initialise solution array
    ys[0] = x0  # set initial value of solution array

    q = interpolate_q_total(ts)
    q = q / 86.4  # 1 kg/s is equivalent to 86.4 tonnes/day

    # total extraction is found by interpolating two extraction rates given and summing them (done using the
    # interpolate_q_total() function)
    if indicator == 'SAME':
        temp = 0

    elif indicator == 'STOP':
        # q[65:101] = 0
        a = q[64] / 36
        for i in range(65, 101):
            q[i] = q[64] - a * (i - 65)

    elif indicator == 'DOUBLE':
        # q[65:101] = q[64]*2
        a = q[64] / 36
        for i in range(65, 101):
            q[i] = q[64] + a * (i - 65)

    elif indicator == 'HALF':
        # q[65:101] = q[64]/2
        a = q[64] / 72
        for i in range(65, 101):
            q[i] = q[64] - a * (i - 65)

    dqdt = np.gradient(q)

    # calculate solution values using Improved Euler
    # #{are we using ambient or initial pressure to calculate rate of change???}
    for i in range(n):
        fk = f(ts[i], ys[i], q[i], dqdt[i], *pars)
        fk1 = f(ts[i] + dt, ys[i] + dt * fk, q[i], dqdt[i], *pars)
        ys[i + 1] = ys[i] + dt * ((fk + fk1) / 2)

    # Return both arrays contained calculated values
    return ts, ys


def helper(t, dt, x0, indicator, a, b, c, p0):
    """ A helper method for curve_fit.

        Parameters:
        -----------
        t : float
            Independent variable.
        dt : float
            Time step length.
        x0 : float
            Initial value of the dependent variable.
        indicator: string
            String that describes the future operation of production. 
        a : float
            Extraction strength parameter.
        b : float
            Recharge strength parameter.
        c : float
            Slow drainage strength parameter.
        p0 : float
            hHdrostatic pressure value of recharge source.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
            this method when used in combination with the lambda function allows us to make quick change to our other
            parameters and resolve for the a, b, c.

            It also break up the time array into the correct format.

    """
    t0 = t[0]
    t1 = t[-1]

    return solve_ode(ode_model, t0, t1, dt, x0, indicator, pars=[a, b, c, p0])[1]


def fit(t, wp, dt, x0, p0):
    """ A helper method for curve_fit.

            Parameters:
            -----------
            t : float
                Independent variable.
            wp : float
                Dependent variable.
            dt : float
                Time step length.
            x0 : float
                Initial pressure value.
            p0 : float
                Hydrostatic pressure value of recharge source.

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
            This is still a very basic implementation of curve fitting, Later im going to account for unncertainty and
            use constant parameter values to solve for the other so locking a to a specific value to solve for b

            the noise of the data is the how rainfall effects to the water levels which in turn effects to
            pressure readings

            no idea about covariance atm

    """
    # para, _ = op.curve_fit(lambda t, a, b, c: helper(t, dt, x0, 'SAME', a, b, c, p0), xdata=t,
    #                        ydata=wp, p0=[0.15, 0.12, 0.6],
    #                        bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)))

    # estimation of anual rainfall taking from ratoius2017 then converted to pressure change
    sigma = [0.8 * 997 * 9.81] * len(t)

    para, cov = op.curve_fit(lambda t, a, b, c: helper(t, dt, x0, 'SAME', a, b, c, p0), xdata=t,
                             ydata=wp, p0=[0.15, 0.12, 0.6],
                             bounds=((0, 0, -np.inf), (np.inf, np.inf, np.inf)),
                             sigma=sigma)

    print(para)  # for testing
    a = para[0]
    b = para[1]
    c = para[2]

    return para, cov


def production_scenarios(t0, t1, dt):
    '''
    This function plots all different production scenarios for comparison


    '''
    # plotting format
    f, ax1 = plt.subplots(nrows=1, ncols=1)

    # plotting no change
    n = int(np.ceil((t1 - t0) / dt))  # number of steps
    t = t0 + np.arange(n + 1) * dt  # time array

    q0 = interpolate_q_total(t)
    q0 = q0 / 86.4

    # operation terminated
    q1 = interpolate_q_total(t)
    q1 = q1 / 86.4
    a = q1[64] / 36
    for i in range(65, 101):
        q1[i] = q1[64] - a * (i - 65)

    # double production
    q2 = interpolate_q_total(t)
    q2 = q2 / 86.4
    for i in range(65, 101):
        q2[i] = q2[64] + a * (i - 65)

    # half production
    q3 = interpolate_q_total(t)
    q3 = q3 / 86.4
    b = q3[64] / 72
    for i in range(65, 101):
        q3[i] = q3[64] - b * (i - 65)

    ln1 = ax1.plot(t, q0, 'k-', label='maintained production')
    ln2 = ax1.plot(t, q1, 'r-', label='operation terminated')
    ln3 = ax1.plot(t, q2, 'g-', label='production doubled')
    ln4 = ax1.plot(t, q3, 'b-', label='production halved')

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=1)
    ax1.set_ylabel('Production [tonnes/day???]')
    ax1.set_xlabel('time [yr]')
    ax1.set_title('different production scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('production_scenarios.png', dpi=300)
    return


def forecast(t0, t1, dt, x0, a, b, c, p0):
    ''' This function is to extrapolate to year 2050, then plot it

        Parameters:
        -----------
        t0 : float
            Time at start.
        t1 : float
            Time at end.
        dt : float
            Time step length.
        x0 : float
            Initial pressure value.
        a : float
            extraction strength parameter.
        b : float
            recharge strength parameter.
        c : float
            slow drainage strength parameter.
        p0 : float
            hydrostatic pressure value of the source of reacharge.

        Notes:
        --------
        plots some things

        need to parse the parameters incase we change our model we want the plots to reflect our model
    '''
    # plotting format
    f, ax1 = plt.subplots(nrows=1, ncols=1)

    # calculating the different scenarios
    t, y_no_change = solve_ode(ode_model, t0, t1, dt, x0, 'SAME', pars=[a, b, c, p0])
    y_stop = solve_ode(ode_model, t0, t1, dt, x0, 'STOP', pars=[a, b, c, p0])[1]
    y_double = solve_ode(ode_model, t0, t1, dt, x0, 'DOUBLE', pars=[a, b, c, p0])[1]
    y_half = solve_ode(ode_model, t0, t1, dt, x0, 'HALF', pars=[a, b, c, p0])[1]

    # plotting the different scenarios against each other
    ln1 = ax1.plot(t, y_no_change / 10 ** 5, 'k-', label='maintained production')
    ln2 = ax1.plot(t, y_stop / 10 ** 5, 'r-', label='operation terminated')
    ln3 = ax1.plot(t, y_double / 10 ** 5, 'g-', label='production doubled')
    ln4 = ax1.plot(t, y_half / 10 ** 5, 'b-', label='production halved')

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=2)
    ax1.set_ylabel('Pressure [MPa]')
    ax1.set_xlabel('time [yr]')
    ax1.set_title('Pressure predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast.png', dpi=300)
    return
