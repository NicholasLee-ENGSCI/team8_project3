# Team 8 Project 3
# Python code for model fitting 


# imports
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import optimize as op

def calculate_pressure(t, initial):
    time_pressure, water_level = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=1).T  # water level (gr_p data)

    # interpolating water level values to the correct time
    water_interp = np.interp(t, time_pressure, water_level)

    # Calculation of water lvl from 1850 - 1875
    # Ratouis2017, figure 16, gives us historical water data
    # Can be approximated to a circle function (x - 1.068182)²  +  (y - 104.7955)²  =  3.732914e+4
    # Calculated using http://www.1728.org/circle2.htm, parameters {0, 298; 15, 297.5; 35,295} respectively
    for i in range(0, 34):
        water_interp[i] = math.sqrt(37329 - (i + 1.068182)**2) + 104.7955

    # Believe this is were mistake is, conversion of water level to pressure then scaling to MPA
    # pressure = ((water_interp * 997 * 9.81) - 2909250)
    pressure = (water_interp-500)*997*9.81
    # pressure = water_interp

    return pressure #+ 1900000


def interpolate_q_total(t):
    ''' interplate two extraction rates to find the total extraction rate, q.

        Parameters:
        -----------
        t : float
            Independent variable.

        Returns:
        --------
        q : float
            total extraction rate.
    '''
    # read extraction rate data 
    tq1, pr1 = np.genfromtxt('gr_q1.txt', delimiter=',', skip_header=1).T  # production rate 1 (gr_q1 data)
    tq2, pr2 = np.genfromtxt('gr_q2.txt', delimiter=',', skip_header=1).T  # production rate 2 (gr_q2 data)

    # total extraction rate is found by summing the two extraction rates given
    # interpolating two data, then summed to find total q
    ex1 = np.interp(t, tq1, pr1)
    ex2 = np.interp(t, tq2, pr2)

    return ex1 + ex2


def pressure_ode_model(t, P, P0, q, dqdt, ap, bp, cp):
    ''' Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        P : float
            Dependent variable.
        P0 : float
            Ambient value of the dependent variable.
        q : float
            total extraction rate.
        dqdt : float
            rate of change of total extraction rate. 
        ap : float
            extraction strength parameter.
        bb : float
            recharge strength parameter.
        cp : float
            slow drainage strength parameter. 
        
        Returns:
        --------
        dxdt : float
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
    '''

    # the first derivative returns dP/dt = -a*q-b*(P-P0)-c*dqdt where all the parameters are provided as inputs 
    return -ap * q - bp * (P - P0) - cp * dqdt


def solve_pressure_ode(f, t, dt, P0, indicator, pars):
    '''Solve the pressure ODE numerically.
        
        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t : array-like
            time of solution.
        dt : float
            Time step length.
        P0 : float
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
    '''
    q = interpolate_q_total(t)
    # q = q / 86.4

    # total extraction is found by interpolating two extraction rates given and summing them (done using the interpolate_q_total() function)     
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

    nt = int(np.ceil((t[-1] - t[0]) / dt))  # calculating number of steps
    ts = t[0] + np.arange(nt + 1) * dt  # initilaise time array
    ys = 0. * ts  # initialise solution array
    ys[0] = P0  # set initial value of solution array

    # calculate solution values using Improved Euler                                                                                                                         #{are we using ambient or initial pressure to calcualte rate of change???}
    for i in range(nt):
        fk = f(ts[i], ys[i], ys[i - 1], q[i], dqdt[i], *pars)
        fk1 = f(ts[i] + dt, ys[i] + dt * fk, ys[i - 1], q[i], dqdt[i], *pars)
        ys[i + 1] = ys[i] + dt * ((fk + fk1) / 2)

    # Return both arrays contained calculated values
    return ts, ys


def fit_pressure(t, dt, P0, indicator, ap, bp, cp):
    ''' Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        dt : float
            Time step length.
        P0 : float
            Ambient value of the dependent variable.
        indicator: string
            string that describes the future operation of production. PLEASE REFER TO NOTES TO MORE INFORMATION. 
        ap : float
            extraction strength parameter.
        bp : float
            recharge strength parameter.
        cp : float
            slow drainage strength parameter. 
        
        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
            Helper method to solve for Pressure values

        Examples:
        ---------

    '''
    return solve_pressure_ode(pressure_ode_model, t, dt, P0, indicator, pars=[ap, bp, cp])[1]

def plot_pressure():
    fig, ax1 = plt.subplots(1)
    ax2 = ax1.twinx()

    time = np.linspace(1950, 2014, 65)
    water_pressure = calculate_pressure(t, 300)  # remember to implement initial value later (300)

    ax1.plot(t, water_pressure, 'bo', marker='o')

    dt = 1  # step size
    x0 = water_pressure[0]  # starting pressure value (change this to a researched value later)

    para_pressure, _ = op.curve_fit(lambda t, ap, bp, cp: fit_pressure(t, dt, x0, 'SAME', ap, bp, cp), xdata=time,
                                   ydata=water_pressure, p0=[0.15, 0.12, 0.6])
    print(para_pressure)
    ap = para_pressure[0]
    bp = para_pressure[1]
    cp = para_pressure[2]

    time_fit, pressure_fit = solve_pressure_ode(pressure_ode_model, time, dt, x0, 'SAME', pars=[ap, bp, cp])
    ax1.plot(time_fit, pressure_fit, 'b--')

    # plt.show() for testing

def plot_pressure_model(t, y):
    '''
    '''
    # plotting format 
    f, ax1 = plt.subplots(nrows=1, ncols=1)
    ln1 = ax1.plot(t, y, 'k-', label='dq/dt')

    ax1.set_ylabel('rate change of production rate [tonnes/day^2]')
    ax1.set_xlabel('time [yr]')
    ax1.set_title('Rate of change of total extraction rate')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('Rate_of_change_of_total_extraction_rate.png', dpi=300)
    return


def temperature_ode_model(t, T, T0, P, P0, alpha, bt):
    ''' Return the derivative dT/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        T : float
            Dependent variable.
        at : float
            cold water inflow parameter.
        bp : float
            cold water inflow parameter.
        ap : float
            cold water inflow parameter.
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
    '''
    # Tx is equal to T if P > P0, otherwise Tx is equal to Tcold. 
    if P > P0:
        Tx = T
    else:
        Tx = 30  # otherwise Tx is equal to temperature of the cold water injection, 30 degrees celsius.

    # the first derivative returns dT/dt = -at*bp*(1/ap)*(Tx-T)-bt*(T-T0) where all the parameters are provided as inputs 
    return -alpha * (P - P0) * (Tx - T) - bt * (T - T0)


def solve_temperature_ode(f, t, dt, T0, P, pars):
    '''Solve the temperature ODE numerically.
        
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
    '''

    nt = int(np.ceil((t[-1] - t[0]) / dt))  # number of steps
    ts = t[0] + np.arange(nt + 1) * dt  # x/t array
    ys = 0. * ts  # array to store solution values
    ys[0] = T0  # set initial value of solution array

    # calculate solution values using Improved Euler
    for i in range(nt):
        fk = f(ts[i], ys[i], ys[i - 1], P[i], P[i - 1], *pars)
        fk1 = f(ts[i] + dt, ys[i] + dt * fk, ys[i - 1], P[i], P[i - 1], *pars)
        ys[i + 1] = ys[i] + dt * ((fk + fk1) / 2)

    # Return both arrays contained calculated values
    return ts, ys


def fit_temperature(t, dt, T0, P, alpha, bt):
    return solve_temperature_ode(temperature_ode_model, t, dt, T0, P, pars=[alpha, bt])[1]

def plot_temperature():
    time, temp = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # Temperature (gr_T data)
    ax2.plot(tT, temp, 'ro', marker='o')

    time_temp = np.interp(t, tT, temp)
    x0 = 149  # starting temperature value
    alpha = 0
    # at = 0.000005
    # bt = 0.065

    # paraT,_ = op.curve_fit(lambda t, alpha, bt: fit_temperature(t, dt, x0, pressurei, alpha, bt), xdata=t, ydata=tTemp)
    # print(paraT)
    # at = 0.1
    # bt = 0

    # alpha = paraT[0]
    # bt = paraT[1]

    # timei, tempi = solve_temperature_ode(temperature_ode_model, t, dt, x0, pressurei, pars=[alpha, bt])
    # ax2.plot(timei, tempi,'r--')

    plt.show()


def production_scenarios():
    '''
    This function plots all different production scenarios for comparison
    '''
    # plotting format 
    f, ax1 = plt.subplots(nrows=1, ncols=1)
    # plotting no change 
    t = np.linspace(1950, 2050, 101)
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


def pressure_forecast():
    '''
    This function is to extrapolate to year 2050, then plot it 
    '''
    # plotting format 
    f, ax1 = plt.subplots(nrows=1, ncols=1)
    # plotting no change 
    t = np.linspace(1950, 2050, 101)
    y_no_change = solve_pressure_ode(pressure_ode_model, t, dt, x0, 'SAME', pars=[ap, bp, cp])[1]
    # plotting stopped production
    y_stop = solve_pressure_ode(pressure_ode_model, t, dt, x0, 'STOP', pars=[ap, bp, cp])[1]
    # plotting double production
    y_double = solve_pressure_ode(pressure_ode_model, t, dt, x0, 'DOUBLE', pars=[ap, bp, cp])[1]
    # plotting half production
    y_half = solve_pressure_ode(pressure_ode_model, t, dt, x0, 'HALF', pars=[ap, bp, cp])[1]
    ln1 = ax1.plot(t, y_no_change / 10 ** 6, 'k-', label='maintained production')
    ln2 = ax1.plot(t, y_stop / 10 ** 6, 'r-', label='operation terminated')
    ln3 = ax1.plot(t, y_double / 10 ** 6, 'g-', label='production doubled')
    ln4 = ax1.plot(t, y_half / 10 ** 6, 'b-', label='production halved')

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


def find_analytic_pressure(t, x0, ap, bp, cp):
    '''
    compares analytic solution with model
    '''

    pAnalytic = x0 - ((ap * cp) / (bp)) * (1 - np.exp(-bp * (t))) + cp
    return pAnalytic


def find_analytic_temp(t, T0, Tcold, a, b, q):
    return


def convergence_analysis(t, x0, ap, bp, cp):
    f, ax = plt.subplots(1, 1)
    h_array = np.linspace(1, 2.7, 50)
    t_values = []
    p_values = []

    # loop through each h-value
    for h in h_array:
        # solve using numerical method for each step size
        t1, p1 = solve_pressure_ode(pressure_ode_model, t, h, x0, 'SAME', pars=[ap, bp, cp])
        # store 1/h in the t array
        t_values.append(1 / h)
        # store the pressure at year 2014 in the y-array
        p_values.append(p1[-1])

    # plot 1/h against the population after 100 years for each step size value
    ax.plot(t_values, p_values, 'rx')
    ax.set_title('Convergence Analysis of Waikite Geyser Recovery Model')
    ax.set_xlabel('1/time step')
    ax.set_ylabel('Difference in final pressure value')
    plt.show()

    return


if __name__ == "__main__":
    # read in water level andd time
    # tp,wl = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T

    # qtot = interpolate_q_total(tp)
    # total extraction rate
    # plot_pressure_model(tp,qtot)
    # total rate of change of extraction rate
    # plot_pressure_model(tp,dqdt_function(tp,qtot))

    # plot analytic vs numeric solution
    x0 = 160000

    fig, ax3 = plt.subplots(1)
    # ax3.plot(timei, pressurei, 'r-', label='Model')
    pressureA = find_analytic_pressure(timei, x0, ap, bp, cp)
    # ax3.plot(timei, pressureA, 'b-', label='Analytic')

    plt.show()

    # plot convergence
    x0 = 160000

    # convergence_analysis(t, x0, ap, bp, cp)

    # pressure_forecast()
    t = np.linspace(1950, 2050, 101)
    y = interpolate_q_total(t)
    a = y[64] / 36
    for i in range(65, 101):
        y[i] = y[64] - a * (i - 65)
    # plot_pressure_model(t,solve_pressure_ode(pressure_ode_model, t, dt, x0, 'STOP', pars=[ap, bp, cp])[1])
    # plot_pressure_model(t,y)
    # production_scenarios()
