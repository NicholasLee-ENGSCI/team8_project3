# Team 8 Project 3
# Python code for model fitting 


# imports
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import optimize as op

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
    tq1,pr1 = np.genfromtxt('gr_q1.txt',delimiter=',',skip_header=1).T # production rate 1 (gr_q1 data)
    tq2,pr2 = np.genfromtxt('gr_q2.txt',delimiter=',',skip_header=1).T # production rate 2 (gr_q2 data)

    # total extraction rate is found by summing the two extraction rates given
    # interpolating two data, then summed to find total q
    ex1 = np.interp(t,tq1,pr1)    
    ex2 = np.interp(t,tq2,pr2)   
    return ex1+ex2

def pressure_ode_model(t, P, P0, q, dqdt, ap, bp, cp):            #{remember to check order of parameters match when bug fixing}, consider changing P to x to follow standard notation???, inital or ambient
    ''' Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        P : float
            Dependent variable.
        q : float
            total extraction rate.
        dqdt : float
            rate of change of total extraction rate. 
        ap : float
            extraction strength parameter.
        b : float
            recharge strength parameter.
        c : float
            slow drainage strength parameter. 
        P0 : float
            Ambient value of the dependent variable.

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
    return -ap*q-bp*(P-P0)-cp*dqdt

def solve_pressure_ode(f,t, dt, P0, indicator, pars):
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
    
    # total extraction is found by interpolating two extraction rates given and summing them (done using the interpolate_q_total() function)                                {remember to add q and dqdt into paramters}
    if (indicator == 'SAME'):
        q = interpolate_q_total(t)
        q = q/86.4
    elif (indicator == 'STOP'):
        q = interpolate_q_total(t)
        q = q/86.4
        q[65:101] = 0
    elif (indicator == 'DOUBLE'):
        q = interpolate_q_total(t)
        q = q/86.4
        q[65:101] = q[64]*2
    elif (indicator == 'HALF'):
        q = interpolate_q_total(t)
        q = q/86.4
        q[65:101] = q[64]/2
    dqdt = np.gradient(q)

    nt = int(np.ceil((t[-1]-t[0])/dt))	#calculating number of steps	
    ts = t[0]+np.arange(nt+1)*dt		#initilaise time array
    ys = 0.*ts						    #initialise solution array
    ys[0] = P0						    #set initial value of solution array

    #calculate solution values using Improved Euler                                                                                                                         #{are we using ambient or initial pressure to calcualte rate of change???}
    for i in range(nt):
        fk = f(ts[i], ys[i], ys[i-1], q[i], dqdt[i], *pars)
        fk1 = f(ts[i] + dt, ys[i] + dt*fk, ys[i-1], q[i], dqdt[i], *pars)
        ys[i+1] = ys[i] + dt*((fk + fk1)/2)

	#Return both arrays contained calculated values
    return ts, ys

def fit_pressure(t, dt, P0, indicator, ap, bp, cp):
    time, pressure = solve_pressure_ode(pressure_ode_model, t, dt, P0, indicator, pars=[ap, bp, cp])
    return pressure

def plot_pressure_model(t,y):       
    '''
    '''
    # plotting format 
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ln1 = ax1.plot(t, y, 'k-', label='dq/dt')

    ax1.set_ylabel('rate change of production rate [tonnes/day^2]')
    ax1.set_xlabel('time [yr]')
    ax1.set_title('Rate of change of total extraction rate')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('Rate_of_change_of_total_extraction_rate.png',dpi=300)
    return

def temperature_ode_model(t, T, T0, P, P0, ap, bp, at, bt):
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
        Tx = 30 # otherwise Tx is equal to temperature of the cold water injection, 30 degrees celsius.
    
    # the first derivative returns dT/dt = -at*bp*(1/ap)*(Tx-T)-bt*(T-T0) where all the parameters are provided as inputs 
    return -at*(bp/ap)*(P-P0)*(Tx-T) - bt*(T-T0)                                                                                         

def solve_temperature_ode(f,t, dt, T0, P, pars):
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

    nt = int(np.ceil((t[-1]-t[0])/dt))	    #number of steps	
    ts = t[0]+np.arange(nt+1)*dt		    #x/t array
    ys = 0.*ts						        #array to store solution values
    ys[0] = T0						        #set initial value of solution array

    #calculate solution values using Improved Euler
    for i in range(nt):

        fk = f(ts[i], ys[i], ys[i-1], P[i], P[i-1], *pars)
        fk1 = f(ts[i] + dt, ys[i] + dt*fk, ys[i-1], P[i], P[i-1], *pars)
        ys[i+1] = ys[i] + dt*((fk + fk1)/2)
    
	#Return both arrays contained calculated values
    return ts, ys

def fit_temperature(t, dt, T0, P, ap, bp, at, bt):
    time, temp = solve_temperature_ode(temperature_ode_model, t, dt, T0, P, pars = [ap, bp, at, bt])
    return temp

def pressure_forecast():
    '''
    This function is to extrapolate to year 2050, then plot it 
    '''
    # plotting format 
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    # plotting no change 
    t = np.linspace(1950,2050,101)
    y_no_change = solve_pressure_ode(pressure_ode_model, t, dt, x0, 'SAME', pars=[ap, bp, cp])[1]
    # plotting stopped production
    y_stop = solve_pressure_ode(pressure_ode_model, t, dt, x0, 'STOP', pars=[ap, bp, cp])[1]
    # plotting double production
    y_double = solve_pressure_ode(pressure_ode_model, t, dt, x0, 'DOUBLE', pars=[ap, bp, cp])[1]
    # plotting half production
    y_half = solve_pressure_ode(pressure_ode_model, t, dt, x0, 'HALF', pars=[ap, bp, cp])[1]
    ln1 = ax1.plot(t, y_no_change/10**6, 'k-', label='maintained production')
    ln2 = ax1.plot(t, y_stop/10**6, 'r-', label='operation terminated')
    ln3 = ax1.plot(t, y_double/10**6, 'g-', label='production doubled')
    ln4 = ax1.plot(t, y_half/10**6, 'b-', label='production halved')

    lns = ln1+ln2+ln3+ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns,labs,loc=2)
    ax1.set_ylabel('Pressure [MPa]')
    ax1.set_xlabel('time [yr]')
    ax1.set_title('Pressure predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast.png',dpi=300)
    return

if __name__ == "__main__":
    #read in water level andd time
    #tp,wl = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T
    

    #qtot = interpolate_q_total(tp)
    #total extraction rate 
    # plot_pressure_model(tp,qtot)
    #total rate of change of extraction rate 
    #plot_pressure_model(tp,dqdt_function(tp,qtot))


    #plotting given water level data
    t = np.linspace(1950,2014,65)

    tP, wl = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T # water level (gr_p data)
    wl = ((wl*997*9.81) - 2909250)*100
    wp = np.interp(t,tP,wl)

    fig, ax1 = plt.subplots(1)
    ax2 = ax1.twinx()
    ax1.plot(tP, wl, 'bo', marker='o')

    

    dt = 1              #step size
    x0 = 160000          #starting pressure value

    ap = -1.74
    bp = 0.56
    cp = 180

    bound = np.inf           #np.inf to ignore bounds
    #p0=[0.15, 0.12, 0.6], bounds=([-bound,-bound,-bound],[bound,bound,bound])
    paraP,_ = op.curve_fit(lambda t, ap, bp, cp : fit_pressure(t, dt, x0,'SAME', ap, bp, cp), xdata=t, ydata=wp,)

    print(paraP)
    ap = -1.74
    bp = 0.56
    cp = 180

    ap = paraP[0]
    bp = paraP[1]
    cp = paraP[2]
    


    timei, pressurei = solve_pressure_ode(pressure_ode_model, t, dt, x0, 'SAME', pars=[ap, bp, cp])
    ax1.plot(timei, pressurei,'b--')
    
    
    #plotting given temperature data
    tT,temp = np.genfromtxt('gr_T.txt',delimiter=',',skip_header=1).T # Temperature (gr_T data)
    ax2.plot(tT, temp, 'ro', marker='o')
    tTemp = np.interp(t,tP,wl)

    x0 = 149           #starting temperature value

    #ap = 0.15
    #bp = 0.12

    at = 0.000005
    bt = 0.065

    #
    #


    #paraT,_ = op.curve_fit(lambda t, at, bt: fit_temperature(t, dt, x0, pressurei, ap, bp, at, bt), xdata=t, ydata=tTemp)
    #print(paraT)
    #at = 0.1
    #bt = 0

    #at = paraT[0]
    #bt = paraT[1]

    

    timei, tempi = solve_temperature_ode(temperature_ode_model, t, dt, x0, wp, pars=[ap, bp, at, bt])
    ax2.plot(timei, tempi,'r--')
    

    plt.show()

    pressure_forecast()
    # t = np.linspace(1950,2050,101)
    # plot_pressure_model(t,solve_pressure_ode(pressure_ode_model, t, dt, x0, 'SAME', pars=[ap, bp, cp])[1])