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

def dqdt_function(t,q):
    ''' differentiate to find the rate of change of extraction with respect to time.

        Parameters:
        -----------
        t : float
            Independent variable.
        q : float 
            Dependent variable. Total extraction output from function interpolate_q_total.
        
        Returns:
        --------
        dqdt : float
            rate of change of total extraction rate.

        Note:
        -----
        dqdt = (q[i+1]-q[i])/(t[i+1]-t[i])
        q is in Tonnes per day and t is in years, so the t needs to convert unit to days. 
    '''
    dqdt = []
    for i in range(0,len(t)-1):
        a = (q[i+1]-q[i])/((t[i+1]-t[i])) # denominator is multiplied by 365 to convert years to days. 
        dqdt.append(a)
    
    # dqdt of the last data point is assumed to be of same rate as the dqdt of second to last data point.
    dqdt.append(a)
    return dqdt

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

def solve_pressure_ode(f,t, dt, P0, pars):
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
    '''
    
    # total extraction is found by interpolating two extraction rates given and summing them (done using the interpolate_q_total() function)                                {remember to add q and dqdt into paramters}
    q = interpolate_q_total(t) 
    q = q*365000                                                                                                                                      
                                                                                                                                                                            #{what is order of parameters?}
    # rate of change of total extraction rate is found by differentiating (done using the dqdt_function() function)                                                         {this is a crude solution but works for now}
    dqdt = dqdt_function(t,q)

    nt = int(np.ceil((t[-1]-t[0])/dt))	#calculating number of steps	
    ts = t[0]+np.arange(nt+1)*dt		#initilaise time array
    ys = 0.*ts						    #initialise solution array
    ys[0] = P0						    #set initial value of solution array
    
    #calculating initial value before loop so initial presure can be specified
    #fk = f(ts[0], ys[0], ys[0], q[0], dqdt[0], *pars)
    #fk1 = f(ts[0] + dt, ys[0] + dt*fk, ys[0], q[0], dqdt[0], *pars)
    #ys[0] = ys[0] + dt*((fk + fk1)/2)

    #fk = f(ts[0], ys[0], 5000, q[0], dqdt[0], *pars) 
    #fk1 = f(ts[0] + dt/2, ys[0] + dt*fk/2, 5000, q[0], dqdt[0], *pars)
    #fk2 = f(ts[0] + dt/2, ys[0] + dt*fk1/2, 5000, q[0], dqdt[0], *pars)
    #fk3 = f(ts[0] + dt, ys[0] + dt*fk2, 5000, q[0], dqdt[0], *pars)
    #ys[1] = ys[0] + dt*((fk + 2*fk1 + 2*fk2 + fk3)/6)

    #calculate solution values using Improved Euler                                                                                                                         #{are we using ambient or initial pressure to calcualte rate of change???}
    for i in range(nt):

        #improved eulers
        fk = f(ts[i], ys[i], ys[i-1], q[i], dqdt[i], *pars)
        fk1 = f(ts[i] + dt, ys[i] + dt*fk, ys[i-1], q[i], dqdt[i], *pars)
        ys[i+1] = ys[i] + dt*((fk + fk1)/2)

        #RK45
        #fk = f(ts[i], ys[i], ys[i-1], q[i], dqdt[i], *pars) 
        #fk1 = f(ts[i] + dt/2, ys[i] + dt*fk/2, ys[i-1], q[i], dqdt[i], *pars)
        #fk2 = f(ts[i] + dt/2, ys[i] + dt*fk1/2, ys[i-1], q[i], dqdt[i], *pars)
        #fk3 = f(ts[i] + dt, ys[i] + dt*fk2, ys[i-1], q[i], dqdt[i], *pars)
        #ys[i+1] = ys[i] + dt*((fk + 2*fk1 + 2*fk2 + fk3)/6)
    
	#Return both arrays contained calculated values
    return ts, ys

def fit_pressure(t, ap, bp, cp):
    dt = 1          #constant step size
    P0 = 5000       #constant inital value 

    time, pressure = solve_pressure_ode(pressure_ode_model, t, dt, P0, pars=[ap, bp, cp])
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
    return -at*(bp/ap)*(P-P0)*(Tx-T)-bt*(T-T0)                                                                                           #{cancel to 0 if pressure makes Tx = T}

def solve_temperature_ode(f,t, dt, x0, pars):
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
    
    # total extraction is found by interpolating two extraction rates given and summing them (done using the interpolate_q_total() function)                                {remember to add q and dqdt into paramters}
    q = interpolate_q_total(t)                                                                                                                                             
                                                                                                                                                                            #{what is order of parameters?}
    # rate of change of total extraction rate is found by differentiating (done using the dqdt_function() function)                                                         {this is a crude solution but works for now}
    dqdt = dqdt_function(t,q)

    nt = int(np.ceil((t[-1]-t[0])/dt))	    #number of steps	
    ts = t[0]+np.arange(nt+1)*dt		    #x/t array
    ys = 0.*ts						        #array to store solution values
    ys[0] = x0						        #set initial value of solution array

    #calculate solution values using Improved Euler                                                                                                                         #{are we using ambient or initial pressure to calcualte rate of change???}
    for i in range(nt):


        fk = f(ts[i], ys[i], q[i], dqdt[i], *pars)

        #calculating the second derivative evaulation
        fk1 = f(ts[i] + dt, ys[i] + dt*fk, q[i], dqdt[i], *pars)

        #return stepped improved euler value
        ys[i+1] = ys[i] + dt*((fk + fk1)/2)
    
	#Return both arrays contained calculated values
    return ts, ys

def fit_temperature(t, aT, bT):
    dt = 1          #constant step size
    T0 = 149       #constant inital value 

    time, pressure = solve_temperature_ode(temperature_ode_model(t, T, T0, P, P0, ap, bp, at, bt))
    return pressure

if __name__ == "__main__":

    #read in water level andd time
    #tp,wl = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T
    

    #qtot = interpolate_q_total(tp)
    #total extraction rate 
    #plot_pressure_model(tp,qtot)
    #total rate of change of extraction rate 
    #plot_pressure_model(tp,dqdt_function(tp,qtot))


    #plotting given water level data
    t = np.linspace(1950,2014,65)

    tP, wl = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T # water level (gr_p data)
    wl = ((wl*997*9.81) - 2909250+5000)
    wp = np.interp(t,tP,wl)

    fig, ax1 = plt.subplots(1)
    ax2 = ax1.twinx()
    ax1.plot(tP, wl, 'bo', marker='o')

    para_bounds = ([-1,-1,-1],[1,1,1])
    para_bounds = ([-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf])

    paraP,_ = op.curve_fit(fit_pressure, t, wp, bounds=para_bounds)
    print(paraP)

    ap = paraP[0]
    bp = paraP[1]
    cp = paraP[2]

    dt = 1      #step size in days
    x0 = 5000   #starting pressure value

    timei, pressurei = solve_pressure_ode(pressure_ode_model, t, dt, x0, pars=[ap, bp, cp])
    ax1.plot(timei, pressurei,'r--')

    

    #plotting given temperature data
    timeTemp,temp = np.genfromtxt('gr_T.txt',delimiter=',',skip_header=1).T # Temperature (gr_T data)
    ax2.plot(timeTemp, temp, 'ro', marker='o')

    #timei, tempi = solve_temperature_ode(pressure_ode_model, t, dt, x0, pars=[ap, bp])
    #ax2.plot(timei, tempi, color='r', marker='o')
    

    

    plt.show()