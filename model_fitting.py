# Team 8 Project 3
# Python code for model fitting 


# imports
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

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
        a = (q[i+1]-q[i])/(365*(t[i+1]-t[i])) # denominator is multiplied by 365 to convert years to days. 
        dqdt.append(a)
    
    # dqdt of the last data point is assumed to be of same rate as the dqdt of second to last data point.
    dqdt.append(a)
    return dqdt

def pressure_ode_model(t, P, q, dqdt, ap, b, c, P0):            #{remember to check order of parameters match when bug fixing}, consider changing P to x to follow standard notation???, inital or ambient
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
            Initial value of the dependent variable.

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
    return -ap*q-b*(P-P0)-c*dqdt

def solve_pressure_ode(f,t, dt, x0, pars):
    '''Solve the pressure ODE numerically.
        
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

    nt = int(np.ceil((t[-1]-t[0])/dt))	#number of steps	
    ts = t[0]+np.arange(nt+1)*dt		    #x/t array
    ys = 0.*ts						    #array to store solution values
    ys[0] = x0						    #set initial value of solution array

    #calculate solution values using Improved Euler                                                                                                                         #{are we using ambient or initial pressure to calcualte rate of change???}
    for i in range(nt):

        fk = f(ts[i], ys[i], q[i], dqdt[i], *pars)

        #calculating the second derivative evaulation
        fk1 = f(ts[i] + dt, ys[i] + dt*fk, q[i], dqdt[i], *pars)

        #return stepped improved euler value
        ys[i+1] = ys[i] + dt*((fk + fk1)/2)
    
	#Return both arrays contained calculated values
    return ts, ys

def plot_pressure_model(t,y):       #{could rewrite this genericaly later and include labels as parameters??? otherwise is a redundant function atm, good for now}
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

def temperature_ode_model(t, T, at, bp, ap, bt, P, P0, Tx, T0):
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
    return -at*bp*(1/ap)*(Tx-T)-bt*(T-T0)


if __name__ == "__main__":
    tp,wl = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T
    qtot = interpolate_q_total(tp)
    # total extraction rate 
    #plot_pressure_model(tp,qtot)
    # total rate of change of extraction rate 
    #plot_pressure_model(tp,dqdt_function(tp,qtot))

    #for nick
    #I've placed my thoughts and things we can come back and fix inbetween {} all over the code
    #{this is an example}



    #Shalin's generic ode solver testing, this would be moved to a main script eventually

    dt = 1      #what is out step size days years?
    x0 = tp[0]


    a = 0.4
    b = 0.5
    c = 0.1

    timei, pressurei = solve_pressure_ode(pressure_ode_model, tp, dt, x0, pars=[a, b, c, x0])
    fig, axes = plt.subplots(1)
    axes.plot(timei, pressurei, color='r', marker='o')
    plt.show()