# imports
import pressure as p
import temperature as t
from matplotlib import pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(1)
ax2 = ax1.twinx()

# all parameters should be able to be changed for better fits without breaking the program
time0 = 1950                                    # starting time
time1 = 2014                                    # ending time

# PRESSURE
dt_p = 1                                        # step size
x0_p = 5000                                     # starting pressure value, PA
p0 = 0                                          # hydrostatic pressure at recharge source

# converting water level to pressure and plotting, the plot is gonna look weird because I changed it for a
# better fit will fix this graph later this is not that important right now
t_p, pressure_data = p.interp(time0, time1, dt_p)  

tp_provided, p_provided = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=1).T  # given pressure data  
p_plot = np.interp(tp_provided, t_p, pressure_data) # the pressure calculated from water level is interpolated to match the time we are provided
ln1 = ax1.plot(tp_provided, p_plot/100000, 'bo', marker='o', label='pressure data')  # pressure data is converted from Pa to bar (1 Pa = 1.e-5 bar)

# estimating parameters still a basic implementation read note in fit for details
ap, bp, cp = p.fit(t_p, pressure_data, dt_p, x0_p, p0)

# numerical solution and plotting
time_fit, pressure_fit = p.solve_ode(p.ode_model, time0, time1, dt_p, x0_p, 'SAME', pars=[ap, bp, cp, p0])
ln2 = ax1.plot(time_fit, pressure_fit/100000, 'b-', label='pressure best fit')  # pressure_fit data is converted from Pa to bar (1 Pa = 1.e-5 bar) 


# TEMPERATURE
dt_t = 1                                    # step size
x0_t = 149                                  # starting temperature
t0 = 147                                    # temperature outside CV/ conduction source

# interpolating temp will also look weird will fix later
t_t, temp_data = t.interp(time0, time1, dt_t)

tT_provided, T_provided = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # given temperature data
ln3 = ax2.plot(tT_provided, T_provided, 'ro', marker='o', label='temperature data')

# estimating parameters still a basic implementation read notes in fit for details
alpha, bt = t.fit(t_t, temp_data, dt_t, x0_t, pressure_fit, p0, t0)

# numerical solution and plotting
timeT_fit, temp_fit = t.solve_ode(t.ode_model, time0, time1, dt_t, x0_t, pressure_fit, pars=[alpha, bt, p0, t0])
ln4 = ax2.plot(timeT_fit, temp_fit, 'r-', label='temperature best fit')

lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc=3)

ax1.set_ylabel('pressure [bar]')
ax2.set_ylabel('temperature [degC]')
ax2.set_xlabel('time [yr]')
ax1.set_xlabel('time [yr]')
ax2.set_title('Best fit model for given pressure and temperature data')

# this is the only plotting that should be in the main file because this plot requires data from temp and pressure
# all other plots should be in their respective parent file e.g. pressure scenarios in pressure
# EITHER show the plot to the screen OR save a version of it to the disk
save_figure = False
if not save_figure:
    plt.show()
else:
    plt.savefig('best_fit.png',dpi=300)

# forecasting pressure for different production scenarios from 2014 to 2050
p.forecast(time0,2050,dt_p,x0_p,ap,bp,cp,p0)

# predicting temperature
tp_pred, y_no_change = p.solve_ode(p.ode_model, time0,2050,dt_p,x0_p, 'SAME', pars=[ap,bp,cp, p0])
y_stop = p.solve_ode(p.ode_model, time0,2050,dt_p,x0_p, 'STOP', pars=[ap,bp,cp, p0])[1]
y_double = p.solve_ode(p.ode_model, time0,2050,dt_p,x0_p, 'DOUBLE', pars=[ap,bp,cp, p0])[1]
y_half = p.solve_ode(p.ode_model, time0,2050,dt_p,x0_p, 'HALF', pars=[ap,bp,cp, p0])[1]
t.forecast(time0, 2050, dt_t, x0_t, tp_pred, y_no_change, y_stop, y_double, y_half, alpha, bt, p0, t0)

def quant_misfit():
    '''
    quantify the misfit between the best fit model and the data, then visualise it. 
    '''
    # quantifying misfit for pressure LPM ODE
    fig, axes = plt.subplots(1,2)
    ln1 = axes[0].plot(tp_provided, p_plot/100000, 'bo', marker='o', label='data')
    px_plot = np.interp(tp_provided, time_fit, pressure_fit)
    ln2 = axes[0].plot(tp_provided, px_plot/100000, 'k-', label='ap = -1.067\nbp = -9.209e-3\ncp = 8.729e+1')

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns,labs,loc=4)
    axes[0].set_ylabel('pressure [bar]')
    axes[0].set_xlabel('time [yr]')
    axes[0].set_title('best fit pressure LPM ODE model')

    p_misfit = px_plot - p_plot
    axes[1].plot(tp_provided, np.zeros(len(tp_provided)), 'k--')
    axes[1].plot(tp_provided, p_misfit/100000, 'rx')

    axes[1].set_ylabel('pressure misfit [bar]')
    axes[1].set_xlabel('time [yr]')
    axes[1].set_title('best fit pressure LPM ODE model')

    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_misfit.png',dpi=300)

    # quantifying misfit for temperature LPM ODE
    fig, axes = plt.subplots(1,2)
    ln1 = axes[0].plot(tT_provided, T_provided, 'bo', marker='o', label='data')
    Tx_plot = np.interp(tT_provided, timeT_fit, temp_fit)
    ln2 = axes[0].plot(tT_provided, Tx_plot, 'k-', label='alpha = 8.409e-7\nbt = 6.355e-2')

    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns,labs,loc=4)
    axes[0].set_ylabel('temperature [degC]')
    axes[0].set_xlabel('time [yr]')
    axes[0].set_title('best fit temperature LPM ODE model')

    T_misfit = Tx_plot - T_provided
    axes[1].plot(tT_provided, np.zeros(len(tT_provided)), 'k--')
    axes[1].plot(tT_provided, T_misfit, 'rx')

    axes[1].set_ylabel('temperature misfit [degC]')
    axes[1].set_xlabel('time [yr]')
    axes[1].set_title('best fit temperature LPM ODE model')

    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_misfit.png',dpi=300)
    return

# calling the misfit plot function 
quant_misfit()
