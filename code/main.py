# imports
import pressure as p
import temperature as t
from matplotlib import pyplot as plt
import numpy as np

# plotting the given data

# boolean variable to run the dual plot 
run_plot1 = False # plotting the water level and total production rate
run_plot2 = False # conversion from water level to pressure
run_plot3 = False # plotting the temperature and total production rate

# the influence of borehole closure program on the water level recovery
if run_plot1:
    # File I/O commands to read in the data
    tq1,pr1 = np.genfromtxt('gr_q1.txt',delimiter=',',skip_header=1).T # production rate 1 (gr_q1 data)
    tq2,pr2 = np.genfromtxt('gr_q2.txt',delimiter=',',skip_header=1).T # production rate 2 (gr_q2 data)
    tp,wl = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T # water level (gr_p data)

    # plotting format 
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(tq1, pr1, 'k-', label='total production rate')
    # ln2 = ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2')
    ln3 = ax2.plot(tp, wl, 'r-', label="water level", markersize = 7)

    # lns = ln1+ln2+ln3 # if we want to plot rhyolite data as well
    lns = ln1+ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns,labs,loc=2)

    # ax1.legend(loc=2)
    ax1.set_ylabel('production rate [tonnes/day]')
    ax2.set_ylabel('water level [m]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('water level and total production rate data')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('given_data_1.png',dpi=300)

# conversion from water level to pressure 
if run_plot2:
    # File I/O commands to read in the data
    tq1,pr1 = np.genfromtxt('gr_q1.txt',delimiter=',',skip_header=1).T # production rate 1 (gr_q1 data)
    tq2,pr2 = np.genfromtxt('gr_q2.txt',delimiter=',',skip_header=1).T # production rate 2 (gr_q2 data)
    tp,wl = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T # water level (gr_p data)

    # plotting format 
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(tq1, pr1, 'k-', label='total production rate')
    # ln2 = ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2')


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
    ln3 = ax2.plot(tp_provided, p_plot/100000, 'bo', marker='o',markersize = 7, label='pressure data')  # pressure data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    # lns = ln1+ln2+ln3 # if we want to plot rhyolite data as well
    lns = ln1+ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns,labs,loc=2)

    # ax1.legend(loc=2)
    ax1.set_ylabel('production rate [tonnes/day]')
    ax2.set_ylabel('pressure [bar]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('total production rate data and calculated pressure')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('given_data_2.png',dpi=300)


# the influence of borehole closure program on the temperature recovery
if run_plot3:
    # File I/O commands to read in the data
    tq1,pr1 = np.genfromtxt('gr_q1.txt',delimiter=',',skip_header=1).T # production rate 1 (gr_q1 data)
    tq2,pr2 = np.genfromtxt('gr_q2.txt',delimiter=',',skip_header=1).T # production rate 2 (gr_q2 data)
    tT,Temp = np.genfromtxt('gr_T.txt',delimiter=',',skip_header=1).T # Temperature (gr_T data)

    # plotting format 
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(tq1, pr1, 'k-', label='total production rate')
    # ln2 = ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2') # if rhyolite data is plotted as well
    # ax1.plot(tT, Temp, 'r*', label='Temperature', markersize = 7)
    ln3 = ax2.plot(tT, Temp, 'r*', label='Temperature', markersize = 7)
    
    # lns = ln1+ln2+ln3 # if we want to plot rhyolite data as well
    lns = ln1+ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns,labs,loc=1)
    # ax2.legend(loc=3)
    # ax1.legend(loc=1)
    ax1.set_ylabel('production rate [tonnes/day]')
    ax2.set_ylabel('Temperature [degC]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('temperature and total production rate data')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('given_data_3.png',dpi=300)


fig, ax1 = plt.subplots(1)
ax2 = ax1.twinx()

# all parameters should be able to be changed for better fits without breaking the program
time0 = 1950                                    # starting time
time1 = 2014                                    # ending time

# PRESSURE
dt_p = 1                                        # step size
x0_p = 5000                                     # starting pressure value, PA
p0 = 5000                                       # hydrostatic pressure at recharge source assuming shallow inflow scott2016 pg297

# converting water level to pressure and plotting, the plot is gonna look weird because I changed it for a
# better fit will fix this graph later this is not that important right now
t_p, pressure_data = p.interp(time0, time1, dt_p)  

tp_provided, p_provided = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=1).T  # given pressure data
p_plot = np.interp(tp_provided, t_p, pressure_data) # the pressure calculated from water level is interpolated to match the time we are provided
ln1 = ax1.plot(tp_provided, p_plot/100000, 'bo', marker='o', label='pressure data')  # pressure data is converted from Pa to bar (1 Pa = 1.e-5 bar)

# estimating parameters still a basic implementation read note in fit for details
para_p, cov_p = p.fit(t_p, pressure_data, dt_p, x0_p, p0)

# numerical solution and plotting
time_fit, pressure_fit = p.solve_ode(p.ode_model, time0, time1, dt_p, x0_p, 'SAME', pars=[para_p[0], para_p[1], para_p[2], p0])
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
para_t, cov_t = t.fit(t_t, temp_data, dt_t, x0_t, pressure_fit, p0, t0)

# numerical solution and plotting
timeT_fit, temp_fit = t.solve_ode(t.ode_model, time0, time1, dt_t, x0_t, pressure_fit, pars=[para_t[0], para_t[1], p0, t0])
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
p.forecast(time0,2050,dt_p,x0_p,para_p[0],para_p[1],para_p[2],p0)

# predicting temperature
tp_pred, y_no_change = p.solve_ode(p.ode_model, time0,2050,dt_p,x0_p, 'SAME', pars=[para_p[0],para_p[1],para_p[2], p0])
y_stop = p.solve_ode(p.ode_model, time0,2050,dt_p,x0_p, 'STOP', pars=[para_p[0],para_p[1],para_p[2], p0])[1]
y_double = p.solve_ode(p.ode_model, time0,2050,dt_p,x0_p, 'DOUBLE', pars=[para_p[0],para_p[1],para_p[2], p0])[1]
y_half = p.solve_ode(p.ode_model, time0,2050,dt_p,x0_p, 'HALF', pars=[para_p[0],para_p[1],para_p[2], p0])[1]
t.forecast(time0, 2050, dt_t, x0_t, tp_pred, y_no_change, y_stop, y_double, y_half, para_t[0], para_t[1], p0, t0)

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

# calculating porosity
# ap = 1.41e-3
# para_p[1] = 5.95e-2
# para_p[2] = 8.5e-3
#para_p[0] = 0.15
#para_p[1] = 0.12
#para_p[2] = 0.14


g = 9.81
A = 150
S0 = 0.3
print("the estimated porosity through inverse modelling is:")
print((g*(para_p[0]-para_p[1]*para_p[2]))/((para_p[0]**2)*A*(1-S0)))


#Uncertainty

f, ax = plt.subplots(nrows=1, ncols=1)

ps = np.random.multivariate_normal(para_p, cov_p, 100)
for pi in ps:
    time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time1, dt_p, x0_p, 'SAME', pars=[pi[0], pi[1], pi[2], p0])
    ax.plot(time_temp, pressure_temp/100000, 'k-', alpha=0.2, lw=0.5,label='Uncertainty')

    time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, 2050, dt_p, x0_p, 'SAME', pars=[pi[0], pi[1], pi[2], p0])
    ax.plot(time_temp[64:], pressure_temp[64:] / 100000, 'r-', alpha=0.2, lw=0.5, label='Uncertainty')

    time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, 2050, dt_p, x0_p, 'STOP', pars=[pi[0], pi[1], pi[2], p0])
    ax.plot(time_temp[64:], pressure_temp[64:] / 100000, 'b-', alpha=0.2, lw=0.5, label='Uncertainty')

    time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, 2050, dt_p, x0_p, 'DOUBLE', pars=[pi[0], pi[1], pi[2], p0])
    ax.plot(time_temp[64:], pressure_temp[64:] / 100000, 'g-', alpha=0.2, lw=0.5, label='Uncertainty')

    time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, 2050, dt_p, x0_p, 'HALF', pars=[pi[0], pi[1], pi[2], p0])
    ax.plot(time_temp[64:], pressure_temp[64:] / 100000, 'y-', alpha=0.2, lw=0.5, label='Uncertainty')

v = 0.03
p_provided = (p_provided * 997 * 9.81) - 2909250 + 5000
ax.errorbar(tp_provided, p_provided/100000, yerr=v, fmt='ro', label='data')
plt.show()


f, ax = plt.subplots(nrows=1, ncols=1)

ps = np.random.multivariate_normal(para_t, cov_t, 100)
for pi in ps:
    time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time1, dt_t, x0_t, y_no_change, pars=[pi[0], pi[1], p0, t0])
    ax.plot(time_hold, temp_hold, 'k-', alpha=0.2, lw=0.5,label='Uncertainty')

    time_hold, temp_hold = t.solve_ode(t.ode_model, time0, 2050, dt_t, x0_t, y_no_change, pars=[pi[0], pi[1], p0, t0])
    ax.plot(time_hold[64:], temp_hold[64:], 'y-', alpha=0.2, lw=0.5, label='Uncertainty')

    time_hold, temp_hold = t.solve_ode(t.ode_model, time0, 2050, dt_t, x0_t, y_stop, pars=[pi[0], pi[1], p0, t0])
    ax.plot(time_hold[64:], temp_hold[64:], 'r-', alpha=0.2, lw=0.5, label='Uncertainty')

    time_hold, temp_hold = t.solve_ode(t.ode_model, time0, 2050, dt_t, x0_t, y_double, pars=[pi[0], pi[1], p0, t0])
    ax.plot(time_hold[64:], temp_hold[64:], 'b-', alpha=0.2, lw=0.5, label='Uncertainty')

    time_hold, temp_hold = t.solve_ode(t.ode_model, time0, 2050, dt_t, x0_t, y_half, pars=[pi[0], pi[1], p0, t0])
    ax.plot(time_hold[64:], temp_hold[64:], 'g-', alpha=0.2, lw=0.5, label='Uncertainty')

v = 1
ax.errorbar(tT_provided, T_provided, yerr=v, fmt='ro', label='data')
plt.show()