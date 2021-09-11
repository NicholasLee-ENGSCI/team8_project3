# imports
import pressure as p
import temperature as t
from matplotlib import pyplot as plt
import numpy as np
import tests as tests

# plotting the given data
save_figure = False    # if this is true all plots will save to disk instead of print

given1 = True  # plotting the water level and total production rate (with reinjection rate considered).
given2 = True  # conversion from water level to pressure.
given3 = True  # plotting the temperature and total production rate (with reinjection rate considered).
validation = True  # plot benchmarking and convergence test.
firstfit = True
bestfit = True  # plot pressure and temperature bestfit LPM ODE models. MUST REMAIN TRUE TO RUN PLOTS THAT FOLLOWS.# plot pressure and temperature forecast to time2, as well as respective change rate forecast. MUST REMAIN TRUE TO RUN PLOTS THAT FOLLOWS.
forecast = True
misfit = True  # plot quantified misfit of the model to data.
inversion = True
uncertainty = True  # plot of pressure and temperature forecast uncertainty.

tq1, pr1 = np.genfromtxt('gr_q1.txt', delimiter=',', skip_header=1).T  # production rate 1 (gr_q1 data)
tq2, pr2 = np.genfromtxt('gr_q2.txt', delimiter=',', skip_header=1).T  # production rate 2 (gr_q2 data)
twl, wl = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=1).T  # water level (gr_p data)
t_given, temp_given = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # Temperature (gr_T data)

np.random.seed(1)
time0 = 1950    # starting time
time1 = 2014    # ending time
time2 = 2050
dt = 1          # step size

# the influence of borehole closure program on the water level recovery
if given1:
    # plotting format 
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(tq1, p.interpolate_q_total(tq1), 'k-', label='total production rate')
    # ln2 = ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2')
    ln3 = ax2.plot(twl, wl, 'r-', label="water level", markersize=7)

    # lns = ln1+ln2+ln3 # if we want to plot rhyolite data as well
    lns = ln1 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=2)

    # ax1.legend(loc=2)
    ax1.set_ylabel('production rate [tonnes/day]')
    ax2.set_ylabel('water level [m]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('water level and total production rate data')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('given_data_1.png', dpi=300)

# conversion from water level to pressure 
if given2:
    # plotting format 
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(tq1, p.interpolate_q_total(tq1), 'k-', label='total production rate')
    # ln2 = ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2')

    t_data, pressure_data = p.interp(time0, time1, dt)

    p_plot = np.interp(twl, t_data, pressure_data)  # the pressure calculated from water level is interpolated to match the time we are provided
    ln3 = ax2.plot(twl, p_plot / 100000, 'bo', marker='o', markersize=7, label='pressure data')  # pressure data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    # lns = ln1+ln2+ln3 # if we want to plot rhyolite data as well
    lns = ln1 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=2)

    # ax1.legend(loc=2)
    ax1.set_ylabel('production rate [tonnes/day]')
    ax2.set_ylabel('pressure [bar]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('total production rate data and calculated pressure')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('given_data_2.png', dpi=300)

# the influence of borehole closure program on the temperature recovery
if given3:
    # plotting format 
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(tq1, p.interpolate_q_total(tq1), 'k-', label='total production rate')
    # ln2 = ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2') # if rhyolite data is plotted as well
    # ax1.plot(tT, Temp, 'r*', label='Temperature', markersize = 7)
    ln3 = ax2.plot(t_given, temp_given, 'r*', label='Temperature', markersize=7)

    # lns = ln1+ln2+ln3 # if we want to plot rhyolite data as well
    lns = ln1 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=1)
    # ax2.legend(loc=3)
    # ax1.legend(loc=1)
    ax1.set_ylabel('production rate [tonnes/day]')
    ax2.set_ylabel('Temperature [degC]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('temperature and total production rate data')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('given_data_3.png', dpi=300)

if firstfit:
    fig, ax1 = plt.subplots(1)
    ax2 = ax1.twinx()

    ############
    # PRESSURE #
    ############
    x0_p = 5000  # starting pressure value, PA
    p0 = 5000  # hydrostatic pressure at recharge source assuming shallow inflow scott2016 pg297

    t_data, pressure_data = p.interp(time0, time1, dt)
    p_plot = np.interp(twl, t_data, pressure_data)  # the pressure calculated from water level is interpolated to match the time we are provided
    ln1 = ax1.plot(twl, p_plot / 100000, 'bo', marker='o', label='pressure data')  # pressure data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    # estimating parameters still a basic implementation read note in fit for details
    para_p, cov_p = p.fit(t_data, pressure_data, dt, x0_p, p0, False)

    # numerical solution and plotting
    time_fit, pressure_fit = p.solve_ode(p.ode_model, time0, time1, dt, x0_p, 'SAME', pars=[para_p[0], para_p[1], 0, p0])
    ln2 = ax1.plot(time_fit, pressure_fit / 100000, 'b-', label='pressure best fit')  # pressure_fit data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    print("for out first model {suitable} the pressure ode parameters are:")
    print("x0=", x0_p, "a=", para_p[0], " b=", para_p[1], "c= 0", "p0=", p0, "\n")


    ###############
    # TEMPERATURE #
    ###############
    x0_t = 149  # starting temperature
    t0 = 147  # temperature outside CV/conduction source

    ln3 = ax2.plot(t_given, temp_given, 'ro', marker='o', label='temperature data')

    # estimating parameters still a basic implementation read notes in fit for details

    # interpolating temp
    t_t, temp_data = t.interp(time0, time1, dt)
    para_t, cov_t = t.fit(t_t, temp_data, dt, x0_t, pressure_fit, p0, t0)

    # numerical solution and plotting
    timeT_fit, temp_fit = t.solve_ode(t.ode_model, time0, time1, dt, x0_t, pressure_fit,
                                      pars=[para_t[0], para_t[1], p0, t0])
    ln4 = ax2.plot(timeT_fit, temp_fit, 'r-', label='temperature best fit')

    print("for out first model {suitable} the temperature parameters are:")
    print("x0=", x0_t, "a=", para_t[0], " b=", para_t[1], "p0=", t0, "\n\n")

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=3)

    ax1.set_ylabel('pressure [bar]')
    ax2.set_ylabel('temperature [degC]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('Best fit model for given pressure and temperature data')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('first_fit.png', dpi=300)

if misfit:
    '''
    quantify the misfit between the best fit model and the data, then visualise it. 
    '''
    ############
    # PRESSURE # quantifying misfit for pressure LPM ODE
    ############
    fig, axes = plt.subplots(1, 2)
    ln1 = axes[0].plot(twl, p_plot, 'bo', marker='o', label='data')
    px_plot = np.interp(twl, time_fit, pressure_fit)
    ln2 = axes[0].plot(twl, px_plot, 'k-', label='ap = {a:.3f}\nbp = {b:.3f}\ncp = {c:.3f}'.format(a=para_p[0], b=para_p[1], c=0))

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc=4)
    axes[0].set_ylabel('pressure [bar]')
    axes[0].set_xlabel('time [yr]')
    axes[0].set_title('first fit pressure LPM ODE model')

    p_misfit = px_plot - p_plot
    axes[1].plot(twl, np.zeros(len(twl)), 'k--')
    axes[1].plot(twl, p_misfit/100000, 'rx')

    axes[1].set_ylabel('pressure misfit [bar]')
    axes[1].set_xlabel('time [yr]')
    axes[1].set_title('first fit pressure LPM ODE model')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('first_pressure_misfit.png', dpi=300)


    ###############
    # TEMPERATURE # quantifying misfit for temperature LPM ODE
    ###############
    fig, axes = plt.subplots(1, 2)
    t_t, temp_data = t.interp(time0, time1, dt)

    ln1 = axes[0].plot(t_given, temp_given, 'bo', marker='o', label='data')
    Tx_plot = np.interp(t_given, timeT_fit, temp_fit)
    ln2 = axes[0].plot(t_given, Tx_plot, 'k-', label='at = {a:.7e}\nbt = {b:.3f}'.format(a=para_t[0], b=para_t[1]))

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc=3)
    axes[0].set_ylabel('temperature [degC]')
    axes[0].set_xlabel('time [yr]')
    axes[0].set_title('first fit temperature LPM ODE model')

    T_misfit = Tx_plot - temp_given
    axes[1].plot(t_given, np.zeros(len(t_given)), 'k--')
    axes[1].plot(t_given, T_misfit, 'rx')

    axes[1].set_ylabel('temperature misfit [degC]')
    axes[1].set_xlabel('time [yr]')
    axes[1].set_title('first fit temperature LPM ODE model')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('first_temperature_misfit.png', dpi=300)

if bestfit:
    fig, ax1 = plt.subplots(1)
    ax2 = ax1.twinx()

    ############
    # PRESSURE #
    ############
    x0_p = 5000     # starting pressure value, PA
    p0 = 5000       # hydrostatic pressure at recharge source assuming shallow inflow scott2016 pg297

    t_data, pressure_data = p.interp(time0, time1, dt)
    p_plot = np.interp(twl, t_data, pressure_data)  # the pressure calculated from water level is interpolated to match the time we are provided
    ln1 = ax1.plot(twl, p_plot / 100000, 'bo', marker='o', label='pressure data')  # pressure data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    # estimating parameters still a basic implementation read note in fit for details
    para_p, cov_p = p.fit(t_data, pressure_data, dt, x0_p, p0, True)

    # numerical solution and plotting
    time_fit, pressure_fit = p.solve_ode(p.ode_model, time0, time1, dt, x0_p, 'SAME', pars=[para_p[0], para_p[1], para_p[2], p0])
    ln2 = ax1.plot(time_fit, pressure_fit / 100000, 'b-', label='pressure best fit')  # pressure_fit data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    print("for out first model {suitable} the pressure ode parameters are:")
    print("x0=", x0_p, "a=", para_p[0], " b=", para_p[1], "c=", para_p[2], "p0=", p0, "\n")

    ###############
    # TEMPERATURE #
    ###############
    x0_t = 149      # starting temperature
    t0 = 147        # temperature outside CV/conduction source


    ln3 = ax2.plot(t_given, temp_given, 'ro', marker='o', label='temperature data')

    # estimating parameters still a basic implementation read notes in fit for details

    # interpolating temp
    t_t, temp_data = t.interp(time0, time1, dt)
    para_t, cov_t = t.fit(t_t, temp_data, dt, x0_t, pressure_fit, p0, t0)

    # numerical solution and plotting
    timeT_fit, temp_fit = t.solve_ode(t.ode_model, time0, time1, dt, x0_t, pressure_fit, pars=[para_t[0], para_t[1], p0, t0])
    ln4 = ax2.plot(timeT_fit, temp_fit, 'r-', label='temperature best fit')

    print("for out first model {suitable} the temperature parameters are:")
    print("x0=", x0_t, "a=", para_t[0], " b=", para_t[1], "p0=", t0, "\n\n")

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=3)

    ax1.set_ylabel('pressure [bar]')
    ax2.set_ylabel('temperature [degC]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('Best fit model for given pressure and temperature data')


    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('best_fit.png', dpi=300)

if misfit:
    '''
    quantify the misfit between the best fit model and the data, then visualise it. 
    '''
    ############
    # PRESSURE # quantifying misfit for pressure LPM ODE
    ############
    fig, axes = plt.subplots(1, 2)
    ln1 = axes[0].plot(twl, p_plot, 'bo', marker='o', label='data')
    px_plot = np.interp(twl, time_fit, pressure_fit)
    ln2 = axes[0].plot(twl, px_plot, 'k-', label='ap = {a:.3f}\nbp = {b:.3f}\ncp = {c:.3f}'.format(a=para_p[0], b=para_p[1], c=para_p[2]))

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc=4)
    axes[0].set_ylabel('pressure [bar]')
    axes[0].set_xlabel('time [yr]')
    axes[0].set_title('best fit pressure LPM ODE model')

    p_misfit = px_plot - p_plot
    axes[1].plot(twl, np.zeros(len(twl)), 'k--')
    axes[1].plot(twl, p_misfit/100000, 'rx')

    axes[1].set_ylabel('pressure misfit [bar]')
    axes[1].set_xlabel('time [yr]')
    axes[1].set_title('best fit pressure LPM ODE model')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('best_pressure_misfit.png', dpi=300)


    ###############
    # TEMPERATURE # quantifying misfit for temperature LPM ODE
    ###############
    fig, axes = plt.subplots(1, 2)
    t_t, temp_data = t.interp(time0, time1, dt)

    ln1 = axes[0].plot(t_given, temp_given, 'bo', marker='o', label='data')
    Tx_plot = np.interp(t_given, timeT_fit, temp_fit)
    ln2 = axes[0].plot(t_given, Tx_plot, 'k-', label='at = {a:7e}\nbt = {b:.3f}'.format(a=para_t[0], b=para_t[1]))

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc=3)
    axes[0].set_ylabel('temperature [degC]')
    axes[0].set_xlabel('time [yr]')
    axes[0].set_title('best fit temperature LPM ODE model')

    T_misfit = Tx_plot - temp_given
    axes[1].plot(t_given, np.zeros(len(t_given)), 'k--')
    axes[1].plot(t_given, T_misfit, 'rx')

    axes[1].set_ylabel('temperature misfit [degC]')
    axes[1].set_xlabel('time [yr]')
    axes[1].set_title('best fit temperature LPM ODE model')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('best_temperature_misfit.png', dpi=300)

if validation:
    n = int(np.ceil(time1-time0))
    ts = time0 + np.arange(n+1)

    ############
    # PRESSURE #
    ############

    y_num = 0. * ts
    y_analytic = 0. * ts
    y_num[0] = 0.05

    for i in range(n):
        y_analytic[i] = p.analytical(i, 100, 0.1, 0.1, 0, 0.05)

        fk = p.ode_model(ts[i], y_num[i], 100, 0, 0.1, 0.1, 0, 0.05)
        fk1 = p.ode_model(ts[i] + 1, y_num[i] + 1 * fk, 100, 0, 0.1, 0.1, 0, 0.05)
        y_num[i + 1] = y_num[i] + 1 * ((fk + fk1) / 2)

    # setting the final value of the analytical solution
    y_analytic[64] = p.analytical(64, 100, 0.1, 0.1, 0, 0.05)

    fig, axs = plt.subplots(1, 2)
    ln1 = axs[0].plot(ts, y_num, 'kx', label='Numerical solution')
    ln2 = axs[0].plot(ts, y_analytic, 'b-', label='Analytical solution')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axs[0].legend(lns, labs, loc='upper right')
    axs[0].set_ylabel('x')
    axs[0].set_xlabel('t')
    axs[0].set_title('Pressure Benchmarking')

    # convergence
    h_array = np.linspace(0.1, 2, 29)
    t_values = []
    p_values = []

    # loop through each h-value
    for h in h_array:
        # solve using numerical method for each step size
        time_converge, p1 = p.solve_ode(p.ode_model, time0, time1, h, x0_p, 'SAME', pars=[para_p[0], para_p[1], para_p[2], p0])
        # store 1/h in the t array
        t_values.append(h)
        # store the pressure at year 2014 in the y-array
        p_values.append(p1[-1])

    # plot pressure against different values of h
    axs[1].plot(t_values, p_values, 'rx')
    axs[1].set_title('Pressure Convergence Analysis')
    axs[1].set_xlabel('Time step')
    axs[1].set_ylabel('Difference in final pressure value')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_validation.png', dpi=300)


    ###############
    # TEMPERATURE #
    ###############

if forecast:
    # forecasting pressure for different production scenarios from 2014 to time2

    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    # calculating the different scenarios
    t_forc, y_same = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'SAME',
                                 pars=[para_p[0], para_p[1], para_p[2], p0])
    y_stop = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'STOP', pars=[para_p[0], para_p[1], para_p[2], p0])[1]
    y_double = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'DOUBLE', pars=[para_p[0], para_p[1], para_p[2], p0])[1]
    y_half = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'HALF', pars=[para_p[0], para_p[1], para_p[2], p0])[1]

    # plotting the different scenarios against each other
    ln5 = ax1.plot(twl, p_plot / 100000, 'ro', marker='o', label='data')
    ln0 = ax1.plot(t_forc[:65], y_same[:65] / 100000, 'k-', label='best fit model')
    ln1 = ax1.plot(t_forc[64:], y_same[64:] / 100000, 'k-', label='maintained production')
    ln2 = ax1.plot(t_forc[64:], y_stop[64:] / 100000, 'g-', label='operation terminated')
    ln3 = ax1.plot(t_forc[64:], y_double[64:] / 100000, 'r-', label='production doubled')
    ln4 = ax1.plot(t_forc[64:], y_half[64:] / 100000, 'y-', label='production halved')

    lns = ln5 + ln0 + ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=4)
    ax1.set_ylabel('Pressure [bar]')
    ax1.set_xlabel('time [yr]')
    ax1.set_title('Pressure predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast.png', dpi=300)

    # rate of pressure change forecast
    # plotting format
    fig, ax2 = plt.subplots(nrows=1, ncols=1)
    t_pred = np.copy(t_forc[64:])
    dy_no_ch = np.gradient(np.copy(y_same[64:]/100000))
    dy_st = np.gradient(np.copy(y_stop[64:]/100000))
    dy_do = np.gradient(np.copy(y_double[64:]/100000))
    dy_ha = np.gradient(np.copy(y_half[64:]/100000))

    ln0 = ax2.plot(t_pred, np.zeros(len(t_pred)), 'k--', label='recovery affected')
    ln1 = ax2.plot(t_pred, dy_no_ch, 'k-', label='maintained production')
    ln2 = ax2.plot(t_pred, dy_st, 'g-', label='operation terminated')
    ln3 = ax2.plot(t_pred, dy_do, 'r-', label='production doubled')
    ln4 = ax2.plot(t_pred, dy_ha, 'y-', label='production halved')

    lns = ln0 + ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=4)
    ax2.set_ylabel('rate of pressure change')
    ax2.set_xlabel('time [yr]')
    ax2.set_title('rate of pressure change predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast_supplement.png', dpi=300)

    ###############
    # TEMPERATURE # predicting temperatures
    ###############
    f, ax1 = plt.subplots(nrows=1, ncols=1)

    tx, t_no_change = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_same, pars=[para_t[0], para_t[1], p0, t0])
    t_no_prod = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_stop, pars=[para_t[0], para_t[1], p0, t0])[1]
    t_double_prod = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_double, pars=[para_t[0], para_t[1], p0, t0])[1]
    t_half_prod = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_half, pars=[para_t[0], para_t[1], p0, t0])[1]

    # plotting the different scenarios against each other
    ln5 = ax1.plot(t_given, temp_given, 'ro', marker='o', label='data')
    ln0 = ax1.plot(tx[:65], t_no_change[:65], 'k-', label='best fit model')
    ln1 = ax1.plot(tx[64:], t_no_change[64:], 'k-', label='maintained production')
    ln2 = ax1.plot(tx[64:], t_no_prod[64:], 'g-', label='operation terminated')
    ln3 = ax1.plot(tx[64:], t_double_prod[64:], 'r-', label='production doubled')
    ln4 = ax1.plot(tx[64:], t_half_prod[64:], 'y-', label='production halved')

    lns = ln5 + ln0 + ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=3)
    ax1.set_ylabel('temperature [degC]')
    ax1.set_xlabel('time [yr]')
    ax1.set_title('Temperature predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_forecast.png', dpi=300)

    # rate of temperature change forecast
    # plotting format
    f, ax2 = plt.subplots(nrows=1, ncols=1)
    t_pred = np.copy(tx[64:])
    dt_no_ch = np.gradient(np.copy(t_no_change[64:]))
    dt_st = np.gradient(np.copy(t_no_prod[64:]))
    dt_do = np.gradient(np.copy(t_double_prod[64:]))
    dt_ha = np.gradient(np.copy(t_half_prod[64:]))

    ln0 = ax2.plot(t_pred, np.zeros(len(t_pred)), 'k--', label='recovery affected')
    ln1 = ax2.plot(t_pred, dt_no_ch, 'k-', label='maintained production')
    ln2 = ax2.plot(t_pred, dt_st, 'g-', label='operation terminated')
    ln3 = ax2.plot(t_pred, dt_do, 'r-', label='production doubled')
    ln4 = ax2.plot(t_pred, dt_ha, 'y-', label='production halved')

    lns = ln0 + ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=3)
    ax2.set_ylabel('rate of temperature change')
    ax2.set_xlabel('time [yr]')
    ax2.set_title('rate of temperature change predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_forecast_supplement.png', dpi=300)

if inversion:
    a = para_p[0]/100000
    b = para_p[1]/100000
    c = para_p[2]/100000

    g = 9.81
    area = 15e+6
    s0 = 0.3

    print("The estimated porosity through inverse modelling is:")
    print((g * (a - b*c))/((a**2) * area * (1 - s0)), "\n")

if uncertainty:

    # plotting pressure forecast with uncertainty
    fig, ax = plt.subplots(nrows=1, ncols=1)

    np.random.seed(1)
    ps = np.random.multivariate_normal(para_p, cov_p, 100)
    for pi in ps:
        temp_hold, pressure_hold = p.solve_ode(p.ode_model, time0, time1, dt, x0_p, 'SAME', pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(temp_hold, pressure_hold / 100000, 'k-', alpha=0.2, lw=0.5, label='best fit model')

        temp_hold, pressure_hold = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'SAME', pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(temp_hold[64:], pressure_hold[64:] / 100000, 'k-', alpha=0.2, lw=0.5, label='maintained production')

        temp_hold, pressure_hold = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'STOP', pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(temp_hold[64:], pressure_hold[64:] / 100000, 'g-', alpha=0.2, lw=0.5, label='operation terminated')

        temp_hold, pressure_hold = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'DOUBLE', pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(temp_hold[64:], pressure_hold[64:] / 100000, 'r-', alpha=0.2, lw=0.5, label='production doubled')

        temp_hold, pressure_hold = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'HALF', pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(temp_hold[64:], pressure_hold[64:] / 100000, 'y-', alpha=0.2, lw=0.5, label='production halved')

    v = 0.03
    wl = (wl * 997 * 9.81) - 2909250 + 5000

    ax.errorbar(twl, wl / 100000, yerr=v, fmt='ro', label='data')

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    ax.set_ylabel('pressure [bar]')
    ax.set_xlabel('time [yr]')
    ax.set_title('pressure forecast with uncertainty')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast_uncertainty.png', dpi=300)


    # rate of pressure change forecast with uncertainty
    f, ax = plt.subplots(nrows=1, ncols=1)

    for pi in ps:
        temp_hold, pressure_hold = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'SAME', pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(temp_hold[64:], np.gradient(pressure_hold[64:] / 100000), 'k-', alpha=0.2, lw=0.5, label='maintained production')

        temp_hold, pressure_hold = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'STOP',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(temp_hold[64:], np.gradient(pressure_hold[64:] / 100000), 'g-', alpha=0.2, lw=0.5, label='operation terminated')

        temp_hold, pressure_hold = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'DOUBLE',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(temp_hold[64:], np.gradient(pressure_hold[64:] / 100000), 'r-', alpha=0.2, lw=0.5, label='production doubled')

        temp_hold, pressure_hold = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'HALF',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(temp_hold[64:], np.gradient(pressure_hold[64:] / 100000), 'y-', alpha=0.2, lw=0.5, label='production halved')

    ax.plot(temp_hold[64:], np.zeros(len(temp_hold[64:])), 'k--', alpha=0.2, lw=0.5, label='recovery affected')

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    ax.set_ylabel('rate of pressure change')
    ax.set_xlabel('time [yr]')
    ax.set_title('rate of pressure change forecast with uncertainty')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast_uncertainty_supplement.png', dpi=300)

    # temperature forecast with uncertainty
    f, ax = plt.subplots(nrows=1, ncols=1)

    ps = np.random.multivariate_normal(para_t, cov_t, 100)
    for pi in ps:
        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time1, dt, x0_t, y_same, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold, temp_hold, 'k-', alpha=0.2, lw=0.5, label='best fit model')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_same, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], temp_hold[64:], 'k-', alpha=0.2, lw=0.5, label='maintained production')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_stop, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], temp_hold[64:], 'g-', alpha=0.2, lw=0.5, label='operation terminated')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_double, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], temp_hold[64:], 'r-', alpha=0.2, lw=0.5, label='production doubled')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_half, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], temp_hold[64:], 'y-', alpha=0.2, lw=0.5, label='production halved')

    v = 1
    ax.errorbar(t_given, temp_given, yerr=v, fmt='ro', label='data')

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    ax.set_ylabel('temperature [degC]')
    ax.set_xlabel('time [yr]')
    ax.set_title('temperature forecast with uncertainty')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_forecast_uncertainty.png', dpi=300)

    # rate of temperature change forecast with uncertainty
    f, ax = plt.subplots(nrows=1, ncols=1)

    for pi in ps:
        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_same, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], np.gradient(temp_hold[64:]), 'k-', alpha=0.2, lw=0.5, label='maintained production')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_stop, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], np.gradient(temp_hold[64:]), 'g-', alpha=0.2, lw=0.5, label='operation terminated')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_double, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], np.gradient(temp_hold[64:]), 'r-', alpha=0.2, lw=0.5, label='production doubled')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_half, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], np.gradient(temp_hold[64:]), 'y-', alpha=0.2, lw=0.5, label='production halved')

    ax.plot(time_hold[64:], np.zeros(len(temp_hold[64:])), 'k--', alpha=0.2, lw=0.5, label='recovery affected')

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    ax.set_ylabel('rate of temperature change')
    ax.set_xlabel('time [yr]')
    ax.set_title('rate of temperature change forecast with uncertainty')

    # EITHER show the plot to the screen OR save a version of it to the disk
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_forecast_uncertainty_supplement.png', dpi=300)