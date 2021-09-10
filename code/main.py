# imports
import pressure as p
import temperature as t
from matplotlib import pyplot as plt
import numpy as np
import tests as tests


############
# PLOTTING #    make sure all are true for submission
############
given1 = False  # plotting the water level and total production rate (with reinjection rate considered).
given2 = False  # conversion from water level to pressure.
given3 = False  # plotting the temperature and total production rate (with reinjection rate considered).
validation = False  # plot benchmarking and convergence test.
suitable = False
improve = True # plot pressure and temperature bestfit LPM ODE models. MUST REMAIN TRUE TO RUN PLOTS THAT FOLLOWS.
inversion = True
forecast = True  # plot pressure and temperature forecast to 2050, as well as respective change rate forecast. MUST REMAIN TRUE TO RUN PLOTS THAT FOLLOWS.
misfit = False  # plot quantified misfit of the model to data.
uncertainty = False  # plot of pressure and temperature forecast uncertainty.


##################
# INITIALISATION #       these variables are reused between functions
##################
tq1, pr1 = np.genfromtxt('gr_q1.txt', delimiter=',', skip_header=1).T  # production rate 1 (gr_q1 data)
tq2, pr2 = np.genfromtxt('gr_q2.txt', delimiter=',', skip_header=1).T  # production rate 2 (gr_q2 data)
tPr, wl = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=1).T  # water level (gr_p data)
tTe, temp = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # Temperature (gr_T data)

# time values and relative step
time0 = 1950
time1 = 2014
time2 = 2050
dt = 1

# the influence of borehole closure program on the water level recovery
if given1:
    # plotting format 
    f, ax1 = plt.subplots(nrows=1, ncols=1)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(tq1, p.interpolate_q_total(tq1), 'k-', label='total production rate')
    # ln2 = ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2')
    ln3 = ax2.plot(tPr, wl, 'r-', label="water level", markersize=7)

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
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('given_data_1.png', dpi=300)

# conversion from water level to pressure 
if given2:
    # plotting format 
    f, ax1 = plt.subplots(nrows=1, ncols=1)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(tq1, p.interpolate_q_total(tq1), 'k-', label='total production rate')
    # ln2 = ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2')

    # converting water level to pressure and plotting, the plot is gonna look weird because I changed it for a
    # better fit will fix this graph later this is not that important right now
    t_p, pressure_data = p.interp(time0, time1, dt, True)

    tp_provided, p_provided = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=1).T  # given pressure data  
    p_plot = np.interp(tp_provided, t_p,
                       pressure_data)  # the pressure calculated from water level is interpolated to match the time we are provided
    ln3 = ax2.plot(tp_provided, p_plot, 'bo', marker='o', markersize=7,
                   label='pressure data')  # pressure data is converted from Pa to bar (1 Pa = 1.e-5 bar)

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
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('given_data_2.png', dpi=300)

# the influence of borehole closure program on the temperature recovery
if given3:
    # plotting format 
    f, ax1 = plt.subplots(nrows=1, ncols=1)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(tq1, p.interpolate_q_total(tq1), 'k-', label='total production rate')
    # ln2 = ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2') # if rhyolite data is plotted as well
    # ax1.plot(tT, Temp, 'r*', label='Temperature', markersize = 7)
    ln3 = ax2.plot(tTe, temp, 'r*', label='Temperature', markersize=7)

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
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('given_data_3.png', dpi=300)

if validation:
    # benchmarking
    n = int(np.ceil(2014 - 1950))  # number of steps
    ts = 1950 + np.arange(n + 1)  # time array

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

    fig, axes = plt.subplots(1, 2)
    ln1 = axes[0].plot(ts, y_num, 'kx', label='numerical solution')
    ln2 = axes[0].plot(ts, y_analytic, 'b-', label='analytical solution')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc=4)
    axes[0].set_ylabel('x')
    axes[0].set_xlabel('t')
    axes[0].set_title('benchmarking')

    # convergence test still needs to be done

    # axes[1].plot(t, np.zeros(len(tp_provided)), 'k--')
    # axes[1].plot(t, , 'rx')

    axes[1].set_ylabel('pressure misfit [bar]')
    axes[1].set_xlabel('time [yr]')
    axes[1].set_title('best fit pressure LPM ODE model')

    ###############
    # TEMPERATURE #
    ###############

    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_misfit.png', dpi=300)

if suitable:
    fig, ax1 = plt.subplots(1)
    ax2 = ax1.twinx()

    ############
    # PRESSURE #
    ############
    x0_p = 0.05  # starting pressure value, PA
    p0 = 0  # hydrostatic pressure at recharge source assuming shallow inflow scott2016 pg297

    t_p, pressure_data = p.interp(time0, time1, dt, False)

    tp_provided, p_provided = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=1).T  # given pressure data
    p_plot = np.interp(tp_provided, t_p, pressure_data)     # the pressure calculated from water level is interpolatedmto match the time we are provided
    ln1 = ax1.plot(t_p, pressure_data, 'bo', marker='o', label='pressure data')  # pressure data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    # estimating parameters READ NOTES
    para_p, cov_p = p.fit(t_p, pressure_data, dt, x0_p, p0)

    print("for out first model {suitable} the pressure ode parameters are:")
    print("x0=", x0_p, "a=", para_p[0], " b=", para_p[1], "c=", para_p[2], "p0=", p0, "\n")

    # numerical solution and plotting
    time_fit, pressure_fit = p.solve_ode(p.ode_model, time0, time1, dt, x0_p, 'SAME',
                                         pars=[para_p[0], para_p[1], para_p[2], p0])
    ln2 = ax1.plot(time_fit, pressure_fit, 'b-',
                   label='pressure best fit')  # pressure_fit data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    ###############
    # TEMPERATURE #
    ###############
    dt_t = 1  # step size
    x0_t = 149  # starting temperature
    t0 = 147 # temperature outside CV/ conduction source

    # interpolating temp will also look weird will fix later
    t_t, temp_data = t.interp(time0, time1, dt_t)

    tT_provided, T_provided = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # given temperature data
    ln3 = ax2.plot(tT_provided, T_provided, 'ro', marker='o', label='temperature data')

    # estimating parameters still a basic implementation read notes in fit for details
    para_t, _ = t.fit(t_t, temp_data, dt_t, x0_t, pressure_fit, p0, t0)

    print("for out first model {suitable} the temperature parameters are:")
    print("x0=", x0_t, "a=", para_t[0], " b=", para_t[1], "p0=", t0, "\n\n")

    # numerical solution and plotting
    timeT_fit, temp_fit = t.solve_ode(t.ode_model, time0, time1, dt_t, x0_t, pressure_fit,
                                      pars=[para_t[0], para_t[1], p0, t0])
    ln4 = ax2.plot(timeT_fit, temp_fit, 'r-', label='temperature best fit')

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=3)

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
        plt.savefig('best_fit.png', dpi=300)

if improve:
    fig, ax1 = plt.subplots(1)
    ax2 = ax1.twinx()

    ############
    # PRESSURE #
    ############
    x0_p = 0.05  # starting pressure value, PA
    p0 = -0.03  # hydrostatic pressure at recharge source assuming shallow inflow scott2016 pg297

    # converting water level to pressure and plotting, the plot is gonna look weird because I changed it for a
    # better fit will fix this graph later this is not that important right now
    t_p, pressure_data = p.interp(time0, time1, dt, True)

    tp_provided, p_provided = np.genfromtxt('gr_p.txt', delimiter=',', skip_header=1).T  # given pressure data
    p_plot = np.interp(tp_provided, t_p,
                       pressure_data)  # the pressure calculated from water level is interpolated to match the time we are provided
    ln1 = ax1.plot(tp_provided, p_plot, 'bo', marker='o',
                   label='pressure data')  # pressure data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    # estimating parameters still a basic implementation read note in fit for details
    para_p, cov_p = p.fit(t_p, pressure_data, dt, x0_p, p0)

    print("for out improved model the pressure ode parameters are:")
    print("x0=", x0_p, "a=", para_p[0], " b=", para_p[1], "c=", para_p[2], "p0=", p0, "\n")

    # numerical solution and plotting
    time_fit, pressure_fit = p.solve_ode(p.ode_model, time0, time1, dt, x0_p, 'SAME',
                                         pars=[para_p[0], para_p[1], para_p[2], p0])
    ln2 = ax1.plot(time_fit, pressure_fit, 'b-',
                   label='pressure best fit')  # pressure_fit data is converted from Pa to bar (1 Pa = 1.e-5 bar)

    ###############
    # TEMPERATURE #
    ###############
    dt_t = 1  # step size
    x0_t = 149  # starting temperature
    t0 = 145.9  # temperature outside CV/ conduction source

    # interpolating temp will also look weird will fix later
    t_t, temp_data = t.interp(time0, time1, dt_t)

    tT_provided, T_provided = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # given temperature data
    ln3 = ax2.plot(tT_provided, T_provided, 'ro', marker='o', label='temperature data')

    # estimating parameters still a basic implementation read notes in fit for details
    para_t, cov_t = t.fit(t_t, temp_data, dt_t, x0_t, pressure_fit, p0, t0)

    print("for our improved model the temperature parameters are:")
    print("x0=", x0_t, "a=", para_t[0], " b=", para_t[1], "p0=", t0, "\n\n")

    # numerical solution and plotting
    timeT_fit, temp_fit = t.solve_ode(t.ode_model, time0, time1, dt_t, x0_t, pressure_fit,
                                      pars=[para_t[0], para_t[1], p0, t0])
    ln4 = ax2.plot(timeT_fit, temp_fit, 'r-', label='temperature best fit')

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=3)

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
        plt.savefig('best_fit.png', dpi=300)

if forecast:
    # forecasting pressure for different production scenarios from 2014 to 2050

    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    # calculating the different scenarios
    t_forc, y_same = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'SAME', pars=[para_p[0], para_p[1], para_p[2], p0])
    y_stop = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'STOP', pars=[para_p[0], para_p[1], para_p[2], p0])[1]
    y_double = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'DOUBLE', pars=[para_p[0], para_p[1], para_p[2], p0])[1]
    y_half = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'HALF', pars=[para_p[0], para_p[1], para_p[2], p0])[1]

    # plotting the different scenarios against each other
    ln1 = ax1.plot(t_forc, y_same, 'k-', label='maintained production')
    ln2 = ax1.plot(t_forc, y_stop, 'r-', label='operation terminated')
    ln3 = ax1.plot(t_forc, y_double, 'g-', label='production doubled')
    ln4 = ax1.plot(t_forc, y_half, 'b-', label='production halved')

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=4)
    ax1.set_ylabel('Pressure [MPa]')
    ax1.set_xlabel('time [yr]')
    ax1.set_title('Pressure predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast.png', dpi=300)

    # rate of pressure change forecast
    # plotting format
    fig, ax2 = plt.subplots(nrows=1, ncols=1)
    t_pred = np.copy(t_forc[64:])
    dy_no_ch = np.gradient(np.copy(y_same[64:]))
    dy_st = np.gradient(np.copy(y_stop[64:]))
    dy_do = np.gradient(np.copy(y_double[64:]))
    dy_ha = np.gradient(np.copy(y_half[64:]))

    ln0 = ax2.plot(t_pred, np.zeros(len(t_pred)), 'k--', label='recovery affected')
    ln1 = ax2.plot(t_pred, dy_no_ch, 'k-', label='maintained production')
    ln2 = ax2.plot(t_pred, dy_st, 'r-', label='operation terminated')
    ln3 = ax2.plot(t_pred, dy_do, 'g-', label='production doubled')
    ln4 = ax2.plot(t_pred, dy_ha, 'b-', label='production halved')

    lns = ln0 + ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=4)
    ax2.set_ylabel('rate of pressure change')
    ax2.set_xlabel('time [yr]')
    ax2.set_title('rate of pressure change predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast_supplement.png', dpi=300)


    ###############
    # TEMPERATURE #
    ###############
    f, ax1 = plt.subplots(nrows=1, ncols=1)

    tx, t_no_change = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_same, pars=[para_t[0], para_t[1], p0, t0])
    t_no_prod = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_stop, pars=[para_t[0], para_t[1], p0, t0])[1]
    t_double_prod = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_double, pars=[para_t[0], para_t[1], p0, t0])[1]
    t_half_prod = t.solve_ode(t.ode_model, time0, time2, dt, x0_t, y_half, pars=[para_t[0], para_t[1], p0, t0])[1]

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
    ln2 = ax2.plot(t_pred, dt_st, 'r-', label='operation terminated')
    ln3 = ax2.plot(t_pred, dt_do, 'g-', label='production doubled')
    ln4 = ax2.plot(t_pred, dt_ha, 'b-', label='production halved')

    lns = ln0 + ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=3)
    ax2.set_ylabel('rate of temperature change')
    ax2.set_xlabel('time [yr]')
    ax2.set_title('rate of temperature change predictions for different scenarios from 2014')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_forecast_supplement.png', dpi=300)




if misfit:
    '''
    quantify the misfit between the best fit model and the data, then visualise it. 
    '''
    # quantifying misfit for pressure LPM ODE
    fig, axes = plt.subplots(1, 2)
    ln1 = axes[0].plot(tp_provided, p_plot, 'bo', marker='o', label='data')
    px_plot = np.interp(tp_provided, time_fit, pressure_fit)
    ln2 = axes[0].plot(tp_provided, px_plot, 'k-', label='ap = 9.110\nbp = 1.116e-1\ncp = 6.571e+1')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc=4)
    axes[0].set_ylabel('pressure [bar]')
    axes[0].set_xlabel('time [yr]')
    axes[0].set_title('best fit pressure LPM ODE model')

    p_misfit = px_plot - p_plot
    axes[1].plot(tp_provided, np.zeros(len(tp_provided)), 'k--')
    axes[1].plot(tp_provided, p_misfit, 'rx')

    axes[1].set_ylabel('pressure misfit [bar]')
    axes[1].set_xlabel('time [yr]')
    axes[1].set_title('best fit pressure LPM ODE model')

    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_misfit.png', dpi=300)

    ###############
    # TEMPERATURE # quantifying misfit for temperature LPM ODE
    ###############
    fig, axes = plt.subplots(1, 2)
    ln1 = axes[0].plot(tT_provided, T_provided, 'bo', marker='o', label='data')
    Tx_plot = np.interp(tT_provided, timeT_fit, temp_fit)
    ln2 = axes[0].plot(tT_provided, Tx_plot, 'k-', label='at = 6.058e-7\nbt = 8.288e-2')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc=3)
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
        plt.savefig('temperature_misfit.png', dpi=300)

if inversion:
    g = 9.81        #
    A = 15e+6       # given between 15-28
    S0 = 0.3        # value of water idk

    print("the estimated porosity through inverse modelling is:")
    print((g * (para_p[0] - para_p[1] * para_p[2])) / ((para_p[0] ** 2) * A * (1 - S0)))

if uncertainty:
    ############
    # PRESSURE #
    ############
    # plotting pressure forecast with uncertainty
    f, ax = plt.subplots(nrows=1, ncols=1)

    ps = np.random.multivariate_normal(para_p, cov_p, 100)
    for pi in ps:
        time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time1, dt, x0_p, 'SAME',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(time_temp, pressure_temp, 'k-', alpha=0.2, lw=0.5, label='best fit model')

        time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'SAME',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(time_temp[64:], pressure_temp[64:], 'r-', alpha=0.2, lw=0.5, label='maintained production')

        time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'STOP',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(time_temp[64:], pressure_temp[64:], 'b-', alpha=0.2, lw=0.5, label='operation terminated')

        time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'DOUBLE',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(time_temp[64:], pressure_temp[64:], 'g-', alpha=0.2, lw=0.5, label='production doubled')

        time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'HALF',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(time_temp[64:], pressure_temp[64:], 'y-', alpha=0.2, lw=0.5, label='production halved')

    v = 0.03
    p_provided = (((p_provided - 297.4) * 997 * 9.81) + 5000) / 100000
    ax.errorbar(tp_provided, p_provided, yerr=v, fmt='ro', label='data')

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    ax.set_ylabel('pressure [bar]')
    ax.set_xlabel('time [yr]')
    ax.set_title('pressure forecast with uncertainty')

    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast_uncertainty.png', dpi=300)

    ###############
    # TEMPERATURE # rate of temperature change forecast with uncertainty
    ###############
    f, ax = plt.subplots(nrows=1, ncols=1)

    ps = np.random.multivariate_normal(para_p, cov_p, 100)
    for pi in ps:
        time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'SAME',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(time_temp[64:], np.gradient(pressure_temp[64:]), 'r-', alpha=0.2, lw=0.5, label='maintained production')

        time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'STOP',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(time_temp[64:], np.gradient(pressure_temp[64:]), 'b-', alpha=0.2, lw=0.5, label='operation terminated')

        time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'DOUBLE',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(time_temp[64:], np.gradient(pressure_temp[64:]), 'g-', alpha=0.2, lw=0.5, label='production doubled')

        time_temp, pressure_temp = p.solve_ode(p.ode_model, time0, time2, dt, x0_p, 'HALF',
                                               pars=[pi[0], pi[1], pi[2], p0])
        ax.plot(time_temp[64:], np.gradient(pressure_temp[64:]), 'y-', alpha=0.2, lw=0.5, label='production halved')

    ax.plot(time_temp[64:], np.zeros(len(time_temp[64:])), 'k--', alpha=0.2, lw=0.5, label='recovery affected')

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    ax.set_ylabel('rate of pressure change')
    ax.set_xlabel('time [yr]')
    ax.set_title('rate of pressure change forecast with uncertainty')

    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('pressure_forecast_uncertainty_supplement.png', dpi=300)

    # temperature forecast with uncertainty
    f, ax = plt.subplots(nrows=1, ncols=1)

    ps = np.random.multivariate_normal(para_t, cov_t, 100)
    for pi in ps:
        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time1, dt_t, x0_t, y_same,
                                           pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold, temp_hold, 'k-', alpha=0.2, lw=0.5, label='best fit model')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt_t, x0_t, y_same,
                                           pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], temp_hold[64:], 'y-', alpha=0.2, lw=0.5, label='maintained production')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt_t, x0_t, y_stop, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], temp_hold[64:], 'r-', alpha=0.2, lw=0.5, label='operation terminated')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt_t, x0_t, y_double, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], temp_hold[64:], 'b-', alpha=0.2, lw=0.5, label='production doubled')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt_t, x0_t, y_half, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], temp_hold[64:], 'g-', alpha=0.2, lw=0.5, label='production halved')

    v = 1
    ax.errorbar(tT_provided, T_provided, yerr=v, fmt='ro', label='data')

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    ax.set_ylabel('temperature [degC]')
    ax.set_xlabel('time [yr]')
    ax.set_title('temperature forecast with uncertainty')

    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_forecast_uncertainty.png', dpi=300)

    # rate of temperature change forecast with uncertainty
    f, ax = plt.subplots(nrows=1, ncols=1)

    ps = np.random.multivariate_normal(para_t, cov_t, 100)
    for pi in ps:
        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt_t, x0_t, y_same,
                                           pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], np.gradient(temp_hold[64:]), 'y-', alpha=0.2, lw=0.5, label='maintained production')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt_t, x0_t, y_stop, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], np.gradient(temp_hold[64:]), 'r-', alpha=0.2, lw=0.5, label='operation terminated')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt_t, x0_t, y_double, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], np.gradient(temp_hold[64:]), 'b-', alpha=0.2, lw=0.5, label='production doubled')

        time_hold, temp_hold = t.solve_ode(t.ode_model, time0, time2, dt_t, x0_t, y_half, pars=[pi[0], pi[1], p0, t0])
        ax.plot(time_hold[64:], np.gradient(temp_hold[64:]), 'g-', alpha=0.2, lw=0.5, label='production halved')

    ax.plot(time_temp[64:], np.zeros(len(time_temp[64:])), 'k--', alpha=0.2, lw=0.5, label='recovery affected')

    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
    ax.set_ylabel('rate of temperature change')
    ax.set_xlabel('time [yr]')
    ax.set_title('rate of temperature change forecast with uncertainty')

    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_forecast_uncertainty_supplement.png', dpi=300)
