
# imports 
from matplotlib import pyplot as plt    # MATPLOTLIB is THE plotting module for Python
import numpy as np
import pressure as p

# boolean variable to run the dual plot 
run_plot1 = True # plotting the production rates 1 and 2, with water level 
run_plot2 = True # conversion from water level to pressure 
run_plot3 = True # plotting the production rates 1 and 2, with temperature 

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