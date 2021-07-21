
# imports 
from matplotlib import pyplot as plt    # MATPLOTLIB is THE plotting module for Python
import numpy as np

# boolean variable to run the dual plot 
run_plot1 = True # plotting the production rates 1 and 2, with water level 
run_plot2 = True # plotting the production rates 1 and 2, with temperature 

# the influence of borehole closure program on the water level recovery
if run_plot1:
    # File I/O commands to read in the data
    tq1,pr1 = np.genfromtxt('gr_q1.txt',delimiter=',',skip_header=1).T # production rate 1 (gr_q1 data)
    tq2,pr2 = np.genfromtxt('gr_q2.txt',delimiter=',',skip_header=1).T # production rate 2 (gr_q2 data)
    tp,wl = np.genfromtxt('gr_p.txt',delimiter=',',skip_header=1).T # water level (gr_p data)

    # plotting format 
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax2 = ax1.twinx()
    ax1.plot(tq1, pr1, 'k-', label='Total extraction rate 1')
    ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2')

    ax2.plot(tp, wl, 'r*', markersize = 7)
    # ax2.set_ylim([0,300])
    ax1.legend(loc=2)
    # ax1.set_ylim([0,35000])
    ax1.set_ylabel('production rate [tonnes/day]')
    ax2.set_ylabel('water level [m]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('The influence of borehole closure program on the water level recovery')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('water_level_plot.png',dpi=300)


# the influence of borehole closure program on the temperature recovery
if run_plot2:
    # File I/O commands to read in the data
    tq1,pr1 = np.genfromtxt('gr_q1.txt',delimiter=',',skip_header=1).T # production rate 1 (gr_q1 data)
    tq2,pr2 = np.genfromtxt('gr_q2.txt',delimiter=',',skip_header=1).T # production rate 2 (gr_q2 data)
    tT,Temp = np.genfromtxt('gr_T.txt',delimiter=',',skip_header=1).T # Temperature (gr_T data)

    # plotting format 
    f,ax1 = plt.subplots(nrows=1,ncols=1)
    ax2 = ax1.twinx()
    ax1.plot(tq1, pr1, 'k-', label='Total extraction rate 1')
    ax1.plot(tq2, pr2, 'b-', label='Total extraction rate 2')

    ax2.plot(tT, Temp, 'r*', markersize = 7)
    # ax2.set_ylim([0,300])
    ax1.legend(loc=1)
    # ax1.set_ylim([0,35000])
    ax1.set_ylabel('production rate [tonnes/day]')
    ax2.set_ylabel('Temperature [degC]')
    ax2.set_xlabel('time [yr]')
    ax1.set_xlabel('time [yr]')
    ax2.set_title('The influence of borehole closure program on the temperature recovery')

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig('temperature_plot.png',dpi=300)