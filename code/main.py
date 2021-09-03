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
ax1.plot(t_p, pressure_data, 'bo', marker='o')

# estimating parameters still a basic implementation read note in fit for details
ap, bp, cp = p.fit(t_p, pressure_data, dt_p, x0_p, p0)

# numerical solution and plotting
time_fit, pressure_fit = p.solve_ode(p.ode_model, time0, time1, dt_p, x0_p, 'SAME', pars=[ap, bp, cp, p0])
ax1.plot(time_fit, pressure_fit, 'b--')


# TEMPERATURE
dt_t = 1                                    # step size
x0_t = 149                                  # starting temperature
t0 = 147                                    # temperature outside CV/ conduction source

# interpolating temp will also look weird will fix later
t_t, temp_data = t.interp(time0, time1, dt_t)
ax2.plot(t_t, temp_data, 'ro', marker='o')

# estimating parameters still a basic implementation read notes in fit for details
alpha, bt = t.fit(t_t, temp_data, dt_t, x0_t, pressure_fit, p0, t0)

# numerical solution and plotting
timeT_fit, temp_fit = t.solve_ode(t.ode_model, time0, time1, dt_t, x0_t, pressure_fit, pars=[alpha, bt, p0, t0])
ax2.plot(timeT_fit, temp_fit, 'r--')

# this is the only plotting that should be in the main file because this plot requires data from temp and pressure
# all other plots should be in their respective parent file e.g. pressure scenarios in pressure
plt.show()

# functions for plotting the scenarios and whatever else will go here but the actual code will be in their respect func.



