# imports
import pressure as p
import temperature as t
from matplotlib import pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(1)
ax2 = ax1.twinx()

time = np.linspace(1950, 2014, 65)
water_pressure = p.calculate(time, 300)  # remember to implement initial value later (300)

ax1.plot(time, water_pressure, 'bo', marker='o')

step = 1  # step size
initial = water_pressure[0]  # starting pressure value (change this to a researched value later)

ap, bp, cp = p.fit(time, water_pressure, step, initial)

time_fit, pressure_fit = p.solve_ode(p.ode_model, time, step, initial, 'SAME', pars=[ap, bp, cp])
ax1.plot(time_fit, pressure_fit, 'b--')

plt.show() #for testing
