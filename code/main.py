# imports
import pressure as p
import temperature as t
from matplotlib import pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(1)
ax2 = ax1.twinx()

step_p = 0.5  # step size
initial_p = 5000 # starting pressure value, PA
initial_t = 149
points = int(1/step_p)

time_p = np.linspace(1950, 2014, 65*points - 1)
water_pressure = p.calculate(time_p, step_p)  # remember to implement initial value later (300)
ax1.plot(time_p, water_pressure, 'bo', marker='o')
ap, bp, cp = p.fit(time_p, water_pressure, step_p, initial_p)

time_fit, pressure_fit = p.solve_ode(p.ode_model, time_p, step_p, initial_p, 'SAME', pars=[ap, bp, cp])
ax1.plot(time_fit, pressure_fit, 'b--')

time, temp_data = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # temperature (gr_t data)

# need to convert pressure to 11 points so it actually fits???
# fucking for loop

step_t = 5
time_t, temp_data = t.calculate(time, step_t)

pressure_t = []
pressure_i = []
for i in range(len(time_t)):
    for j in range(len(time_p)):
        if time_t[i] == time_p[j]:
            pressure_t.append(pressure_fit[j])
            pressure_i.append(pressure_fit[j-1])


ax2.plot(time, temp_data, 'ro', marker='o')
#alpha, bp = t.fit(time_t, temp_data, step_t, initial_t, pressure_t, pressure_i)


plt.show() #for testing
