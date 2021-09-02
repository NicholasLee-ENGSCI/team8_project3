# imports
import pressure as p
import temperature as t
from matplotlib import pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(1)
ax2 = ax1.twinx()

step_p = 0.5  # step size
initial_p = 5000  # starting pressure value, PA
points = int(1 / step_p)

time_p = np.linspace(1950, 2014, 65 * points - 1)
water_pressure = p.calculate(time_p, step_p)  # remember to implement initial value later (300)
ax1.plot(time_p, water_pressure, 'bo', marker='o')
ap, bp, cp = p.fit(time_p, water_pressure, step_p, initial_p)
p0 = 0

time_fit, pressure_fit = p.solve_ode(p.ode_model, time_p, step_p, initial_p, 'SAME', pars=[ap, bp, cp, p0])
ax1.plot(time_fit, pressure_fit, 'b--')

time, temp_data = np.genfromtxt('gr_T.txt', delimiter=',', skip_header=1).T  # temperature (gr_t data)

# need to convert pressure to 11 points so it actually fits???
# fucking for loop

step_t = 5
initial_t = 149
t0 = 147
time_t, temp_data = t.calculate(time, step_t)

pressure_t = []
for i in range(len(time_t)):
    for j in range(len(time_p)):
        if time_t[i] == time_p[j]:
            pressure_t.append(pressure_fit[j])

ax2.plot(time, temp_data, 'ro', marker='o')
alpha, bt = t.fit(time_t, temp_data, step_t, initial_t, pressure_t, p0, t0)

#alpha = 0.000001
#bt = 0.08

timet_fit, temp_fit = t.solve_ode(t.ode_model, time_t, step_t, initial_t, pressure_t, pars=[alpha, bt, p0, t0])
ax2.plot(timet_fit, temp_fit, 'r--')


plt.show()  # for testing
