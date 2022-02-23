# Parameter set-up
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from tqdm import trange

g = 1
omega = 1
R0 = 2.9
Rbar = R0 + g / (omega ** 2)

# Initial conditions
# theta_0 = np.random.rand() * 0.1 - 0.2
# r_0 = Rbar * (1 + np.random.rand() * 0.1 - 0.2)
x_0 = Rbar * 0.01
z_0 = -Rbar * 1.1


def implicit_midpoint_integrator(f_func, energy_func, y_0, ts, solver=scipy.optimize.anderson):
    y = y_0
    y_history = np.zeros((len(ts), len(y)))
    y_history[0, :] = y

    energy_history = np.zeros((len(ts),))
    energy_history[0] = energy_func(y)

    for i in trange(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        t_mid = ts[i - 1] + dt / 2

        try:

            y = solver(lambda y_new: y_new - y - dt * f_func(t_mid, (y_new + y) / 2), y, f_tol=1e-7, f_rtol=1e-6)
        except Exception:
            break
        # y_old = y.copy()
        # while True:
        #     y_new = y_old + dt * f_func(t_mid, (y + y_old) / 2)
        #     if np.linalg.norm(y_new - y) < 1e-14:
        #         break
        #     y = y_new
        y_history[i, :] = y
        energy_history[i] = energy_func(y)

    return y, y_history, energy_history


def system_source(_, y):
    x, xdot, z, zdot = y[0], y[1], y[2], y[3]
    return np.array([
        xdot,
        - omega ** 2 * x * (1 - R0 / np.sqrt(x ** 2 + z ** 2)),
        zdot,
        -g - omega ** 2 * z * (1 - R0 / np.sqrt(x ** 2 + z ** 2)),
    ])


def energy(y):
    x, xdot, z, zdot = y[0], y[1], y[2], y[3]
    return g * z + 0.5 * omega ** 2 * (np.sqrt(x ** 2 + z ** 2) - R0) ** 2 + 0.5 * (xdot ** 2 + zdot ** 2)


ts = np.arange(0, 500, 0.5)
_, trajectory, energies = implicit_midpoint_integrator(system_source, energy, np.array([x_0, 0, z_0, 0]), ts)

fig, axs = plt.subplots(5, 1, figsize=(9, 9))
labels = ['$x$', '$\\dot{x}$', '$y$', '$\\dot{y}$']
for j in range(4):
    axs[j].plot(ts, trajectory[:, j])
    axs[j].set(ylabel=labels[j])
axs[-1].plot(ts, energies)
axs[-1].set(xlabel='t', ylabel='energy')
plt.show()
