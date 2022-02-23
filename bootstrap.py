# Parameter set-up
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from tqdm import trange

overall = 0.75
g = 1 * overall
omega = 2
R0 = (2 ** 2 - 1) / omega ** 2 * overall
Rbar = R0 + g / (omega ** 2)

# Initial conditions
# theta_0 = np.random.rand() * 0.1 - 0.2
# r_0 = Rbar * (1 + np.random.rand() * 0.1 - 0.2)
eps_ratio = 0.5
theta_0 = 0.01 * eps_ratio
r_0 = Rbar * (1.0 + eps_ratio * 0.05)


def implicit_midpoint_integrator(f_func, energy_func, y_0, ts, solver=scipy.optimize.anderson):
    y = y_0
    y_history = np.zeros((len(ts), len(y)))
    y_history[0, :] = y

    energy_history = np.zeros((len(ts), 7))
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
        energy_history[i, :] = energy_func(y)

    return y, y_history, energy_history


def system_source(_, y):
    r, rdot, theta, thetadot = y[0], y[1], y[2], y[3]
    return np.array([
        rdot,
        g * np.cos(theta) + r * thetadot ** 2 - omega ** 2 * (r - R0),
        thetadot,
        -(2 * rdot * thetadot + g * np.sin(theta)) / r,
    ])


def energy(y):
    r, rdot, theta, thetadot = y[0], y[1], y[2], y[3]
    spring_P = 0.5 * omega ** 2 * (r - R0) ** 2 - g * (r-Rbar) * np.cos(theta)
    spring_K = 0.5 * rdot ** 2
    pendulum_P = -g * Rbar * np.cos(theta)
    pendulum_K = 0.5 * (r * thetadot) ** 2
    return [spring_P + spring_K + pendulum_P + pendulum_K, spring_P, spring_K, pendulum_P, pendulum_K,
            spring_P + spring_K, pendulum_P + pendulum_K]


ts = np.arange(0, 200, 0.2)
_, trajectory, energies = implicit_midpoint_integrator(system_source, energy, np.array([r_0, 0, theta_0, 0]), ts)

fig, axs = plt.subplots(6, 1, figsize=(9, 9))
labels = ['r', '$\\dot{r}$', '$\\theta$', '$\\dot{\\theta}$']
for j in range(4):
    axs[j].plot(ts, trajectory[:, j])
    axs[j].set(ylabel=labels[j])
axs[-2].plot(ts, energies[:, 1:5])
axs[-2].set(xlabel='t', ylabel='energy')
axs[-2].legend(['s_P', 's_K', 'p_P', 'p_K'])
axs[-1].plot(ts, energies[:, [0, 5, 6]])
axs[-1].set(xlabel='t', ylabel='energy')
axs[-1].legend(['total', 'spring', 'pendulum'])
plt.show()
