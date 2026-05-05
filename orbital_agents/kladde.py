import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 50, 500)
rp = 75
rs = 81
omega = 0.25
theta_p = np.pi / 4
theta_s = np.pi / 3
v = 4
xp = rp * np.cos(omega * t + theta_p)
yp = rp * np.sin(omega * t + theta_p)

plt.figure(figsize=(6, 6))
plt.plot(t, xp)
plt.plot(t, yp, label='Planet X')
for theta_i in np.linspace(0.01, np.pi / 2, 20):
    xs = rs * np.cos(theta_s) + v * t * np.cos(theta_i) * t
    plt.plot(t, xs, label=f'theta_i={theta_i:.1f}')
# xs = rs * np.cos(theta_s) + v * t * np.cos(theta_i) * t
plt.xlim(0, 2)
plt.ylim(35, 50)
plt.legend()
# plt.show()

plt.figure(figsize=(6, 6))
plt.plot(t, yp, label='Planet Y')
for theta_i in np.linspace(0.01, np.pi / 2, 20):
    ys = rs * np.sin(theta_s) + v * t * np.sin(theta_i) * t
    plt.plot(t, ys, label=f'theta_i={theta_i:.1f}')
# ys = rs * np.cos(theta_s) + v * t * np.cos(theta_i) * t
plt.xlim(0, 5)
plt.ylim(35, 100)
plt.legend()
plt.show()

from scipy.optimize import brentq


def intercept(rs, theta_s, rp, theta_p, omega, v, t_max=100):
    def f(t):
        return (v * t) ** 2 - (rp**2 + rs**2 - 2 * rp * rs * np.cos(omega * t + theta_p - theta_s))

    # Find a bracket where f changes sign
    t_grid = np.linspace(1e-6, t_max, 10000)
    signs = np.sign(f(t_grid))
    idx = np.where(np.diff(signs))[0]
    if len(idx) == 0:
        return None, None  # no solution found

    # Take the first (earliest) intercept
    t_sol = brentq(f, t_grid[idx[0]], t_grid[idx[0] + 1])

    # Recover theta_i from the original equations
    dx = rp * np.cos(omega * t_sol + theta_p) - rs * np.cos(theta_s)
    dy = rp * np.sin(omega * t_sol + theta_p) - rs * np.sin(theta_s)
    theta_i = np.arctan2(dy, dx)

    return t_sol, theta_i


ic = intercept(rs, theta_s, rp, theta_p, omega, v, t_max=50)
print(f'Intercept time: {ic[0]:.2f} s, Intercept angle: {ic[1] / np.pi:.2f}π')
