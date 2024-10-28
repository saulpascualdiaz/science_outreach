import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

def lorenz(t, xyz):
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Time range
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Initial conditions
initial_conditions = [1.0, 1.0, 1.0]
initial_conditions_perturbed = [1.0, 1.0, 1.0]  # Slightly different initial conditions

# Solve the system
solution1 = solve_ivp(lorenz, t_span, initial_conditions, t_eval=t_eval)
solution2 = solve_ivp(lorenz, t_span, initial_conditions_perturbed, t_eval=t_eval)

# Plot the results with transparency
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution1.y[0], solution1.y[1], solution1.y[2], color='blue', alpha=0.6, lw=1.5, label="Initial Condition 1")
ax.plot(solution2.y[0], solution2.y[1], solution2.y[2], color='red', alpha=0.6, lw=1.5, label="Initial Condition 2")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("Lorenz Attractor - Sensitivity to Initial Conditions")
plt.legend()
plt.show()
