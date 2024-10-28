import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Gravitational constant (arbitrary units)
G = 1

# Define masses of the three bodies
m1, m2, m3 = 1.0, 1.0, 1.0  # masses of the three bodies

# Updated initial conditions with closer positions for more interaction
# Positions (x, y) and velocities (vx, vy) for each body
initial_state = [
    -0.6, 0.0,   # Body 1 position
     0.5, 0.0,   # Body 2 position
     0.0, 0.5,   # Body 3 position
    0.0, 1.2,    # Body 1 velocity
   -1.0, -1.2,   # Body 2 velocity
    1.0, 0.0     # Body 3 velocity
]

# Define the differential equations for the three-body system
def three_body(t, state):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state

    # Distances between bodies
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    # Accelerations due to gravity
    ax1 = G * m2 * (x2 - x1) / r12**3 + G * m3 * (x3 - x1) / r13**3
    ay1 = G * m2 * (y2 - y1) / r12**3 + G * m3 * (y3 - y1) / r13**3
    ax2 = G * m1 * (x1 - x2) / r12**3 + G * m3 * (x3 - x2) / r23**3
    ay2 = G * m1 * (y1 - y2) / r12**3 + G * m3 * (y3 - y2) / r23**3
    ax3 = G * m1 * (x1 - x3) / r13**3 + G * m2 * (x2 - x3) / r23**3
    ay3 = G * m1 * (y1 - y3) / r13**3 + G * m2 * (y2 - y3) / r23**3

    # Return derivatives
    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

# Time range for simulation
t_span = (0, 20)
t_eval = np.linspace(*t_span, 5000)

# Solve the system
solution = solve_ivp(three_body, t_span, initial_state, t_eval=t_eval)

# Extract positions for plotting
x1, y1 = solution.y[0], solution.y[1]
x2, y2 = solution.y[2], solution.y[3]
x3, y3 = solution.y[4], solution.y[5]

# Plot the results
plt.figure(figsize=(10, 8))
plt.plot(x1, y1, label='Body 1', color='blue')
plt.plot(x2, y2, label='Body 2', color='red')
plt.plot(x3, y3, label='Body 3', color='green')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Three-Body Problem - Chaotic Trajectories with Closer Initial Conditions")
plt.legend()
plt.grid()
plt.show()
