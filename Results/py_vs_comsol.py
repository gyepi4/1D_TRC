import numpy as np
import matplotlib.pyplot as plt

W_diss = np.array([10,      25,     50,     75,     100,    125,    150])
python = np.array([53.4,    58.5,   67,     75.5,   84,     92.5,   101])
COMSOL = np.array([53.052,  57.624, 65.183, 72.682, 80.125, 87.516, 94.863])

plt.figure(figsize=(10,5))
plt.plot(W_diss, python, 'o-', label=f'Analytical Model')
plt.plot(W_diss, COMSOL, 'o-', label=f'COMSOL Model')
plt.xlabel("W$_{diss}$ [W]")
plt.ylabel("T$_{junc}$ [°C]")
plt.legend()
plt.title("Simulation Results (Analytical vs Numerical)")
plt.show()