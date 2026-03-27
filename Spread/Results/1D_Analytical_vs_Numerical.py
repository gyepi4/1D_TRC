import numpy as np
import matplotlib.pyplot as plt

W_diss          = np.array([10,      25,     50,     75,     100,    125,    150])
python_NoSpread = np.array([52.847,  57.116, 64.233, 71.349, 78.465, 85.582, 92.698])
python_Spread   = np.array([52.870,  57.176, 64.352, 71.529, 78.705, 85.881, 93.057])
COMSOL          = np.array([53.052,  57.624, 65.183, 72.682, 80.125, 87.516, 94.863])

plt.figure(figsize=(10,5))
plt.plot(W_diss, python_NoSpread, 'o-', label=f'Analytical Model (No Spread)')
plt.plot(W_diss, python_Spread, 'o-', label=f'Analytical Model (Spread)')
plt.plot(W_diss, COMSOL, 'o-', label=f'Numerical Model')
plt.xlabel("W$_{diss}$ [W]")
plt.ylabel("T$_{junc}$ [°C]")
plt.legend()
plt.title("Simulation Results (Analytical vs Numerical)")
plt.show()