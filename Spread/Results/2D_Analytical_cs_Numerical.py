import numpy as np
import matplotlib.pyplot as plt

W_diss          = np.array([10,      25,     50,     75,      100])
python_NoSpread_under = np.array([52.847,  57.116, 64.233, 71.349,  78.465]) # cooling channel spans = heatsink size
python_NoSpread_over = np.array([58.487,  71.217, 92.435, 113.652, 134.870])  # cooling channel spans = die size
python_Spread   = np.array([55.495,  63.739, 77.477, 91.216,  104.954])
COMSOL          = np.array([55.014,  62.529, 74.992, 87.393,  99.738])

plt.figure(figsize=(10,5))
plt.plot(W_diss, python_NoSpread_over, 'o-', label=f'Analytical Model (No Spread with overpreduction)')
plt.plot(W_diss, python_NoSpread_under, 'o-', label=f'Analytical Model (No Spread with underprediction)')
plt.plot(W_diss, python_Spread, 'o-', label=f'Analytical Model (Spread)')
plt.plot(W_diss, COMSOL, 'o-', label=f'Numerical Model')
plt.xlabel("W$_{diss}$ [W]")
plt.ylabel("T$_{junc}$ [°C]")
plt.legend()
plt.title("Simulation Results (Analytical vs Numerical)")
plt.show()