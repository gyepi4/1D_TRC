import numpy as np
import matplotlib.pyplot as plt

W_diss                      = np.array([10,      25,     50,     75,      100])

# Sim results with w_ch = 1mm and nr_ch = 1
python_Spread_1mm_lee       = np.array([54.916, 62.291, 74.582, 86.874, 99.165]) 
python_Spread_1mm_song      = np.array([55.156, 62.889, 75.778, 88.667, 101.556]) 
COMSOL_1mm                  = np.array([55.576, 63.928, 77.753, 91.487, 105.14])

# Sim results with w_ch = 3mm and nr_ch = 1
python_Spread_3mm_lee       = np.array([54.724,  61.809, 73.618, 85.427,  97.235]) 
python_Spread_3mm_song      = np.array([54.962,  62.405, 74.811, 87.216,  99.622]) 
COMSOL_3mm                  = np.array([55.014,  62.529, 74.992, 87.393,  99.738])

# Sim results with w_ch = 5mm and nr_ch = 1
python_Spread_5mm_lee       = np.array([54.561, 61.404, 72.807, 84.211, 95.615]) 
python_Spread_5mm_song      = np.array([54.800, 61.999, 73.998, 85.997, 97.996]) 
COMSOL_5mm                  = np.array([55.082, 62.698, 75.320, 87.873, 100.36])

# Sim results with w_ch = 3mm and nr_ch = 2
python_Spread_3mm_2ch           = np.array([53.711,  59.276, 68.553, 77.829,  87.105]) 
COMSOL_3mm_2ch                  = np.array([54.233,  60.584, 71.142, 81.675,  92.183])



plt.figure(figsize=(10,5))
plt.plot(W_diss, python_Spread_3mm_lee, 'o-', label=f'Analytical Model (Lee et al.)')
plt.plot(W_diss, python_Spread_3mm_song, 'o-', label=f'Analytical Model (Song et al.)')
plt.plot(W_diss, COMSOL_5mm, 'o-', label=f'Numerical Model')
plt.xlabel("W$_{diss}$ [W]")
plt.ylabel("T$_{junc}$ [°C]")
plt.legend()
plt.title("Simulation Results (Analytical vs Numerical)")
plt.show()