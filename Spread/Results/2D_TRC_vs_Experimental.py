import numpy as np
import matplotlib.pyplot as plt

velocity        = np.array([1, 3, 5])

# Sim results with a = 0.014
Analytical_avg      = np.array([1.07, 0.77, 0.64])
Analytical_max      = np.array([1.15, 0.85, 0.72])
song_avg            = np.array([0.98, 0.69, 0.56]) 
song_max            = np.array([1.04, 0.74, 0.62]) 
experimental_avg    = np.array([1.07, 0.75, 0.63])
experimental_max    = np.array([1.12, 0.80, 0.68])


plt.figure(figsize=(10,5))
# Analytical (gijs)
plt.plot(velocity, Analytical_max, '^-', label='Analytical Model ($\Psi_{max})$')
plt.plot(velocity, Analytical_avg, 'o-', label='Analytical Model ($\Psi_{avg}$)')
# Analytical (song)
# plt.plot(velocity, song_max, 'o-', label='Analytical Model ($\Psi_{max})$')
# plt.plot(velocity, song_avg, 'o-', label='Analytical Model ($\Psi_{avg}$)')
# Experiments
plt.plot(velocity, experimental_max, '^--', label='Experimental data (max)')
plt.plot(velocity, experimental_avg, 'o--', label='Experimental data (avg)')
plt.ylabel("R$_{tot}$ [C/W]")
plt.xlabel("v [m/s]")
plt.legend()
plt.title("Comparison between experimental and analytical data")
plt.show()