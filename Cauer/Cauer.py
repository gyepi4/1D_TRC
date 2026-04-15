import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
Cauer thermal ladder model (transient)

Each physical layer is one RC element:
  - R_i  : thermal resistance of layer i [K/W]
  - C_i  : thermal capacitance of layer i [J/K]

Network topology (Cauer):
  Q_diss -> [R_TIM, C_TIM] -> [R_PCB, C_PCB] -> [R_HS, C_HS] -> [R_conv, C_conv=0] -> T_ambient

Nodes (temperatures):
  T[0] = T_junction  (top of TIM)
  T[1] = T_TIM/PCB interface
  T[2] = T_PCB/HS interface
  T[3] = T_HS/coolant interface
  T[4] = T_ambient   (fixed boundary)

Each node i obeys:
  C_i * dT[i]/dt = Q_in[i] - Q_out[i]

where Q flows left-to-right through R_i, and C_i drains to ambient (ground).
"""

# =============================================================================
# Material and geometry (reuse from your existing model)
# =============================================================================

# Heatsink
t_HS  = 6e-3;   k_HS  = 160;  rho_HS  = 2700;  cp_HS  = 900    # aluminium
l_HS  = 30e-3;  w_HS  = 20e-3

# PCB
t_pcb = 2e-3;   k_pcb = 160;  rho_pcb = 2700;  cp_pcb = 900

# TIM
t_TIM = 0.1e-3; k_TIM = 3;    rho_TIM = 3300;  cp_TIM = 800    # typical TIM

# Die
W_diss = 10     # [W]
l_die  = 10e-3; w_die = 10e-3

# Coolant and convection (from your existing model — paste your R_conv here)
T_amb  = 50     # [°C]  coolant inlet temperature

A_die = l_die * w_die
A_HS  = l_HS  * w_HS

# --- steady-state resistances (1D for Cauer; spreading handled separately) ---
# For the Cauer model we use the 1D resistances per layer.
# The spreading effect is a steady-state correction you already have.
R_TIM_1D  = t_TIM / (k_TIM  * A_die)
R_PCB_1D  = t_pcb / (k_pcb  * A_die)
R_HS_1D   = t_HS  / (k_HS   * A_HS)

# Paste your computed R_conv value here, or recompute it:
# (simplified placeholder — replace with your actual value)
R_conv = 0.05   # [K/W]  <-- replace with your computed R_conv

# =============================================================================
# Thermal capacitances  C = rho * cp * Volume
# =============================================================================

C_TIM = rho_TIM * cp_TIM * (t_TIM * A_die)
C_PCB = rho_pcb * cp_pcb * (t_pcb * A_die)
C_HS  = rho_HS  * cp_HS  * (t_HS  * A_HS)
# Coolant thermal mass is large and fast-mixing; treat as fixed T boundary
# so C_conv = inf → node 3 is clamped. We handle this by fixing T[3] = T_amb
# after solving, or by giving it a large C. Here we fix the boundary directly.

print(f"Steady-state resistances:")
print(f"  R_TIM  = {R_TIM_1D:.5f} K/W")
print(f"  R_PCB  = {R_PCB_1D:.5f} K/W")
print(f"  R_HS   = {R_HS_1D:.5f}  K/W")
print(f"  R_conv = {R_conv:.5f}   K/W")
print(f"  R_tot  = {R_TIM_1D+R_PCB_1D+R_HS_1D+R_conv:.5f} K/W")
print()
print(f"Thermal capacitances:")
print(f"  C_TIM = {C_TIM:.6f} J/K")
print(f"  C_PCB = {C_PCB:.6f} J/K")
print(f"  C_HS  = {C_HS:.4f}  J/K")

# =============================================================================
# Cauer ODE system
#
# State vector: T = [T0, T1, T2, T3]
#   T0 = junction (top of TIM)
#   T1 = TIM/PCB interface
#   T2 = PCB/HS interface
#   T3 = HS/coolant interface
#
# Fixed boundary: T4 = T_amb (coolant)
#
# Heat flows:
#   Q01 = (T0 - T1) / R_TIM   (through TIM resistance)
#   Q12 = (T1 - T2) / R_PCB   (through PCB resistance)
#   Q23 = (T2 - T3) / R_HS    (through HS resistance)
#   Q34 = (T3 - T4) / R_conv  (through convective resistance)
#
# KCL at each node:
#   C_TIM * dT0/dt = W_diss - Q01          (power injected at junction)
#   C_PCB * dT1/dt = Q01    - Q12
#   C_HS  * dT2/dt = Q12    - Q23
#   0     * dT3/dt = Q23    - Q34  → quasi-static: T3 solved algebraically
#
# Node 3 has no capacitance in this discretisation (it is the HS surface,
# whose thermal mass is already in C_HS). Treat T3 as algebraic:
#   T3 = T_amb + Q23 * R_conv  (instantaneous, since C=0 there)
# which couples back through Q23. Solve implicitly by keeping T3 as a state
# with a very small C_boundary to avoid index-1 DAE:
# =============================================================================

C_boundary = 1e-6   # [J/K] — numerically small, makes T3 respond fast

R = [R_TIM_1D, R_PCB_1D, R_HS_1D, R_conv]
C = [C_TIM,    C_PCB,    C_HS,    C_boundary]

def cauer_odes(t, T):
    T4 = T_amb   # fixed coolant temperature
    Q = [
        (T[0] - T[1]) / R[0],   # Q through TIM
        (T[1] - T[2]) / R[1],   # Q through PCB
        (T[2] - T[3]) / R[2],   # Q through HS
        (T[3] - T4  ) / R[3],   # Q through convection
    ]
    dTdt = [
        (W_diss - Q[0]) / C[0],  # node 0: junction
        (Q[0]   - Q[1]) / C[1],  # node 1: TIM/PCB
        (Q[1]   - Q[2]) / C[2],  # node 2: PCB/HS
        (Q[2]   - Q[3]) / C[3],  # node 3: HS surface
    ]
    return dTdt

# =============================================================================
# Solve: step power on at t=0, run until steady state
# =============================================================================

T0_init = [T_amb, T_amb, T_amb, T_amb]   # start at coolant temperature

# Time span: run long enough to reach steady state
# Thermal time constant ~ max(R_i * C_i); use 10x that
tau_max = max(r * c for r, c in zip(R, C))
t_end   = 50 * tau_max
t_span  = (0, t_end)
t_eval  = np.logspace(-4, np.log10(t_end), 500)

sol = solve_ivp(cauer_odes, t_span, T0_init, t_eval=t_eval,
                method='Radau', rtol=1e-8, atol=1e-10)

T_junc_transient = sol.y[0]
T_tim_pcb        = sol.y[1]
T_pcb_hs         = sol.y[2]
T_hs_surf        = sol.y[3]

# Steady-state values
T_ss = sol.y[:, -1]
R_tot_1D = sum(R)
T_junc_ss_expected = T_amb + W_diss * R_tot_1D

print()
print(f"Steady-state check:")
print(f"  T_junc  (Cauer, t→∞)  = {T_ss[0]:.3f} °C")
print(f"  T_junc  (R_tot * Q)   = {T_junc_ss_expected:.3f} °C")
print(f"  T_TIM/PCB             = {T_ss[1]:.3f} °C")
print(f"  T_PCB/HS              = {T_ss[2]:.3f} °C")
print(f"  T_HS surface          = {T_ss[3]:.3f} °C")

# =============================================================================
# Thermal impedance Zth(t) = (T_junc(t) - T_amb) / W_diss
# =============================================================================

Zth = (T_junc_transient - T_amb) / W_diss

# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# -- left: temperature profiles over time --
ax = axes[0]
ax.semilogx(sol.t, T_junc_transient, label='T_junction')
ax.semilogx(sol.t, T_tim_pcb,        label='T_TIM/PCB interface')
ax.semilogx(sol.t, T_pcb_hs,         label='T_PCB/HS interface')
ax.semilogx(sol.t, T_hs_surf,        label='T_HS surface')
ax.axhline(T_amb, color='k', linestyle='--', linewidth=0.8, label='T_ambient')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Temperature [°C]')
ax.set_title('Cauer model — transient temperature response')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# -- right: Zth(t) --
ax = axes[1]
ax.semilogx(sol.t, Zth)
ax.axhline(R_tot_1D, color='r', linestyle='--', linewidth=0.8,
           label=f'Steady-state R_tot = {R_tot_1D:.4f} K/W')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Z_th [K/W]')
ax.set_title('Thermal impedance Z_th(t)')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('cauer_thermal_model.png', dpi=150)
plt.show()