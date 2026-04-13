import numpy as np

"""
Thermal resistance circuit for liquid-cooled chip stack.

Stack (top to bottom):
  Die → TIM → PCB → Heatsink base (spreading) → Coolant (convection)

parameters
----------
R       : Thermal resistance [K/W]
k       : Thermal conductivity [W/mK]
l       : Length [m]
w       : Width [m]
t       : Thickness [m]
m_dot   : Mass flow [kg/s]
rho     : Density [kg/m3]
mu      : Dynamic viscosity [Pa*s]
cp      : Specific heat [J/kgK]
"""

# == Heatsink ==    
t_HS    = 6e-3
k_HS    = 160
l_HS    = 30e-3         
w_HS    = 20e-3  

# == PCB ==
t_pcb   = 2e-3
k_pcb   = 160

# == TIM ==
t_TIM   = 0#0.1e-3
k_TIM   = 3

# == Die ==
W_diss  = 10
l_die   = 10e-3
w_die   = 10e-3

# == Channel geometry ==
l_ch    = l_HS          
h_ch    = 10e-3
w_ch    = 1e-3

# == Coolant ==
T_cl        = 50
glycol_pct  = 0

# Water properties
rho_w   = 988.02
mu_w    = 0.000547
cp_w    = 4130.87
k_w     = 0.641

# Glycol properties
rho_g   = 1050.440
mu_g    = 1.538e-3
cp_g    = 3499.0
k_g     = 0.4108

# Coolant mixture
x       = glycol_pct / 100.0
rho_f   = (1 - x) * rho_w + x * rho_g
mu_f    = (1 - x) * mu_w  + x * mu_g
cp_f    = (1 - x) * cp_w  + x * cp_g
k_f     = (1 - x) * k_w   + x * k_g

u       = 2
m_dot   = u * rho_f * (w_ch * h_ch)

# =============================================================================
# Geometry and simple 1D resistances
# =============================================================================

A_die   = l_die * w_die         
A_HS    = l_HS  * w_HS      

# TIM and PCB: heat not yet spread, so area = A_die
R_TIM   = t_TIM / (k_TIM * A_die)
R_pcb   = t_pcb / (k_pcb * A_die)

# =============================================================================
# Convective resistance
# =============================================================================

A_ch    = w_ch * h_ch
P_ch    = w_ch + 2*h_ch
D_h     = (4 * A_ch) / P_ch

Re      = rho_f * u * l_ch / mu_f
Pr      = (cp_f * mu_f) / k_f

if Re < 5e5:
    Nu      = 0.3387 * Pr**(1/3) * Re**(1/2) / (1 + (0.0468/Pr)**(2/3))**(1/4)
    regime  = "laminar"
else:
    Nu      = Pr**(1/3) * (0.037 * Re**(4/5) - 871)
    regime  = "turbulent"

h_cl    = (2 * Nu * k_f) / l_ch
A_conv  = 2 * (h_ch * l_ch) + w_ch * l_ch    
R_conv  = 1 / (h_cl * A_conv)

# =============================================================================
# Spreading + conduction resistance (Lee et al. 1995, Song et al. 1994)
#
# NOTE: R_spread already includes the 1D bulk conduction through t_HS
#       via the eps*tau/sqrt(pi) term in Psi. Do NOT add R_m,HS separately.
# =============================================================================

A_s     = A_die    
A_p     = l_HS * w_HS    

a       = np.sqrt(A_s / np.pi)     
b       = np.sqrt(A_p / np.pi)     

eps     = a / b                               
tau     = t_HS / b                
Bi      = 1 / (np.pi * k_HS * b * R_conv)

lambda_c = np.pi + (1.0 / (np.sqrt(np.pi) * eps))
Phi_c    = (np.tanh(lambda_c * tau) + (lambda_c / Bi)) / \
           (1 + (lambda_c / Bi) * np.tanh(lambda_c * tau))

Psi_avg  = (eps * tau) / np.sqrt(np.pi) + \
            0.5 * (1 - eps)**(3/2) * Phi_c
Psi_max  = (eps * tau) / np.sqrt(np.pi) + \
            (1/np.sqrt(np.pi)) * (1 - eps) * Phi_c

R_spread = Psi_max / (np.sqrt(np.pi) * k_HS * a)

# =============================================================================
# Total thermal resistance and junction temperature
# =============================================================================

# Series circuit: die → TIM → PCB → spreading/conduction in HS → convection
R_tot   = R_TIM + R_pcb + R_spread + R_conv
T_junc  = T_cl + W_diss * R_tot

# =============================================================================
# Pressure drop (Darcy-Weisbach)
# =============================================================================

if Re < 2300:
    f_D = 64 / Re
else:
    f_D = (0.790 * np.log(Re) - 1.64)**(-2)

dP      = f_D * (l_ch / D_h) * (0.5 * rho_f * u**2)
dP_bar  = dP * 1e-5

# =============================================================================
# Print results
# =============================================================================

print(f"{'Layer':<30} {'t [mm]':>8}  {'k [W/mK]':>9}  {'A [cm²]':>8}  {'R [K/W]':>9}  {'share':>6}")
print("-" * 80)
for name, t, k, A, R in [
    ("TIM",        t_TIM,  k_TIM,  A_die,   R_TIM),
    ("PCB",        t_pcb,  k_pcb,  A_die,   R_pcb),
    ("Spreading",  None,   None,   A_die,   R_spread),
    ("Convection", None,   None,   A_conv,  R_conv),
]:
    t_str = f"{t*1e3:>7.2f}" if t is not None else f"{'—':>7}"
    k_str = f"{k:>8.1f}"     if k is not None else f"{'—':>8}"
    A_str = f"{A*1e4:>7.2f}" if A is not None else f"{'—':>7}"
    print(f"  {name:<28} {t_str}   {k_str}   {A_str}   {R:>9.5f}  {100*R/R_tot:>5.1f}%")
print("-" * 80)
print(f"  {'TOTAL':<57}   {R_tot:>9.5f}  100.0%")

print()
print(f"Dimensionless numbers:")
print("-" * 80)
print(f"  Re                   =   {Re:.4f} [-]")
print(f"  Nu                   =   {Nu:.4f} [-]")
print(f"  f_D                  =   {f_D:.6f} [-]")

print()
print(f"System behavior:")
print("-" * 80)
print(f"  T_cl                 =   {T_cl:.3f} [°C]")
print(f"  Glycol percentage    =   {glycol_pct:.1f} [%]")
print(f"  Temperature rise     =   {W_diss * R_tot:.3f} [°C]")
print(f"  T_junc               =   {T_junc:.3f} [°C]")
print(f"  u                    =   {u:.4f} [m/s]")
print(f"  m_dot                =   {m_dot:.5f} [kg/s]")
print(f"  dP                   =   {dP_bar:.5f} [bar]")
print(f"  Regime               =   {regime}")