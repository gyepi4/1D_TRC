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

# == Die (IMCQ120R017M2HXTMA1) ==
# Source: https://plm.prodrive.nl/Component/2210-0000-0338?tab=d
W_diss  = 25
l_die   = 15.50e-3
w_die   = 12e-3

# # == TIM ==
# # Source: https://plm.prodrive.nl/Component/5115-0000-0130?tab=d
# t_TIM   = 1.3e-3
# k_TIM   = 5.4

# == Heatsink (EN-AW-6060 T66) ==    
# Source: Roloff / Matek
t_HS    = 3e-3
k_HS    = 160
l_HS    = 30e-3         
w_HS    = 20e-3  

# == Channel geometry ==
l_ch    = l_HS          
h_ch    = 10e-3
w_ch    = 1e-3

# == Coolant ==
T_cl        = 45
glycol_pct  = 40

# Water properties
# Source: www.thermalexcel.com/english/tables/eau_atm.html 
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

u       = 3
m_dot   = u * rho_f * (w_ch * h_ch)

# =============================================================================
# Geometry and simple 1D resistances
# =============================================================================

A_die   = l_die * w_die         
A_HS    = l_HS  * w_HS      

# 45-degree spreading within TIM
# Effective area grows by t_TIM on each side in both directions
# l_TIM_bot = l_die + 2 * t_TIM     # bottom face of TIM (die side = top, HS side = bottom)
# w_TIM_bot = w_die + 2 * t_TIM

# A_TIM_top = l_die    * w_die        # = A_die, heat source footprint
# A_TIM_bot = l_TIM_bot * w_TIM_bot   # expanded footprint at HS interface

# # Geometric mean area for frustum-shaped heat flow
# A_TIM_mean = np.sqrt(A_TIM_top * A_TIM_bot)

# R_TIM = t_TIM / (k_TIM * A_TIM_bot)

# R_TIM   = t_TIM / (k_TIM * A_die)

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

Psi_avg  = (eps * tau) / np.sqrt(np.pi) + 0.5 * (1 - eps)**(3/2) * Phi_c
Psi_max  = (eps * tau) / np.sqrt(np.pi) + (1/np.sqrt(np.pi)) * (1 - eps) * Phi_c

R_spread = Psi_max / (np.sqrt(np.pi) * k_HS * a)

# =============================================================================
# Total thermal resistance and junction temperature
# =============================================================================

R_tot   = R_spread + R_conv
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
    # ("TIM",        t_TIM,  k_TIM,  A_die,   R_TIM),
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