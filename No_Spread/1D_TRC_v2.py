import numpy as np
import matplotlib.pyplot as plt

"""
Defining the properties of the different layers

Sources 
-------


parameters
----------
R : Thermal resistance in K/W
k : Thermal conductivity in W/mK
l : Length in m
w : Width in m
t : Thickness in m
m_dot: mass flow in kg/s
rho : Density in Kg/m3
mu : Viscosity in Pas
cp : Specific heat
"""

# == Die ==
R_die   = 0.05
W_diss  = 100
l_die   = 30e-3 #30e-3
w_die   = 20e-3 #20e-3

# == TIM ==
t_TIM   = 0.1e-3
k_TIM   = 3

# == PCB ==
t_pcb   = 2e-3
k_pcb   = 160

# == Heatsink ==
t_HS    = 5e-3
k_HS    = 160

l_ch    = l_die
h_ch    = 10e-3
w_ch    = 3e-3

# == Coolant ==
T_cl        = 50
glycol_pct  = 0

# Glycol properties
rho_g   = 1060.0
mu_g    = 1.4e-3
cp_g    = 3500.0
k_g     = 0.253

# water properties
rho_w   = 988.0 
mu_w    = 5.5e-4
cp_w    = 4181.0
k_w     = 0.644

# Coolant mixture
x           = glycol_pct / 100.0

rho_f   = (1 - x) * rho_w + x * rho_g
mu_f    = (1 - x) * mu_w  + x * mu_g
cp_f    = (1 - x) * cp_w  + x * cp_g
k_f     = (1 - x) * k_w   + x * k_g

u       = 2
m_dot   = u * (rho_f * (w_ch * h_ch))

"""
Deriving the thermal resistance
"""

A_die = l_die * w_die

R_TIM   = t_TIM / (k_TIM * A_die)
R_pcb   = t_pcb / (k_pcb * A_die)
R_HS    = t_HS  / (k_HS  * A_die)

"""
Deriving the heat transfer coefficient of the coolant and R_conv

Sources 
-------
https://www.nuclear-power.com/nuclear-engineering/fluid-dynamics/internal-flow/hydraulic-diameter-2/
https://en.wikipedia.org/wiki/Nusselt_number
https://www.nuclear-power.com/nuclear-engineering/heat-transfer/convection-convective-heat-transfer/laminar-vs-turbulent-nusselt-number/
https://www.comsol.com/blogs/calculating-the-heat-transfer-coefficient-for-flat-and-corrugated-plates

parameters
----------
A_ch : cross-sectional area in m2
P_ch : wetted perimeter in m
D_h : Hydraulic diameter
u : flow speed in m/s
Re : Reynolds number
Pr: Prandtl number
Nu : Nusselt number
h : heat transfer coefficient in W/m2K
"""

Re  = rho_f * u * l_ch / mu_f
Pr   = (cp_f * mu_f) / k_f

if Re < 5e5:
    Nu = ((0.3387*Pr**(1/3)*Re**(1/2))) / ((1+ (0.0468/Pr)**(2/3))**(1/4))
    regime = "laminar"
elif Re > 5e5:
    Nu     = Pr**(1/3) * (0.037*Re**(4/5) - 871)    
    regime = "turbulent"

h_cl    = 2 * Nu * k_f / l_ch
A_conv  = 2 * (h_ch * l_ch) + w_ch * l_ch  
R_conv  = 1 / (h_cl * A_conv)

"""
Deriving the total thermal resistance and the junction temperature

parameters
----------
T_junc : Junction temperature in °C
"""

R_tot   = R_TIM + R_pcb + R_HS + R_conv
T_junc  = T_cl + (W_diss * R_tot)

"""
Deriving the pressure drop

Sources 
-------
https://www.nuclear-power.com/nuclear-engineering/fluid-dynamics/major-head-loss-friction-loss/darcy-weisbach-equation/
https://nl.wikipedia.org/wiki/Darcy-Weisbach-vergelijking

parameters
----------
f_D : Darcy friction factor
dP : pressure drop in Pa
"""

A_ch = w_ch * h_ch
P_ch = 2 * (w_ch + h_ch)
D_h  = 4 * A_ch / P_ch

if Re < 2300:
    f_D = 64 / Re
else:
    f_D = (0.790 * np.log(Re) - 1.64)**(-2)

dP     = f_D * (l_ch / D_h) * (0.5 * rho_f * u**2)
dP_bar = dP / 1e5

"""
Print statements
"""

print(f"{'Layer':<30} {'t [mm]':>8}  {'k [W/mK]':>9}  {'A [cm²]':>8}  {'R [K/W]':>9}  {'share':>6}")
print("-" * 80)
for name, t, k, A, R in [
    ("PCB",         t_pcb,  k_pcb,   A_die,  R_pcb   ),
    ("TIM",         t_TIM,  k_TIM,   A_die,  R_TIM   ),
    ("Heatsink",    t_HS,   k_HS,    A_die,  R_HS    ),
    ("Convection",  None,   None,    A_conv, R_conv  ),
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
print(f"Re                  =   {Re:.4f} [-]")
print(f"Pr                  =   {Pr:.4f} [-]")
print(f"Nu                  =   {Nu:.4f} [-]")
print(f"f_D                 =   {f_D:.6f} [-]")
print()
print(f"System behavior:")
print("-" * 80)
print(f"T_cl                =   {T_cl:.3f} [°C]")
print(f"Glycol percentage   =   {glycol_pct:.1f} [%]")
print(f"Temperature rise    =   {W_diss * R_tot:.1f} [°C]")
print(f"T_junc              =   {T_junc:.3f} [°C]")
print(f"u                   =   {u:.4f} [m/s]")
print(f"mass flow           =   {m_dot:.4f} [kg/s]")
print(f"dP                  =   {dP_bar:.5f} [bar]")
print(f"Regime              =   {regime}")

