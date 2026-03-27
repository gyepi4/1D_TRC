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
W_diss  = 25
l_die   = 30e-3
w_die   = 20e-3

# == PCB ==
t_pcb   = 2e-3
k_pcb   = 160

# == TIM ==
t_TIM   = 0.1e-3
k_TIM   = 3

# == Heatsink ==
t_HS    = 6e-3
k_HS    = 160

l_ch    = l_die
h_ch    = 10e-3
w_ch    = 3e-3

# == Coolant ==
T_cl        = 50
glycol_pct  = 0
m_dot       = 0.08

# water properties
rho_w   = 988.0 
mu_w    = 5.5e-4
cp_w    = 4181.0
k_w     = 0.644

# Glycol properties
rho_g   = 1060.0
mu_g    = 1.4e-3
cp_g    = 3500.0 
k_g     = 0.253

# Coolant mixture
x           = glycol_pct / 100.0

rho_f   = (1 - x) * rho_w + x * rho_g
mu_f    = (1 - x) * mu_w  + x * mu_g
cp_f    = (1 - x) * cp_w  + x * cp_g
k_f     = (1 - x) * k_w   + x * k_g

"""
Deriving the thermal resistance
"""

A_die = l_die * w_die

R_pcb   = t_pcb / (k_pcb * A_die)
R_TIM   = t_TIM / (k_TIM * A_die)
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

A_ch = w_ch * h_ch
P_ch = 2 * (w_ch + h_ch)
D_h  = 4 * A_ch / P_ch

u    = m_dot / (rho_f * A_ch)
Re   = rho_f * u * D_h / mu_f
Pr   = (cp_f * mu_f) / k_f

if Re < 5e5:
    Nu     = 0.664 * Re**(1/2) * Pr**(1/3)
    regime = "laminar"
elif Re > 5e5:
    Nu     = 0.0296 * Re**(4/5) * Pr**(1/3)      
    regime = "turbulent"

h_cl = Nu * k_f / D_h
R_conv    = 1 / (h_cl * A_die)

"""
Deriving the total thermal resistance and the junction temperature

parameters
----------
T_junc : Junction temperature in °C
"""

R_tot   = R_die + R_pcb + R_TIM + R_HS + R_conv
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
    ("Die",         None,   None,   None,   R_die   ),
    ("PCB",         t_pcb,  k_pcb,  A_die,  R_pcb   ),
    ("TIM",         t_TIM,  k_TIM,  A_die,  R_TIM   ),
    ("Heatsink",    t_HS,   k_HS,   A_die,  R_HS    ),
    ("Convection",  None,   None,   None,   R_conv  ),
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
print(f"T_cl                =   {T_cl:.1f} [°C]")
print(f"Glycol percentage   =   {glycol_pct:.1f} [%]")
print(f"Temperature rise    =   {W_diss * R_tot:.1f} [°C]")
print(f"T_junc              =   {T_junc:.1f} [°C]")
print(f"u                   =   {u:.4f} [m/s]")
print(f"mass flow           =   {m_dot:.4f} [kg/s]")
print(f"dP                  =   {dP_bar:.5f} [bar]")
print(f"Regime              =   {regime}")

"""
Function that should the influence parameters can have on the junction temperature

Returns
----------
plt: Reynolds number vs. Glycol pct
plt: Junction temperature vs. Glycol pct
plt: Reynolds number vs. Channel width
plt: Junction temperature vs. Channel width
plt: Mass flow vs Junction temperature
"""

def compute_Tj_glycol(glycol_pct):
    x     = glycol_pct / 100.0
    rho_f = (1 - x) * rho_w + x * rho_g
    mu_f  = (1 - x) * mu_w  + x * mu_g
    cp_f  = (1 - x) * cp_w  + x * cp_g
    k_f   = (1 - x) * k_w   + x * k_g

    A_ch  = w_ch * h_ch
    P_ch  = 2 * (w_ch + h_ch)
    D_h   = 4 * A_ch / P_ch

    u    = m_dot / (rho_f * A_ch)
    Re   = rho_f * u * D_h / mu_f
    Pr   = (cp_f * mu_f) / k_f

    if Re < 5e5:
        Nu     = 0.664 * Re**(1/2) * Pr**(1/3)
        regime = "laminar"
    elif Re > 5e5:
        Nu     = 0.0296 * Re**(4/5) * Pr**(1/3)      
        regime = "turbulent"

    h_coolant = Nu * k_f / D_h
    R_conv    = 1 / (h_coolant * A_die)
    R_tot     = R_die + R_pcb + R_TIM + R_HS + R_conv
    return T_cl + W_diss * R_tot, R_conv, Re

def compute_Tj_w(w_ch):
    x     = glycol_pct / 100.0
    rho_f = (1 - x) * rho_w + x * rho_g
    mu_f  = (1 - x) * mu_w  + x * mu_g
    cp_f  = (1 - x) * cp_w  + x * cp_g
    k_f   = (1 - x) * k_w   + x * k_g

    A_ch  = w_ch * h_ch
    P_ch  = 2 * (w_ch + h_ch)
    D_h   = 4 * A_ch / P_ch

    u    = m_dot / (rho_f * A_ch)
    Re   = rho_f * u * D_h / mu_f
    Pr   = (cp_f * mu_f) / k_f

    if Re < 5e5:
        Nu     = 0.664 * Re**(1/2) * Pr**(1/3)
        regime = "laminar"
    elif Re > 5e5:
        Nu     = 0.0296 * Re**(4/5) * Pr**(1/3)      
        regime = "turbulent"

    h_coolant = Nu * k_f / D_h
    R_conv    = 1 / (h_coolant * A_die)
    R_tot     = R_die + R_pcb + R_TIM + R_HS + R_conv

    if Re < 2300:
        f_D = 64 / Re                              
    else:
        f_D = (0.790 * np.log(Re) - 1.64)**(-2)    

    dP     = f_D * (l_ch / D_h) * (0.5 * rho_f * u**2)
    dP_bar = dP / 1e5 

    return T_cl + W_diss * R_tot, R_conv, Re, u, dP_bar

def compute_Tj_Nu(Nu):
    x     = glycol_pct / 100.0
    rho_f = (1 - x) * rho_w + x * rho_g
    mu_f  = (1 - x) * mu_w  + x * mu_g
    cp_f  = (1 - x) * cp_w  + x * cp_g
    k_f   = (1 - x) * k_w   + x * k_g

    A_ch  = w_ch * h_ch
    P_ch  = 2 * (w_ch + h_ch)
    D_h   = 4 * A_ch / P_ch

    u    = m_dot / (rho_f * A_ch)
    Re   = rho_f * u * D_h / mu_f
    Pr   = (cp_f * mu_f) / k_f

    h_coolant = Nu * k_f / D_h
    R_conv    = 1 / (h_coolant * A_die)
    R_tot     = R_die + R_pcb + R_TIM + R_HS + R_conv
    return T_cl + W_diss * R_tot, R_conv, Re, Nu

def compute_Tj_m_dot(m_dot):
    x     = glycol_pct / 100.0
    rho_f = (1 - x) * rho_w + x * rho_g
    mu_f  = (1 - x) * mu_w  + x * mu_g
    cp_f  = (1 - x) * cp_w  + x * cp_g
    k_f   = (1 - x) * k_w   + x * k_g

    A_ch  = w_ch * h_ch
    P_ch  = 2 * (w_ch + h_ch)
    D_h   = 4 * A_ch / P_ch

    u    = m_dot / (rho_f * A_ch)
    Re   = rho_f * u * D_h / mu_f
    Pr   = (cp_f * mu_f) / k_f

    if Re < 5e5:
        Nu     = 0.664 * Re**(1/2) * Pr**(1/3)
        regime = "laminar"
    elif Re > 5e5:
        Nu     = 0.0296 * Re**(4/5) * Pr**(1/3)      
        regime = "turbulent"

    h_coolant = Nu * k_f / D_h
    R_conv    = 1 / (h_coolant * A_die)
    R_tot     = R_die + R_pcb + R_TIM + R_HS + R_conv
    return T_cl + W_diss * R_tot, R_conv, Re, m_dot

glycol_range = np.linspace(0, 60, 200)
width_range  = np.linspace(1e-3, 15e-3, 200)
m_dot_range  = np.linspace(0.08, 0.5, 200)
Nu_range     = np.linspace(100, 1000, 200)

T_j_glycol, R_conv_glycol, Re_glycol = zip(*[compute_Tj_glycol(g) for g in glycol_range])
T_j_glycol    = np.array(T_j_glycol)
R_conv_glycol = np.array(R_conv_glycol)
Re_glycol     = np.array(Re_glycol)

T_j_w, R_conv_w, Re_w, u_w, dP_bar_w = zip(*[compute_Tj_w(w) for w in width_range])
T_j_w    = np.array(T_j_w)
R_conv_w = np.array(R_conv_w)
Re_w     = np.array(Re_w)
u_w      = np.array(u_w)
dP_bar_w = np.array(dP_bar_w)
width_mm = width_range * 1e3

T_j_Nu, R_conv_Nu, Re_Nu, Nu = zip(*[compute_Tj_Nu(n) for n in Nu_range])
T_j_Nu    = np.array(T_j_Nu)
R_conv_Nu = np.array(R_conv_Nu)
Re_Nu     = np.array(Re_Nu)

T_j_m_dot, R_conv_m_dot, Re_m_dot, m_dot = zip(*[compute_Tj_m_dot(m) for m in m_dot_range])
T_j_m_dot    = np.array(T_j_m_dot)
R_conv_m_dot = np.array(R_conv_m_dot)
Re_m_dot     = np.array(Re_m_dot)

# plt: Glycol percentage vs. temperature junction
plt.figure(figsize=(10,5))
plt.plot(glycol_range, T_j_glycol)
plt.xlabel("Glycol concentration [%]")
plt.ylabel("T$_{junc}$ [°C]")
plt.title("Glycol concentration vs Junction temperature")
plt.savefig("T_junc_vs_glycol.png", dpi=150, bbox_inches="tight")


# plt: Glycol percentage vs. Reynolds number
plt.figure(figsize=(10,5))
plt.plot(glycol_range, Re_glycol)
plt.xlabel("Glycol concentration [%]")
plt.ylabel("Re [-]")
plt.title("Glycol concentration vs Reynolds number")
plt.savefig("T_junc_vs_w.png", dpi=150, bbox_inches="tight")

# plt: Channel width vs. temperature junction
plt.figure(figsize=(10,5))
plt.plot(width_mm, T_j_w)
plt.xlabel("Channel width [mm]")
plt.ylabel("T$_{junc}$ [°C]")
plt.title("Channel width vs Junction temperature")
plt.savefig("T_junc_vs_w.png", dpi=150, bbox_inches="tight")

# plt: Channel width vs. Reynolds number
plt.figure(figsize=(10,5))
plt.plot(width_mm, Re_w)
plt.xlabel("Channel width [mm]")
plt.ylabel("Re [-]")
plt.title("Channel width vs Reynolds number")
plt.savefig("Re_vs_w.png", dpi=150, bbox_inches="tight")

# plt: Channel width vs. pressure drop
plt.figure(figsize=(10,5))
plt.plot(width_mm, dP_bar_w)
plt.xlabel("Channel width [mm]")
plt.ylabel("$\Delta p$ [Bar]")
plt.title("Pressure drop vs Channel width")
plt.savefig("dP_vs_w.png", dpi=150, bbox_inches="tight")

# plt: mass flow vs. pressure drop
plt.figure(figsize=(10,5))
plt.plot(m_dot, T_j_m_dot)
plt.xlabel("$\dot m$ [kg/s]")
plt.ylabel("T$_{junc}$ [°C]")
plt.title("Mass flow vs Junction temperature}")
plt.savefig("m_dot_vs_Tjunc.png", dpi=150, bbox_inches="tight")

# plt: Nu vs. temperature junction
plt.figure(figsize=(10,5))
plt.plot(Nu, T_j_Nu)
plt.xlabel("Nu [-]")
plt.ylabel("T$_{junc}$ [°C]")
plt.title("Nusselt number vs Junction temperature")
plt.savefig("Nu_vs_Tjunc.png", dpi=150, bbox_inches="tight")