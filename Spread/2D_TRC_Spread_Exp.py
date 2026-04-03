import numpy as np

"""
Thermal resistance circuit for liquid-cooled chip stack.

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
t_HS    = 3e-3
k_HS    = 160
l_HS    = 100e-3         
w_HS    = 100e-3  

# == Die ==
W_diss  = 10
l_die   = 25.4e-3
w_die   = 25.4e-3

# == Channel geometry ==
nr_ch   = 14
l_ch    = 100e-3          
h_ch    = 25e-3
w_ch    = 2e-3

# == Coolant ==
T_cl        = 50
glycol_pct  = 0

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

u       = 2
m_dot   = 2*u * rho_f * (w_ch * h_ch)

"""
Geometry
"""

A_die   = l_die * w_die         
A_HS    = l_HS  * w_HS      

"""
Convective resistance

Sources
-------
For Nusselt number the Eq. was used which is also used in COMSOL

parameters
----------
A_ch    : channel cross-sectional area [m2]
P_ch    : wetted perimeter [m]
D_h     : hydraulic diameter [m]
Re      : Reynolds number [-]
Pr      : Prandtl number [-]
Nu      : Nusselt number [-]
h_cl    : convective heat transfer coefficient [W/m2K]
A_conv  : convective wall area (top + 2 sides) [m2]
"""

A_ch    = nr_ch * (w_ch * h_ch)
P_ch    = nr_ch * (w_ch + 2*h_ch)
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
R_conv  = 1 / (h_cl * nr_ch * A_conv)

"""
Spreading resistance (Lee et al., 1995)

The model considers a spreader plate of equivalent radius b and thickness t,
with a heat source of equivalent radius a on top, and a uniform convective
resistance R_o on the bottom.

parameters
----------
A_s         : heat source area (die footprint) [m2]
A_p         : spreader plate area [m2]
a           : equivalent source radius [m]
b           : equivalent plate radius [m]
eps         : dimensionless source radius, a/b [-]
tau         : dimensionless plate thickness, t/b [-]
Bi          : effective Biot number, 1/(pi*k*b*R_o) [-]
lambda_c    : eigenvalue [-]
Phi_c       : intermediate function [-]
Psi_avg     : dimensionless average spreading resistance [-]
Psi_max     : dimensionless maximum spreading resistance [-]
R_spread    : spreading resistance [K/W]
"""

def lee_spreading(A_s, A_p, t, k, R_o):
    """
    Compute spreading resistance using Lee et al. (1995) correlations.

    Parameters
    ----------
    A_s  : heat source area [m2]
    A_p  : spreader plate area [m2]
    t    : plate thickness [m]
    k    : plate thermal conductivity [W/mK]
    R_o  : external (convective) resistance at plate bottom [K/W]

    Returns
    -------
    Psi_avg, Psi_max, R_spread_avg, R_spread_max
    """
    a   = np.sqrt(A_s / np.pi)         # Eq. 1: equivalent source radius
    b   = np.sqrt(A_p / np.pi)         # Eq. 2: equivalent plate radius

    eps = a / b                         # Eq. 7: dimensionless source radius
    tau = t / b                         # Eq. 8: dimensionless plate thickness
    Bi  = 1 / (np.pi * k * b * R_o)   # Eq. 9: effective Biot number

    lambda_c = np.pi + 1.0 / (np.sqrt(np.pi) * eps)                        # Eq. 16
    Phi_c    = (np.tanh(lambda_c * tau) + lambda_c / Bi) / \
               (1 + (lambda_c / Bi) * np.tanh(lambda_c * tau))             # Eq. 15

    Psi_avg = (eps * tau) / np.sqrt(np.pi) + 0.5 * (1-eps)**(3/2) * Phi_c # Eq. 13
    Psi_max = (eps * tau) / np.sqrt(np.pi) + (1/np.sqrt(np.pi)) * (1-eps) * Phi_c  # Eq. 14

    R_spread_avg = Psi_avg / (np.sqrt(np.pi) * k * a)                      # Eq. 18
    R_spread_max = Psi_max / (np.sqrt(np.pi) * k * a)                      # Eq. 18

    return Psi_avg, Psi_max, R_spread_avg, R_spread_max

Psi_avg, Psi_max, R_spread_avg, R_spread_max = lee_spreading(A_die, A_HS, t_HS, k_HS, R_conv)
R_spread = R_spread_max   # use max (hotspot) for junction temperature

"""
Total thermal resistance and junction temperature
"""

R_tot   = R_spread + R_conv
T_junc  = T_cl + W_diss * R_tot

"""
Pressure drop (Darcy-Weisbach)
"""

if Re < 2300:
    f_D = 64 / Re
else:
    f_D = (0.790 * np.log(Re) - 1.64)**(-2)

dP      = f_D * (l_ch / D_h) * (0.5 * rho_f * u**2)
dP_bar  = dP * 1e-5

"""
Print results
"""

print(f"{'Layer':<30} {'t [mm]':>8}  {'k [W/mK]':>9}  {'A [cm²]':>8}  {'R [K/W]':>9}  {'share':>6}")
print("-" * 80)
for name, t, k, A, R in [
    ("Spreading",  None, None, A_die,  R_spread),
    ("Convection", None, None, A_conv, R_conv),
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


# ==============================================================================
# VERIFICATION against Lee et al. (1994) Tables 1 & 2
#
# The paper measured R_fin-air directly for a 100x100mm aluminium heat sink
# at three air velocities. We plug those in as R_o and reproduce their table.
#
# Geometry (paper): b = 58mm (100x100mm base), t = 3mm, k_Al ~ 160 W/mK
# Source sizes:     a = 14.3mm (25.4x25.4mm die), a = 5.4mm (8.9x10.2mm die)
# ==============================================================================

print()
print("=" * 95)
print("VERIFICATION: Lee et al. (1994) — Table 1 (avg) and Table 2 (max)")
print("=" * 95)

k_Al    = 160.0          # aluminium heat sink [W/mK]
t_paper = 3e-3           # base thickness [m]
l_paper = 100e-3
w_paper = 100e-3
A_base  = l_paper * w_paper

# Two source sizes from the paper
sources = [
    {"label": "a=14.3mm (25.4x25.4mm)", "A_s": 25.4e-3 * 25.4e-3, "a_mm": 14.3},
    {"label": "a=5.4mm  (8.9x10.2mm) ", "A_s": 8.9e-3  * 10.2e-3, "a_mm":  5.4},
]

# Measured R_fin-air and corresponding Bi from the paper
cases = [
    {"velocity": 1, "Bi_paper": 0.046, "R_fin_air": 0.79},
    {"velocity": 3, "Bi_paper": 0.074, "R_fin_air": 0.49},
    {"velocity": 5, "Bi_paper": 0.099, "R_fin_air": 0.37},
]

# Paper's calculated R_spread values for cross-check (Table 1 avg, Table 2 max)
paper_Psi_avg = {14.3: [0.750, 0.743, 0.737], 5.4: [0.655, 0.653, 0.652]}
paper_Psi_max = {14.3: [0.971, 0.962, 0.955], 5.4: [0.774, 0.773, 0.771]}
paper_R_spread_avg = {14.3: [0.20, 0.20, 0.19], 5.4: [0.46, 0.46, 0.46]}
paper_R_spread_max = {14.3: [0.25, 0.25, 0.25], 5.4: [0.54, 0.54, 0.54]}
paper_R_hs_avg = {14.3: [0.98, 0.69, 0.56], 5.4: [1.25, 0.95, 0.82]}
paper_R_hs_max = {14.3: [1.04, 0.74, 0.62], 5.4: [1.33, 1.03, 0.91]}

for src in sources:
    a_mm = src["a_mm"]
    print()
    print(f"  Source: {src['label']}   (ε = {np.sqrt(src['A_s']/np.pi) / np.sqrt(A_base/np.pi):.3f})")
    print(f"  {'v[m/s]':>7}  {'Bi':>6}  {'R_o[°C/W]':>10}  "
          f"{'Ψ_avg':>7}  {'Ψ_max':>7}  "
          f"{'Rs_avg':>8}  {'Rs_max':>8}  "
          f"{'Rhs_avg calc':>13}  {'Rhs_avg paper':>14}  "
          f"{'Rhs_max calc':>13}  {'Rhs_max paper':>14}")
    print("  " + "-" * 130)

    for i, case in enumerate(cases):
        R_o = case["R_fin_air"]
        Psi_avg_c, Psi_max_c, Rs_avg, Rs_max = lee_spreading(
            src["A_s"], A_base, t_paper, k_Al, R_o
        )
        R_hs_avg_calc = R_o + Rs_avg
        R_hs_max_calc = R_o + Rs_max

        print(f"  {case['velocity']:>7}  {case['Bi_paper']:>6.3f}  {R_o:>10.2f}  "
              f"  {Psi_avg_c:>6.3f}  {Psi_max_c:>6.3f}  "
              f"  {Rs_avg:>7.3f}  {Rs_max:>7.3f}  "
              f"  {R_hs_avg_calc:>12.3f}  {paper_R_hs_avg[a_mm][i]:>13.2f}  "
              f"  {R_hs_max_calc:>12.3f}  {paper_R_hs_max[a_mm][i]:>13.2f}")

print()
print("  Note: Small differences in Ψ vs paper are expected — the paper's Bi is derived")
print("  from their measured h, whereas here we back-calculate Bi from R_o via Eq. 9.")