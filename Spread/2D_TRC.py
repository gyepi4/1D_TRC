import numpy as np
import matplotlib.pyplot as plt

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
mu      : Dynamic viscosity [Pa·s]
cp      : Specific heat [J/kgK]
"""

# == Die ==
R_die   = 0.05
W_diss  = 150
l_die   = 30e-3
w_die   = 20e-3

# == PCB ==
t_pcb   = 2e-3
k_pcb   = 160

# == Heatsink ==
t_HS    = 6e-3
k_HS    = 160

# == Channel geometry ==
l_ch    = l_die
h_ch    = 10e-3
w_ch    = 3e-3
w_wall  = 1e-3

# == Coolant ==
T_cl        = 50
glycol_pct  = 0

# Glycol properties (constant)
rho_g   = 1060.0
mu_g    = 1.4e-3
cp_g    = 3500.0
k_g     = 0.253

# Water properties (constant at T_cl)
rho_w   = 988.0
mu_w    = 5.5e-4
cp_w    = 4181.0
k_w     = 0.644

# Coolant mixture
x       = glycol_pct / 100.0
rho_f   = (1 - x) * rho_w + x * rho_g
mu_f    = (1 - x) * mu_w  + x * mu_g
cp_f    = (1 - x) * cp_w  + x * cp_g
k_f     = (1 - x) * k_w   + x * k_g

u       = 2
m_dot   = u * rho_f * (w_ch * h_ch)

"""
Standard 1D conduction resistances
"""

A_die   = l_die * w_die

R_pcb   = t_pcb / (k_pcb * A_die)
R_HS    = t_HS  / (k_HS  * A_die)

"""
Convective resistance

Nusselt number from Churchill-Bernstein flat-plate correlation
(matches COMSOL's built-in heat flux BC formulation).
L = l_ch is the characteristic length; the factor 2 converts
the local Nu at x=L to the plate-averaged h.

Sources
-------
COMSOL Heat Transfer Module, Heat Flux BC formulation
https://www.nuclear-power.com/nuclear-engineering/fluid-dynamics/internal-flow/hydraulic-diameter-2/

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

A_ch    = w_ch * h_ch
P_ch    = 2 * (w_ch + h_ch)
D_h     = 4 * A_ch / P_ch

Re      = rho_f * u * l_ch / mu_f
Pr      = (cp_f * mu_f) / k_f

if Re < 5e5:
    Nu      = 0.3387 * Pr**(1/3) * Re**(1/2) / (1 + (0.0468/Pr)**(2/3))**(1/4)
    regime  = "laminar"
else:
    Nu      = Pr**(1/3) * (0.037 * Re**(4/5) - 871)
    regime  = "turbulent"

h_cl    = 2 * Nu * k_f / l_ch
A_conv  = 2 * (h_ch * l_ch) + w_ch * l_ch      # top + 2 sides
R_conv  = 1 / (h_cl * A_conv)

"""
Spreading resistance (Lee et al., 1995)

Heat enters the heatsink over A_die and spreads to reach the
channel wall area A_conv. Since A_conv > A_die the geometry is
spreading (not constriction). The Lee et al. circular-source
model converts both areas to equivalent circular radii.

R_spread_total gives total conduction + spreading from source to
cooled surface. Subtracting the 1D conduction over A_p isolates
the pure spreading correction R_spread, which is added on top of
the explicit R_HS term.

Sources
-------
Lee, S., Song, S., Au, V., and Moran, K., "Constriction/Spreading
Resistance Model for Electronics Packaging," ASME/JSME Thermal
Engineering Conference, 1995.

Simons, R.E., "Simple Formulas for Estimating Thermal Spreading
Resistance," Electronics Cooling, May 2004.
https://www.electronics-cooling.com/2004/05/simple-formulas-for-estimating-thermal-spreading-resistance/

Qpedia, "Spreading Resistance of Single and Multiple Heat Sources,"
Qpedia Thermal eMagazine, September 2010.

parameters
----------
A_s         : heat source area (die footprint) [m2]
A_p         : spreader plate area (channel wall area) [m2]
a           : equivalent source radius [m]
b           : equivalent plate radius [m]
eps         : dimensionless source radius = a/b [-]
tau         : dimensionless plate thickness = t_HS/b [-]
Bi          : effective Biot number = h_cl*b/k_HS [-]
lambda_c    : eigenvalue parameter [-]
Phi_c       : auxiliary function [-]
Psi_avg     : dimensionless average spreading resistance [-]
R_spread    : pure spreading resistance (additive to R_HS) [K/W]
"""

A_s     = A_die     # heat source: die footprint
A_p     = A_conv    # heat sink:   channel wall area (A_conv > A_die -> spreading)

a       = np.sqrt(A_s / np.pi)      # equivalent source radius (eq. 9)
b       = np.sqrt(A_p / np.pi)      # equivalent plate radius  (eq. 9)

eps     = a / b                     # dimensionless source radius  (eq. 2)
tau     = t_HS / b                  # dimensionless plate thickness (eq. 3)
Bi      = h_cl * b / k_HS          # effective Biot number         (eq. 4)

lambda_c    = np.pi + 1.0 / np.sqrt(np.pi * eps)                               # eq. 8
Phi_c       = (np.tanh(lambda_c * tau) + lambda_c / Bi) / \
              (1 + (lambda_c / Bi) * np.tanh(lambda_c * tau))                  # eq. 7
Psi_avg     = (eps * tau / np.sqrt(np.pi)) + 0.5 * (1 - eps)**1.5 * Phi_c     # eq. 6

R_spread_total  = Psi_avg / (k_HS * np.sqrt(A_s))  # total: 1D conduction + spreading
R_HS_plate      = t_HS / (k_HS * A_p)              # 1D conduction reference over A_p
R_spread        = R_spread_total - R_HS_plate       # pure spreading correction

"""
Total thermal resistance and junction temperature
"""

R_tot   = R_pcb + R_HS + R_spread + R_conv
T_junc  = T_cl + W_diss * R_tot

"""
Pressure drop (Darcy-Weisbach)

Sources
-------
https://www.nuclear-power.com/nuclear-engineering/fluid-dynamics/major-head-loss-friction-loss/darcy-weisbach-equation/

parameters
----------
f_D     : Darcy friction factor [-]
dP      : pressure drop [Pa]
"""

if Re < 2300:
    f_D = 64 / Re
else:
    f_D = (0.790 * np.log(Re) - 1.64)**(-2)

dP      = f_D * (l_ch / D_h) * (0.5 * rho_f * u**2)
dP_bar  = dP / 1e5

"""
Print results
"""

print(f"{'Layer':<30} {'t [mm]':>8}  {'k [W/mK]':>9}  {'A [cm²]':>8}  {'R [K/W]':>9}  {'share':>6}")
print("-" * 80)
for name, t, k, A, R in [
    ("PCB",           t_pcb,  k_pcb,  A_die,   R_pcb    ),
    ("Heatsink (1D)", t_HS,   k_HS,   A_die,   R_HS     ),
    ("Spreading",     None,   None,   None,    R_spread  ),
    ("Convection",    None,   None,   A_conv,  R_conv    ),
]:
    t_str = f"{t*1e3:>7.2f}" if t is not None else f"{'—':>7}"
    k_str = f"{k:>8.1f}"     if k is not None else f"{'—':>8}"
    A_str = f"{A*1e4:>7.2f}" if A is not None else f"{'—':>7}"
    print(f"  {name:<28} {t_str}   {k_str}   {A_str}   {R:>9.5f}  {100*R/R_tot:>5.1f}%")
print("-" * 80)
print(f"  {'TOTAL':<57}   {R_tot:>9.5f}  100.0%")

print()
print(f"Spreading resistance parameters (Lee et al., 1995):")
print("-" * 80)
print(f"  A_s (die footprint)  =   {A_s*1e4:.2f} [cm²]")
print(f"  A_p (channel walls)  =   {A_p*1e4:.2f} [cm²]")
print(f"  eps = a/b            =   {eps:.4f} [-]")
print(f"  tau = t/b            =   {tau:.4f} [-]")
print(f"  Bi                   =   {Bi:.4f} [-]")
print(f"  Phi_c                =   {Phi_c:.4f} [-]")
print(f"  Psi_avg              =   {Psi_avg:.4f} [-]")
print(f"  R_spread             =   {R_spread:.5f} [K/W]")

print()
print(f"Dimensionless numbers:")
print("-" * 80)
print(f"  Re                   =   {Re:.4f} [-]")
print(f"  Pr                   =   {Pr:.4f} [-]")
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