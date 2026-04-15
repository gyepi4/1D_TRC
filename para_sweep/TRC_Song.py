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
mu      : Dynamic viscosity [Pa*s]
cp      : Specific heat [J/kgK]
"""

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_PARAMS = {
    # Die (IMCQ120R017M2HXTMA1)
    # Source: https://plm.prodrive.nl/Component/2210-0000-0338?tab=d
    "W_diss"        : 25,
    "l_die"         : 15.5e-3,
    "w_die"         : 12e-3,

    # Heatsink (EN-AW-6060 T66)
    # Source: Roloff / Matek
    "t_HS"          : 4e-3,
    "k_HS"          : 160,
    "l_HS"          : 30e-3,
    "w_HS"          : 20e-3,

    # Channel geometry
    "h_ch"          : 10.4e-3,
    "w_ch"          : 10.4e-3,
    "l_ch"          : 30e-3,

    # Coolant
    "T_cl"          : 45,
    "glycol_pct"    : 40,
    "u"             : 1.8,           

    # Water properties
    # Source: www.thermalexcel.com/english/tables/eau_atm.html
    "rho_w"         : 988.0,
    "mu_w"          : 0.000547,
    "cp_w"          : 4130.87,
    "k_w"           : 0.641,

    # Glycol properties
    "rho_g"         : 1050.440,
    "mu_g"          : 1.538e-3,
    "cp_g"          : 3499.0,
    "k_g"           : 0.4108,

    "Nu"            : None,
}   


# =============================================================================
# COMPUTE FUNCTION
# =============================================================================

def compute_thermal(params):
    """
    Compute thermal resistance circuit for liquid-cooled chip stack.

    Sources
    -------
    Lee et al.
    Song et al.
    COMSOL

    parameters
    ----------
    A_ch        : channel cross-sectional area [m2]
    P_ch        : wetted perimeter [m]
    D_h         : hydraulic diameter [m]
    Re          : Reynolds number [-]
    Pr          : Prandtl number [-]
    Nu          : Nusselt number [-]
    h_cl        : convective heat transfer coefficient [W/m2K]
    A_conv      : convective wall area (top + 2 sides) [m2]
    A_s         : heat source area (die footprint) [m2]
    A_p         : spreader plate area (channel wall area) [m2]
    a           : equivalent source radius [m]
    b           : equivalent plate radius [m]
    eps         : dimensionless source radius [-]
    tau         : dimensionless plate thickness [-]
    Bi          : Biot number [-]
    lambda_c    : eigenvalue [-]
    Phi_c       : function [-]
    Psi_avg     : dimensionless average spreading resistance [-]
    R_spread    : Spreading resistance [K/W]
    """
    p = {**DEFAULT_PARAMS, **params}

    W_diss      = p["W_diss"]
    l_die       = p["l_die"]
    w_die       = p["w_die"]
    t_HS        = p["t_HS"]
    k_HS        = p["k_HS"]
    l_HS        = p["l_HS"]
    w_HS        = p["w_HS"]
    h_ch        = p["h_ch"]
    w_ch        = p["w_ch"]
    l_ch        = p["l_ch"]
    T_cl        = p["T_cl"]
    glycol_pct  = p["glycol_pct"]
    u           = p["u"]
    rho_w       = p["rho_w"]
    mu_w        = p["mu_w"]
    cp_w        = p["cp_w"]
    k_w         = p["k_w"]
    rho_g       = p["rho_g"]
    mu_g        = p["mu_g"]
    cp_g        = p["cp_g"]
    k_g         = p["k_g"]
    Nu_override = p["Nu"]

    x       = glycol_pct / 100.0
    rho_f   = (1 - x) * rho_w + x * rho_g
    mu_f    = (1 - x) * mu_w  + x * mu_g
    cp_f    = (1 - x) * cp_w  + x * cp_g
    k_f     = (1 - x) * k_w   + x * k_g

    m_dot   = u * rho_f * (w_ch * h_ch)

    # print(f"rho_f = {rho_f}")

    # == Determining the convective resistance (Source: COMSOL) == 
    A_ch    = w_ch * h_ch
    P_ch    = w_ch + 2 * h_ch
    D_h     = (4 * A_ch) / P_ch

    Re      = rho_f * u * l_ch / mu_f
    Pr      = (cp_f * mu_f) / k_f

    if Nu_override is not None:
        Nu      = Nu_override      
        regime  = "overridden"
    elif Re < 5e5:
        Nu      = 0.3387 * Pr**(1/3) * Re**(1/2) / (1 + (0.0468/Pr)**(2/3))**(1/4)
        regime  = "laminar"
    else:
        Nu      = Pr**(1/3) * (0.037 * Re**(4/5) - 871)
        regime  = "turbulent"

    h_cl    = (2 * Nu * k_f) / l_ch
    A_conv  = 2 * (h_ch * l_ch) + w_ch * l_ch
    R_conv  = 1 / (h_cl * A_conv)

    # == Determining the spreading resistance (Source: Lee et al. & Song et al.) ==
    A_s     = l_die * w_die
    A_p     = l_HS  * w_HS

    a       = np.sqrt(A_s / np.pi)
    b       = np.sqrt(A_p / np.pi)

    eps     = a / b
    tau     = t_HS / b
    Bi      = 1 / (np.pi * k_HS * b * R_conv)

    lambda_c = np.pi + (1.0 / (np.sqrt(np.pi) * eps))
    Phi_c    = (np.tanh(lambda_c * tau) + (lambda_c / Bi)) / (1 + (lambda_c / Bi) * np.tanh(lambda_c * tau))

    Psi_max  = (eps * tau) / np.sqrt(np.pi) + (1/np.sqrt(np.pi)) * (1 - eps) * Phi_c

    R_spread = Psi_max / (np.sqrt(np.pi) * k_HS * a)

    # == Total thermal resistance == 
    R_tot   = R_spread + R_conv
    T_junc  = T_cl + W_diss * R_tot

    # == Pressure drop ==
    if Re < 2300:
        f_D = 64 / Re
    else:
        f_D = (0.790 * np.log(Re) - 1.64)**(-2)

    dP = f_D * (l_ch / D_h) * (0.5 * rho_f * u**2)

    return {
        "T_junc"    : T_junc,
        "R_tot"     : R_tot,
        "R_spread"  : R_spread,
        "R_conv"    : R_conv,
        "Nu"        : Nu,
        "Re"        : Re,
        "Pr"        : Pr,
        "h_cl"      : h_cl,
        "dP"        : dP,
        "dP_bar"    : dP * 1e-5,
        "m_dot"     : m_dot,
        "regime"    : regime,
    }


def sweep(param_key, values, extra_params=None):
    """
    Sweep parameters
    """
    if extra_params is None:
        extra_params = {}
    results = None
    for v in values:
        r = compute_thermal({**extra_params, param_key: v})
        if results is None:
            results = {k: [] for k in r}
            results[param_key] = []
        results[param_key].append(v)
        for k, val in r.items():
            results[k].append(val)
    return results


# =============================================================================
# BASELINE
# =============================================================================

baseline = compute_thermal({})
print("=== Baseline ===")
print(f"  T_junc   = {baseline['T_junc']:.2f} °C")
print(f"  R_tot    = {baseline['R_tot']:.3f} K/W")
print(f"  R_spread = {baseline['R_spread']:.3f} K/W")
print(f"  R_conv   = {baseline['R_conv']:.3f} K/W")
print(f"  Nu       = {baseline['Nu']:.2f}")
print(f"  Re       = {baseline['Re']:.0f}  ({baseline['regime']})")
print(f"  dP       = {baseline['dP_bar']:.4f} bar")
print(f"  mdot     = {baseline['m_dot']:} kg/s")


# =============================================================================
# SWEEPS
# =============================================================================

# == w_ch vs. Tjun ==
width_mm    = np.linspace(1, 20, 50)
r           = sweep("w_ch", width_mm * 1e-3)
T_j_w       = r["T_junc"]

plt.figure(figsize=(10, 5))
plt.plot(width_mm, T_j_w)
plt.xlabel("Channel width [mm]")
plt.ylabel("T$_{junc}$ [°C]")
plt.title("Channel width vs. Junction temperature")
plt.savefig("T_junc_vs_w_ch.png", dpi=150, bbox_inches="tight")


# == Nu vs. Tjun ==
Nu_sweep    = np.linspace(100, 1000, 100)
r           = sweep("Nu", Nu_sweep)
T_j_Nu       = r["T_junc"]

plt.figure(figsize=(10, 5))
plt.plot(Nu_sweep, T_j_Nu)
plt.xlabel("Nusselt number [-]")
plt.ylabel("T$_{junc}$ [°C]")
plt.title("Nusselt number vs. Junction temperature")
plt.savefig("T_junc_vs_Nu.png", dpi=150, bbox_inches="tight")


# == Glycol content vs. Tjun ==
glycol_arr  = np.linspace(0, 100, 100)
r           = sweep("glycol_pct", glycol_arr)
T_j_g       = r["T_junc"]

plt.figure(figsize=(10, 5))
plt.plot(glycol_arr, T_j_g)
plt.xlabel("Glycol content [%]")
plt.ylabel("T$_{junc}$ [°C]")
plt.title("Glycol percentage vs. Junction temperature")
plt.savefig("T_junc_vs_glycol.png", dpi=150, bbox_inches="tight")


# == Die area vs. Tjun ==
aspect      = DEFAULT_PARAMS["l_die"] / DEFAULT_PARAMS["w_die"]
l_die_arr   = np.linspace(5e-3, 30e-3, 50)
A_die_mm2   = []
T_j_A       = []

for l in l_die_arr:
    res = compute_thermal({"l_die": l, "w_die": l / aspect})
    A_die_mm2.append(l * (l / aspect) * 1e6)
    T_j_A.append(res["T_junc"])

plt.figure(figsize=(10, 5))
plt.plot(A_die_mm2, T_j_A)
plt.xlabel("Die area [mm2]")
plt.ylabel("T$_{junc}$ [°C]")
plt.title("Die area vs. Junction temperature")
plt.savefig("T_junc_vs_die_area.png", dpi=150, bbox_inches="tight")


# == Tcl vs. Tjun ==
T_cl_arr    = np.linspace(5, 45, 50)
r           = sweep("T_cl", T_cl_arr)
T_j_Tcl     = r["T_junc"]

plt.figure(figsize=(10, 5))
plt.plot(T_cl_arr, T_j_Tcl)
plt.xlabel("Coolant inlet temperature [°C]")
plt.ylabel("T$_{junc}$ [°C]")
plt.title("Coolant temperature vs. Junction temperature")
plt.savefig("T_junc_vs_T_cl.png", dpi=150, bbox_inches="tight")

# == t_hs vs. Tjun ==
t_hs_arr    = np.linspace(1e-3, 10e-3, 50)
r           = sweep("t_HS", t_hs_arr)
T_j_t_hs     = r["T_junc"]

t_hs_optima  = np.argmin(T_j_t_hs)

plt.figure(figsize=(10, 5))
plt.plot(t_hs_arr*1e3, T_j_t_hs)
plt.scatter(t_hs_arr[t_hs_optima]*1e3, T_j_t_hs[t_hs_optima], color="red", zorder=5, label=f"Optimum = {t_hs_arr[t_hs_optima]*1e3:.3f} [mm]")
plt.xlabel("Heatsink thickness [m]")
plt.ylabel("T$_{junc}$ [°C]")
plt.legend()
plt.title("Heatsink thickness vs. Junction temperature")
plt.savefig("T_junc_vs_t_hs.png", dpi=150, bbox_inches="tight")



