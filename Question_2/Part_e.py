import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# Global Parameters 
# Nernst potentials (mV)
eNa = 50.0
eK  = -90.0
eL  = -70.0

# Conductances (nS)
gNa  = 450.0
gK   = 50.0
gL   = 2.0
gCaL = 20.0

# SK current parameters (from Part (c))
gSK_default = 3.0  
ks_val = 0.5       

# T-type Ca²⁺ current parameters (gCaT will be provided)
theta_a = -65.0
sigma_a = -7.8
theta_b = 0.4
sigma_b = -0.1
phi_r = 0.2
tau_r0 = 40.0
tau_r1 = 17.5
theta_rT = 68.0
sigma_rT = 2.2

# Parameter values:
# g_H will be set per simulation.
k_r = 0.3
E_H = -30.0
tau_rf = 100.0    
tau_rs = 1500.0   
theta_rf = -105.0
theta_rs = -105.0
sigma_rf = 5.0
sigma_rs = 25.0

# Other constants:
RTF   = 26.7
Ca_ex = 2.5
f     = 0.1
eps   = 0.0015
kca   = 0.3

# Capacitance (pF)
C = 100.0

# Injection and Applied Current Parameters for Part (e)
# For subparts (i) and (ii), we use:
inj_on = 200.0   
inj_off = 600.0  
Iapp_default = -100.0  

# Initial Conditions for the Extended Model
# State vector: [n, h, ca, r, v, r_f, r_s]
n0  = 0.0002
h0  = 0.01
ca0 = 0.103
r0  = 0.01    # T-type inactivation variable
v0  = -72.14
rf0 = 0.03    # Fast component of I_H
rs0 = 0.03    # Slow component of I_H
init_cond = [n0, h0, ca0, r0, v0, rf0, rs0]

def safe_exp(x, lower=-50, upper=50):
    return np.exp(np.clip(x, lower, upper))


# ODE System: Extended Model with T-type and I_H currents 
def DiffEquations_E(t, y, gCaT, g_H_val, Iapp_val, inj_on, inj_off, gSK_val):
    """
    Extended HH model including:
      - Na+, K+, L-type Ca²⁺ currents,
      - Ca²⁺-dependent K⁺ (SK) current,
      - T-type Ca²⁺ current,
      - Hyperpolarization-activated inward current (I_H).
    State vector: y = [n, h, ca, r, v, r_f, r_s]
    
    Parameters:
      gCaT: T-type Ca²⁺ conductance (nS)
      g_H_val: I_H conductance (nS)
      Iapp_val: Applied current (pA)
      inj_on, inj_off: Injection interval (ms)
      gSK_val: SK current conductance (nS)
    """
    n, h, ca, r, v, r_f, r_s = y

    # --- Na+ and K+ currents (as in previous parts) ---
    # K+ gating for n (using thetaN=-30, sigmaN=-5, tauNbar=10)
    ninf = 1.0/(1.0 + np.exp((v - (-30.0))/(-5.0)))
    tauN = 10.0/np.cosh((v - (-30.0))/(2.0*(-5.0)))
    # Na+ inactivation for h (tauH=1)
    alphaH = 0.128 * safe_exp(-(v + 50.0)/18.0)
    betaH  = 4.0/(1.0 + np.exp(-(v + 27.0)/5.0))
    hinf   = alphaH/(alphaH + betaH)
    # Na+ activation: minf (thetaM=-35, sigmaM=-5)
    minf = 1.0/(1.0+np.exp((v - (-35.0))/(-5.0)))
    iNa = gNa * (minf**3) * h * (v - eNa)
    iK  = gK  * (n**4) * (v - eK)
    
    # --- L-type Ca²⁺ current ---
    sinf = 1.0/(1.0+safe_exp((v - (-20.0))/(-0.05)))
    denom_CaL = 1.0 - safe_exp((2.0*v)/RTF)
    if abs(denom_CaL) < 1e-9:
        iCaL = 0.0
    else:
        iCaL = gCaL * (sinf**2) * v * (Ca_ex/denom_CaL)
    
    # --- Leak current ---
    iL = gL * (v - eL)
    
    # --- SK current ---
    k_inf = (ca**2) / (ca**2 + ks_val**2)
    iSK = gSK_val * k_inf * (v - eK)
    
    # --- T-type Ca²⁺ current ---
    a_inf = 1.0/(1.0 + np.exp((v - theta_a)/sigma_a))
    b_inf = 1.0/(1.0 + np.exp((r - theta_b)/sigma_b)) - 1.0/(1.0 + np.exp(-theta_b/sigma_b))
    denom_CaT = 1.0 - safe_exp((2.0*v)/RTF)
    if abs(denom_CaT) < 1e-9:
        iCaT = 0.0
    else:
        iCaT = gCaT * (a_inf**3) * (b_inf**3) * v * (Ca_ex/denom_CaT)
    
    # --- Hyperpolarization-activated current (I_H) ---
    I_H = g_H_val * (k_r * r_f - (1 - k_r)*r_s) * (v - E_H)
    
    # --- Applied current ---
    Iapp_current = Iapp_val if (t >= inj_on and t <= inj_off) else 0.0
    
    # --- T-type inactivation variable r dynamics ---
    r_inf = 1.0/(1.0+np.exp((v - (-67.0))/(2.0)))  # theta_r=-67, sigma_r=2
    tau_r = tau_r0 + tau_r1/(1.0+np.exp((r - theta_rT)/sigma_rT))
    dr_dt = phi_r * (r_inf - r) / tau_r
    
    # --- Calcium dynamics ---
    dca_dt = -f * (eps * iCaL + kca * (ca - 0.1))
    
    # --- Voltage equation: Sum all currents ---
    dv_dt = (-iNa - iK - iCaL - iL - iSK - iCaT - I_H + Iapp_current) / C
    
    # --- n and h dynamics ---
    dn_dt = (ninf - n) / tauN
    dh_dt = (hinf - h) / 1.0  # tauH = 1
    
    # --- I_H components: dynamics for r_f and r_s ---
    r_f_inf = 1.0/(1.0 + np.exp((v - theta_rf)/sigma_rf))
    r_s_inf = 1.0/(1.0 + np.exp((v - theta_rs)/sigma_rs))
    drf_dt = (r_f_inf - r_f) / tau_rf
    drs_dt = (r_s_inf - r_s) / tau_rs
    
    return [dn_dt, dh_dt, dca_dt, dr_dt, dv_dt, drf_dt, drs_dt]

# Helper function to run the extended model simulation
def run_model_E(gCaT, g_H_val, Iapp_val, inj_on, inj_off, gSK_val):
    t_start = 0.0
    t_end = 1400.0
    dt = 0.01
    t_eval = np.arange(t_start, t_end+dt, dt)
    sol = solve_ivp(DiffEquations_E, [t_start, t_end], init_cond, t_eval=t_eval,
                    args=(gCaT, g_H_val, Iapp_val, inj_on, inj_off, gSK_val),
                    method='RK45', rtol=1e-6, atol=1e-8)
    return sol.t, sol.y[4]  # Return time and voltage


# Subpart (i):
# For Iapp = -100 pA, gCaT = 3 nS, inj_on = 200 ms, inj_off = 600 ms,
# simulate for g_H = 1, 2, 4 nS and overlay the voltage traces.
Iapp_val_e = -100.0
gCaT_e = 3.0
inj_on_e = 200.0
inj_off_e = 600.0
gSK_val_e = gSK_default  # 3 nS
g_H_values = [1.0, 2.0, 4.0]

plt.figure(figsize=(10,6))
for g_H_val in g_H_values:
    t_sim, v_sim = run_model_E(gCaT_e, g_H_val, Iapp_val_e, inj_on_e, inj_off_e, gSK_val_e)
    plt.plot(t_sim, v_sim, label=f"$g_{{H}}$ = {g_H_val} nS")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Traces for Different $g_{H}$\n($I_{app}$ = -100 pA, $g_{CaT}$ = 3 nS)")
plt.axvspan(inj_on_e, inj_off_e, color='gray', alpha=0.3, label="Injection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Subpart (ii):
# For gCaT = 0 and 3 nS, and g_H = 0 and 4 nS, overlay responses.
combinations = [
    (0.0, 0.0),  # (gCaT, g_H) = (0, 0)
    (0.0, 4.0),  # (0, 4)
    (3.0, 0.0),  # (3, 0)
    (3.0, 4.0)   # (3, 4)
]
plt.figure(figsize=(10,6))
for (gCaT_val, g_H_val) in combinations:
    t_sim, v_sim = run_model_E(gCaT_val, g_H_val, Iapp_val_e, inj_on_e, inj_off_e, gSK_val_e)
    plt.plot(t_sim, v_sim, label=f"$g_{{CaT}}$ = {gCaT_val}, $g_{{H}}$ = {g_H_val}")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Traces for Different ($g_{CaT}$, $g_{H}$) Combinations")
plt.axvspan(inj_on_e, inj_off_e, color='gray', alpha=0.3, label="Injection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Subpart (iii):
# For Iapp = -100 pA and gCaT = 3 nS, fix inj_on = 100 ms and vary injection duration.
Iapp_val_iii = -100.0
gCaT_fixed = 3.0
# For this subpart, inj_on is fixed at 100 ms.
inj_on_iii = 100.0
# Let's vary injection durations (in ms)
injection_durations = [100.0, 200.0, 300.0, 400.0]

# For this subpart, we also need to choose a value for g_H, say 2.0 nS.
g_H_fixed = 2.0

plt.figure(figsize=(10,6))
for dur in injection_durations:
    inj_off_iii = inj_on_iii + dur
    t_sim, v_sim = run_model_E(gCaT_fixed, g_H_fixed, Iapp_val_iii, inj_on_iii, inj_off_iii, gSK_val_e)
    plt.plot(t_sim, v_sim, label=f"Duration = {dur} ms")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Traces for Varying Injection Durations\n($I_{app}$ = -100 pA, $g_{CaT}$ = 3 nS, $g_H$ = 2 nS)")
plt.axvspan(inj_on_iii, inj_on_iii + max(injection_durations), color='gray', alpha=0.3, label="Injection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
