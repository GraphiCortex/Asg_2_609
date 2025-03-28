import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# Global Parameters (from the professor’s MATLAB code and earlier parts)
# Nernst potentials (mV)
eNa = 50.0
eK  = -90.0
eL  = -70.0

# Conductances (nS)
gNa  = 450.0
gK   = 50.0
gL   = 2.0
gCaL = 20.0

# Applied current (pA)
# SK current parameters (we keep gSK constant as in Part (c) by default)
gSK_current_default = 3.0  
ks_val = 0.5

# T-type Ca current parameters – gCaT will be a variable parameter
# Activation parameters for T-type current:
theta_a = -65.0
sigma_a = -7.8
# Inactivation function parameters:
theta_b = 0.4
sigma_b = -0.1
# Parameters for inactivation ODE:
phi_r = 0.2
theta_r = -67.0
sigma_r = 2.0
tau_r0 = 40.0
tau_r1 = 17.5
theta_rT = 68.0
sigma_rT = 2.2

# Other constants:
RTF   = 26.7
Ca_ex = 2.5
f     = 0.1
eps   = 0.0015
kca   = 0.3

# Capacitance (pF)
C = 100.0

# Initial Conditions for the T-type augmented model:
# State vector: [n, h, ca, r, v]
n0  = 0.0002
h0  = 0.01
ca0 = 0.103
r0  = 0.01    # T-type inactivation variable initial condition
v0  = -72.14
init_cond = [n0, h0, ca0, r0, v0]

# Define a "safe" exponential function to prevent\reduce overflow warnings.
def safe_exp(x, lower=-50, upper=50):
    return np.exp(np.clip(x, lower, upper))

# ODE System: Extended model with T-type Ca current 
def DiffEquations_T(t, y, gCaT, Iapp_val, inj_on, inj_off, gSK_current):
    """
    Extended HH model with:
      - Na+, K+, L-type Ca2+ currents,
      - Ca2+-dependent K+ (SK) current,
      - T-type Ca2+ current.
    State vector: y = [n, h, ca, r, v]
    gCaT: T-type Ca current conductance (nS)
    Iapp_val: applied current (pA)
    inj_on, inj_off: injection interval (ms)
    gSK_current: SK current conductance (nS)
    """
    n, h, ca, r, v = y

    # --- Na and K currents (using parameters from earlier parts) ---
    # K+ gating for n
    ninf = 1.0/(1.0+np.exp((v - (-30.0))/(-5.0)))  # thetaN=-30, sigmaN=-5
    tauN = 10.0/np.cosh((v - (-30.0))/(2.0*(-5.0)))   # tauNbar=10
    # Na+ inactivation for h
    alphaH = 0.128 * safe_exp(-(v+50.0)/18.0)
    betaH  = 4.0/(1.0+np.exp(-(v+27.0)/5.0))
    hinf   = alphaH/(alphaH+betaH)
    # Na+ activation: thetaM=-35, sigmaM=-5
    minf = 1.0/(1.0+np.exp((v - (-35.0))/(-5.0)))
    iNa = gNa * (minf**3) * h * (v - eNa)
    iK  = gK  * (n**4) * (v - eK)
    
    # --- L-type Ca2+ current ---
    # Use safe_exp to prevent overflow in the exponent.
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
    iSK = gSK_current * k_inf * (v - eK)
    
    # --- T-type Ca2+ current ---
    a_inf = 1.0/(1.0+np.exp((v - theta_a)/sigma_a))
    b_inf = 1.0/(1.0+np.exp((r - theta_b)/sigma_b)) - 1.0/(1.0+np.exp(-theta_b/sigma_b))
    denom_CaT = 1.0 - safe_exp((2.0*v)/RTF)
    if abs(denom_CaT) < 1e-9:
        iCaT = 0.0
    else:
        iCaT = gCaT * (a_inf**3) * (b_inf**3) * v * (Ca_ex/denom_CaT)
    
    # --- Applied current ---
    Iapp_current = Iapp_val if (t >= inj_on and t <= inj_off) else 0.0

    # --- T-type inactivation variable r dynamics ---
    r_inf = 1.0/(1.0+np.exp((v - (-67.0))/(2.0)))  # theta_r=-67, sigma_r=2
    tau_r = tau_r0 + tau_r1/(1.0+np.exp((r - theta_rT)/sigma_rT))  # theta_rT=68, sigma_rT=2.2
    dr_dt = phi_r * (r_inf - r) / tau_r

    # --- Calcium dynamics ---
    dca_dt = -f * (eps * iCaL + kca * (ca - 0.1))
    
    # --- Voltage equation: Sum all currents ---
    dv_dt = (-iNa - iK - iCaL - iL - iSK - iCaT + Iapp_current) / C
    
    dn_dt = (ninf - n) / tauN
    dh_dt = (hinf - h) / 1.0  # tauH = 1
    
    return [dn_dt, dh_dt, dca_dt, dr_dt, dv_dt]

# Helper function to run the simulation with specified parameters.
def run_model(gCaT, Iapp_val, inj_on, inj_off, gSK_current_val):
    t_start = 0.0
    t_end = 1000.0
    dt = 0.01
    t_eval = np.arange(t_start, t_end+dt, dt)
    sol = solve_ivp(DiffEquations_T, [t_start, t_end], init_cond, t_eval=t_eval,
                    args=(gCaT, Iapp_val, inj_on, inj_off, gSK_current_val),
                    method='RK45', rtol=1e-6, atol=1e-8)
    return sol.t, sol.y[4]  # Return time and voltage (v is 5th variable)

# Subpart (i):
# For Iapp = -100 pA, simulate for gCaT = 1, 2, 3, 4 nS and overlay voltage responses.
Iapp_val_i = -100.0
inj_on_i = 200.0
inj_off_i = 600.0
gSK_current_fixed = gSK_current_default  # 3 nS fixed for now
gCaT_values = [1.0, 2.0, 3.0, 4.0]

plt.figure(figsize=(10,6))
for gCaT_val in gCaT_values:
    t_sim, v_sim = run_model(gCaT_val, Iapp_val_i, inj_on_i, inj_off_i, gSK_current_fixed)
    plt.plot(t_sim, v_sim, label=f"$g_{{CaT}}$ = {gCaT_val} nS")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Traces for Different $g_{CaT}$ ($I_{app}$ = -100 pA)")
plt.axvspan(inj_on_i, inj_off_i, color='gray', alpha=0.3, label="Injection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Subpart (ii):
# For gCaT = 3 nS, vary Iapp = -10, -50, -100 pA and overlay voltage responses.
gCaT_fixed = 3.0
Iapp_values = [-10.0, -50.0, -100.0]
plt.figure(figsize=(10,6))
for I_val in Iapp_values:
    t_sim, v_sim = run_model(gCaT_fixed, I_val, inj_on_i, inj_off_i, gSK_current_fixed)
    plt.plot(t_sim, v_sim, label=f"$I_{{app}}$ = {I_val} pA")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Traces for Different $I_{app}$ ($g_{CaT}$ = 3 nS)")
plt.axvspan(inj_on_i, inj_off_i, color='gray', alpha=0.3, label="Injection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Subpart (iii):
# For Iapp = -100 pA and gCaT = 3 nS, set inj_on = 100 ms and vary the duration of injection (to = 100, 200, 300, 400 ms).
Iapp_val_iii = -100.0
gCaT_fixed = 3.0
inj_on_iii = 100.0
injection_durations = [100.0, 200.0, 300.0, 400.0]  # durations in ms

plt.figure(figsize=(10,6))
for dur in injection_durations:
    inj_off_iii = inj_on_iii + dur
    t_sim, v_sim = run_model(gCaT_fixed, Iapp_val_iii, inj_on_iii, inj_off_iii, gSK_current_fixed)
    plt.plot(t_sim, v_sim, label=f"Duration = {dur} ms")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Traces for Varying Injection Durations ($I_{app}$ = -100 pA, $g_{CaT}$ = 3 nS)")
plt.axvspan(inj_on_iii, inj_on_iii + max(injection_durations), color='gray', alpha=0.3, label="Injection Window")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
