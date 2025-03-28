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

# Applied current (pA)
Iapp = 100.0

# Current injection interval (ms)
t_inj_on  = 200.0
t_inj_off = 600.0

# Time constants (ms)
tauNbar = 10.0
tauH    = 1.0

# Half-activation constants (mV) & slopes
thetaM = -35.0
sigmaM = -5.0
thetaN = -30.0
sigmaN = -5.0
thetaS = -20.0
sigmaS = -0.05

# Capacitance (pF)
C = 100.0

# Other constants
RTF   = 26.7
Ca_ex = 2.5
f     = 0.1
eps   = 0.0015
kca   = 0.3

# Additional Parameter for SK Current 
gSK_current = 3.0  
ks_val = 0.5  

# Initial Conditions (order: [n, h, ca, v])
n0  = 0.0002
h0  = 0.01
ca0 = 0.103
v0  = -72.14
init_cond = [n0, h0, ca0, v0]

# ODE System: DiffEquations_SK
def DiffEquations_SK(t, y):
    """
    Modified HH ODEs that include the Ca²⁺-dependent K⁺ (SK) current.
    State vector: [n, h, ca, v]
    """
    n, h, ca, v = y

    # K+ gating for n
    ninf = 1.0 / (1.0 + np.exp((v - thetaN)/sigmaN))
    tauN = tauNbar / np.cosh((v - thetaN)/(2.0 * sigmaN))
    
    # Na+ inactivation for h
    alphaH = 0.128 * np.exp(-(v + 50.0)/18.0)
    betaH  = 4.0 / (1.0 + np.exp(-(v + 27.0)/5.0))
    hinf   = alphaH / (alphaH + betaH)
    
    # Na+ activation:
    minf = 1.0 / (1.0 + np.exp((v - thetaM)/sigmaM))
    
    # Ionic Currents
    iNa = gNa * (minf**3) * h * (v - eNa)
    iK  = gK  * (n**4) * (v - eK)
    
    # L-type Ca²⁺ current
    sinf = 1.0 / (1.0 + np.exp((v - thetaS)/sigmaS))
    denom = 1.0 - np.exp((2.0*v)/RTF)
    if abs(denom) < 1e-9:
        iCaL = 0.0
    else:
        iCaL = gCaL * (sinf**2) * v * (Ca_ex/denom)
    
    # Leak current
    iL = gL * (v - eL)
    
    k_inf = (ca**2) / (ca**2 + ks_val**2)
    iSK = gSK_current * k_inf * (v - eK)
    
    # Applied current: active from t_inj_on to t_inj_off
    iap = Iapp if (t >= t_inj_on and t <= t_inj_off) else 0.0

    # Differential Equations
    dn_dt  = (ninf - n) / tauN
    dh_dt  = (hinf - h) / tauH
    dca_dt = -f * (eps * iCaL + kca * (ca - 0.1))
    # SK current is an additional outward current
    dv_dt  = (-iNa - iK - iCaL - iL - iSK + iap) / C

    return [dn_dt, dh_dt, dca_dt, dv_dt]

# Simulation: Run the modified model (with SK current) from 0 to 800 ms
t_start = 0.0
t_end   = 800.0
dt      = 0.01
t_eval  = np.arange(t_start, t_end + dt, dt)

sol_SK = solve_ivp(DiffEquations_SK, [t_start, t_end], init_cond, t_eval=t_eval,
                   method='RK45', rtol=1e-6, atol=1e-8)
time_SK = sol_SK.t
n_SK, h_SK, ca_SK, v_SK = sol_SK.y

# Compute interspike intervals (ISIs) for the SK model
def get_interspike_intervals(time_arr, voltage_arr, threshold=0.0):
    peaks, _ = find_peaks(voltage_arr, height=threshold)
    if len(peaks) < 2:
        return np.array([])
    spike_times = time_arr[peaks]
    return np.diff(spike_times)

isis_SK = get_interspike_intervals(time_SK, v_SK, threshold=0.0)
print("Part (c) with gSK = 3 nS:")
print("Number of spikes detected:", len(isis_SK)+1 if len(isis_SK) > 0 else 0)
print("Interspike intervals (ms):", isis_SK)

# Plot the voltage trace for the SK model
plt.figure(figsize=(10,5))
plt.plot(time_SK, v_SK, 'k-', linewidth=2, label="Voltage Trace (with SK)")
spikes_SK, _ = find_peaks(v_SK, height=0.0)
plt.plot(time_SK[spikes_SK], v_SK[spikes_SK], 'ro', label="Detected Spikes")
plt.axvspan(t_inj_on, t_inj_off, color='gray', alpha=0.3, label="Current Injection")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Trace with Ca²⁺-dependent K⁺ Current (gSK = 3 nS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Quantify the Relationship: Vary gSK and Measure Firing Frequency
gSK_range = np.linspace(0, 6, 13)  # from 0 to 6 nS in 0.5 nS increments
freq_vs_gSK = []

for gSK_val in gSK_range:
    # Update the SK conductance for this simulation
    gSK_current = gSK_val
    sol_temp = solve_ivp(DiffEquations_SK, [t_start, t_end], init_cond, t_eval=t_eval,
                         method='RK45', rtol=1e-6, atol=1e-8)
    t_temp = sol_temp.t
    v_temp = sol_temp.y[3]
    # Count spikes only during the injection period
    mask_temp = (t_temp >= t_inj_on) & (t_temp <= t_inj_off)
    peaks_temp, _ = find_peaks(v_temp[mask_temp], height=0.0)
    num_spikes = len(peaks_temp)
     
    freq = num_spikes / ((t_inj_off - t_inj_on)/1000.0)
    freq_vs_gSK.append(freq)

# Plot firing frequency versus gSK_current
plt.figure(figsize=(8,5))
plt.plot(gSK_range, freq_vs_gSK, 'o-', color='b')
plt.xlabel("g_SK (nS)")
plt.ylabel("Firing Frequency (Hz)")
plt.title("Firing Frequency vs. SK Current Conductance")
plt.grid(True)
plt.tight_layout()
plt.show()

