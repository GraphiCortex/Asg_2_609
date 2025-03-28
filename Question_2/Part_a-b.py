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

# Initial Conditions (order: [n, h, ca, v])
n0  = 0.0002
h0  = 0.01
ca0 = 0.103
v0  = -72.14
init_cond = [n0, h0, ca0, v0]

# Differential Equations Function: DiffEquations
def DiffEquations(t, y):
    """
    y[0] = n
    y[1] = h
    y[2] = ca
    y[3] = v
    """
    n, h, ca, v = y

    # K+ gating variable (n)
    ninf = 1.0 / (1.0 + np.exp((v - thetaN)/sigmaN))
    tauN = tauNbar / np.cosh((v - thetaN)/(2.0 * sigmaN))  # tauNbar divided by cosh((v-thetaN)/(2*sigmaN))
    
    # Na+ inactivation (h)
    alphaH = 0.128 * np.exp(-(v + 50.0)/18.0)
    betaH  = 4.0 / (1.0 + np.exp(-(v + 27.0)/5.0))
    hinf   = alphaH / (alphaH + betaH)
    
    # Na+ activation: 
    minf = 1.0 / (1.0 + np.exp((v - thetaM)/sigmaM))
    
    # Ionic currents:
    iNa = gNa * (minf**3) * h * (v - eNa)
    iK  = gK * (n**4) * (v - eK)
    
    # L-type Ca++ current: 
    sinf = 1.0 / (1.0 + np.exp((v - thetaS)/sigmaS))
    # To avoid division by zero in the denominator:
    denom = 1.0 - np.exp((2.0 * v) / RTF)
    if abs(denom) < 1e-9:
        iCaL = 0.0
    else:
        iCaL = gCaL * (sinf**2) * v * (Ca_ex / denom)
    
    # Leak current
    iL = gL * (v - eL)
    
    # Applied current: injected only between t_inj_on and t_inj_off
    iap = Iapp if (t >= t_inj_on and t <= t_inj_off) else 0.0

    # Differential equations:
    dn_dt  = (ninf - n) / tauN
    dh_dt  = (hinf - h) / tauH
    dca_dt = -f * (eps * iCaL + kca * (ca - 0.1))
    dv_dt  = (-iNa - iK - iCaL - iL + iap) / C

    return [dn_dt, dh_dt, dca_dt, dv_dt]

# Simulation: Run the model from 0 to 800 ms with dt=0.01 ms
t_start = 0.0
t_end   = 800.0
dt      = 0.01
t_eval  = np.arange(t_start, t_end + dt, dt)

sol = solve_ivp(DiffEquations, [t_start, t_end], init_cond, t_eval=t_eval,
                method='RK45', rtol=1e-6, atol=1e-8)
time_array = sol.t
n_sol, h_sol, ca_sol, v_sol = sol.y  # v_sol is the voltage trace


# Part (a): Compute Interspike Intervals (ISIs) & Plot Voltage Trace

def get_interspike_intervals(time_arr, voltage_arr, threshold=0.0):
    """
    Identify spikes using peaks above the given threshold and return the interspike intervals.
    """
    peaks, _ = find_peaks(voltage_arr, height=threshold)
    if len(peaks) < 2:
        return np.array([])
    spike_times = time_arr[peaks]
    return np.diff(spike_times)

isis = get_interspike_intervals(time_array, v_sol, threshold=0.0)
print("Part (a):")
print("Number of spikes detected:", len(isis)+1 if len(isis)>0 else 0)
print("Interspike intervals (ms):", isis)

# Plot the full voltage trace with spike markers and highlight the injection period.
plt.figure(figsize=(10,5))
plt.plot(time_array, v_sol, 'k-', linewidth=2, label="Voltage Trace")
spike_indices, _ = find_peaks(v_sol, height=0.0)
plt.plot(time_array[spike_indices], v_sol[spike_indices], 'ro', label="Detected Spikes")
plt.axvspan(t_inj_on, t_inj_off, color='gray', alpha=0.3, label="Current Injection")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Simulated Voltage Trace")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Part (b): Extract Spike Segments and Determine Threshold 

# Spike indices:
spike1_idx = spike_indices[0]
spike4_idx = spike_indices[3]

def extract_spike_segment(time_arr, voltage_arr, peak_idx, window_ms=5.0):
    """Extract a segment of the spike ±window_ms around the spike peak."""
    dt_local = time_arr[1] - time_arr[0]
    pts_window = int(window_ms / dt_local)
    start_idx = max(0, peak_idx - pts_window)
    end_idx = min(len(time_arr) - 1, peak_idx + pts_window)
    # Recenter time so that the spike peak is at 0
    return time_arr[start_idx:end_idx+1] - time_arr[peak_idx], voltage_arr[start_idx:end_idx+1]

# Extract segments for spike #1 and spike #4
t_spike1, v_spike1 = extract_spike_segment(time_array, v_sol, spike1_idx, window_ms=5.0)
t_spike4, v_spike4 = extract_spike_segment(time_array, v_sol, spike4_idx, window_ms=5.0)

def find_spike_threshold(t_seg, v_seg, dvdt_thresh=10.0):
    """Calculate dV/dt for the spike segment and return (time, voltage) where it first exceeds dvdt_thresh."""
    dt_seg = t_seg[1] - t_seg[0]
    dvdt = np.gradient(v_seg, dt_seg)
    crossing = np.where(dvdt >= dvdt_thresh)[0]
    if len(crossing) == 0:
        return None, None
    cross_idx = crossing[0]
    return t_seg[cross_idx], v_seg[cross_idx]

# Compute threshold for each spike
thresh_time1, thresh_v1 = find_spike_threshold(t_spike1, v_spike1, dvdt_thresh=10.0)
thresh_time4, thresh_v4 = find_spike_threshold(t_spike4, v_spike4, dvdt_thresh=10.0)

print("\nPart (b):")
print("Spike #1 threshold crossing: time = {:.2f} ms, voltage = {:.2f} mV".format(thresh_time1, thresh_v1))
print("Spike #4 threshold crossing: time = {:.2f} ms, voltage = {:.2f} mV".format(thresh_time4, thresh_v4))

# Plot spike segments in two subplots with threshold markers
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(t_spike1, v_spike1, 'b-', label="Spike #1")
if thresh_time1 is not None:
    axs[0].plot(thresh_time1, thresh_v1, 'ro', markersize=8, label="Threshold")
axs[0].set_title("Spike #1 with Threshold")
axs[0].set_xlabel("Time (ms) relative to peak")
axs[0].set_ylabel("Voltage (mV)")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t_spike4, v_spike4, 'b-', label="Spike #4")
if thresh_time4 is not None:
    axs[1].plot(thresh_time4, thresh_v4, 'ro', markersize=8, label="Threshold")
axs[1].set_title("Spike #4 with Threshold")
axs[1].set_xlabel("Time (ms) relative to peak")
axs[1].set_ylabel("Voltage (mV)")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
