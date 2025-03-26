import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# Global-like parameters 
gNa, vNa = 120.0, 50.0
gK,  vK  = 36.0,  -77.0
gL,  vL  = 0.3,   -54.4
C        = 1.0
Iapp     = 20.0   
t_on, t_off = 150.0, 350.0  # injection window
dt = 0.01
t_end = 500.0

# Temperature parameters
T1 = 25.0
Q10 = 1.5

# Initial conditions
v_init = -65.0
m_init = 0.1
h_init = 0.1
n_init = 0.1

# Rate functions (alpha/beta) 

def alpha_m(v):
    if abs(v + 40) < 1e-9:
        return 0.1 * (1e-9 / (1 - np.exp(-1e-9/10)))
    return 0.1*(v + 40)/(1 - np.exp(-(v+40)/10))

def beta_m(v):
    return 4.0*np.exp(-(v+65)/18)

def alpha_h(v):
    return 0.07*np.exp(-(v+65)/20)

def beta_h(v):
    return 1.0/(1 + np.exp(-(v+35)/10))

def alpha_n(v):
    if abs(v + 55) < 1e-9:
        return 0.01*(1e-9 / (1 - np.exp(-1e-9/10)))
    return 0.01*(v + 55)/(1 - np.exp(-(v+55)/10))

def beta_n(v):
    return 0.125*np.exp(-(v+65)/80)

# ODE system with temperature scaling factor
def HH_ode(t, y, factor):
    v, m, h, n = y

    # gating rates scaled by factor
    am = factor * alpha_m(v)
    bm = factor * beta_m(v)
    ah = factor * alpha_h(v)
    bh = factor * beta_h(v)
    an = factor * alpha_n(v)
    bn = factor * beta_n(v)

    # gating derivatives
    dm = am*(1 - m) - bm*m
    dh = ah*(1 - h) - bh*h
    dn = an*(1 - n) - bn*n

    # ionic currents
    iNa = gNa*(m**3)*h*(v - vNa)
    iK  = gK *(n**4)*(v - vK)
    iL  = gL *(v - vL)

    # applied current
    iap = Iapp if (t >= t_on and t <= t_off) else 0.0

    dv = (-iNa - iK - iL + iap)/C
    return [dv, dm, dh, dn]

T2_vals = np.arange(30, 71, 5)
freqs = []

# We'll store voltage/time for T2=30 and T2=60 to overlay spikes
trace_30 = None
trace_60 = None

for T2 in T2_vals:
    factor = Q10 ** ((T2 - T1)/10.0)
    # Solve ODE
    t_span = (0.0, t_end)
    t_eval = np.arange(0.0, t_end+dt, dt)
    y0 = [v_init, m_init, h_init, n_init]
    sol = solve_ivp(HH_ode, t_span, y0, t_eval=t_eval, args=(factor,),
                    method='RK45', rtol=1e-6, atol=1e-8)
    V = sol.y[0]
    t = sol.t

    # Count spikes during injection
    mask = (t >= t_on) & (t <= t_off)
    peaks, _ = find_peaks(V[mask], height=0)
    num_spikes = len(peaks)
    # injection lasts 200 ms => freq in Hz
    freq = num_spikes / 0.2
    freqs.append(freq)

    # Save trace if T2=30 or T2=60
    if T2 == 30:
        trace_30 = (t, V)
    elif T2 == 60:
        trace_60 = (t, V)

# Plot freq vs T2
plt.figure(figsize=(7,5))
plt.plot(T2_vals, freqs, 'o-b')
plt.xlabel('T2 (°C)')
plt.ylabel('Firing Frequency (Hz)')
plt.title(f'1(a): Freq vs T2, Q10={Q10}, T1={T1}°C, Iapp=20')
plt.grid(True)
plt.tight_layout()
plt.show()

# Extract a single spike from T2=30 and T2=60 to overlay
def extract_spike(t, V, threshold=0.0, window=5.0):
    """Find the first spike above 'threshold' and extract +/- window ms around it."""
    peaks, _ = find_peaks(V, height=threshold)
    if len(peaks)==0:
        return None, None
    pk_idx = peaks[0]
    pk_time = t[pk_idx]
    # pick indices in [pk_time-window, pk_time+window]
    start_t = pk_time - window
    end_t   = pk_time + window
    # find closest indices
    start_i = np.searchsorted(t, start_t)
    end_i   = np.searchsorted(t, end_t)
    return t[start_i:end_i] - pk_time, V[start_i:end_i]

plt.figure(figsize=(8,5))
colors = {30:'r', 60:'g'}
for Tsel, trace in [(30, trace_30), (60, trace_60)]:
    if trace is not None:
        tV, VV = trace
        t_spk, V_spk = extract_spike(tV, VV, threshold=0.0, window=5.0)
        if t_spk is not None:
            plt.plot(t_spk, V_spk, label=f'T2={Tsel}°C', color=colors[Tsel])

plt.xlabel('Time (ms) relative to spike peak')
plt.ylabel('Membrane Potential (mV)')
plt.title('Single Spike Overlay at T2=30 and 60°C')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
