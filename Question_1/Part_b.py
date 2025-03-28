import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# Same global parameters as above
gNa, vNa = 120.0, 50.0
gK,  vK  = 36.0,  -77.0
gL,  vL  = 0.3,   -54.4
C        = 1.0
Iapp     = 20.0
t_on, t_off = 150.0, 350.0
dt    = 0.01
t_end = 500.0

# We fix T1=25, T2=35
T1 = 25.0
T2 = 35.0

# Initial conditions
v_init = -65.0
m_init = 0.1
h_init = 0.1
n_init = 0.1

def alpha_m(v):
    if abs(v+40)<1e-9:
        return 0.1*(1e-9/(1-np.exp(-1e-9/10)))
    return 0.1*(v+40)/(1 - np.exp(-(v+40)/10))

def beta_m(v):
    return 4.0*np.exp(-(v+65)/18)

def alpha_h(v):
    return 0.07*np.exp(-(v+65)/20)

def beta_h(v):
    return 1.0/(1 + np.exp(-(v+35)/10))

def alpha_n(v):
    if abs(v+55)<1e-9:
        return 0.01*(1e-9/(1-np.exp(-1e-9/10)))
    return 0.01*(v+55)/(1 - np.exp(-(v+55)/10))

def beta_n(v):
    return 0.125*np.exp(-(v+65)/80)

def HH_ode(t, y, factor):
    v, m, h, n = y
    am = factor*alpha_m(v)
    bm = factor*beta_m(v)
    ah = factor*alpha_h(v)
    bh = factor*beta_h(v)
    an = factor*alpha_n(v)
    bn = factor*beta_n(v)

    dm = am*(1 - m) - bm*m
    dh = ah*(1 - h) - bh*h
    dn = an*(1 - n) - bn*n

    iNa = gNa*(m**3)*h*(v - vNa)
    iK  = gK *(n**4)*(v - vK)
    iL  = gL *(v - vL)

    iap = Iapp if (t >= t_on and t <= t_off) else 0.0
    dv  = (-iNa - iK - iL + iap)/C
    return [dv, dm, dh, dn]

# 2(b): vary Q10 from 1.5 to 8 (step 0.2), measure frequency
Q10_values = np.arange(1.5, 8.01, 0.2)
freqs = []

t_span = (0.0, t_end)
t_eval = np.arange(0.0, t_end+dt, dt)
y0 = [v_init, m_init, h_init, n_init]

for Q in Q10_values:
    factor = Q**((T2 - T1)/10.0)
    sol = solve_ivp(HH_ode, t_span, y0, t_eval=t_eval, args=(factor,),
                    method='RK45', rtol=1e-6, atol=1e-8)
    V = sol.y[0]
    t = sol.t
    # count spikes during [150, 350]
    mask = (t >= t_on) & (t <= t_off)
    peaks, _ = find_peaks(V[mask], height=0)
    num_spikes = len(peaks)
    freq = num_spikes / 0.2  # injection is 200 ms
    freqs.append(freq)

# Plot freq vs Q10
plt.figure(figsize=(7,5))
plt.plot(Q10_values, freqs, '-ob')
plt.xlabel('$Q_{10}$')
plt.ylabel('Firing Frequency (Hz)')
plt.title('Firing Frequency vs. $Q_{10}$ coefficient')
plt.grid(True)
plt.tight_layout()
plt.show()

# Now Q10=10 case => observe spike train
Q10_target = 10.0
factor_10 = Q10_target**((T2 - T1)/10.0)
sol_10 = solve_ivp(HH_ode, t_span, y0, t_eval=t_eval, args=(factor_10,),
                   method='RK45', rtol=1e-6, atol=1e-8)
V10 = sol_10.y[0]
t10 = sol_10.t

# Full trace
plt.figure(figsize=(9,5))
plt.plot(t10, V10, 'm-')
plt.axvspan(t_on, t_off, color='gray', alpha=0.2, label='Injection')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Voltage Trace for $Q_{10}=10$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Zoomed view on injection window
mask_inj = (t10 >= t_on) & (t10 <= t_off)
plt.figure(figsize=(7,4))
plt.plot(t10[mask_inj], V10[mask_inj], 'm-')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Zoomed Voltage Trace (150–350 ms) for $Q_{10}=10$')
plt.grid(True)
plt.tight_layout()
plt.show()
