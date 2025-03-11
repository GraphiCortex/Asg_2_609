import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Global parameters (set these as in the MATLAB code)
eNa = 50    # mV
eK = -90    # mV
eL = -70    # mV

gNa = 450   # nS
gK = 50     # nS
gL = 2      # nS
gCaL = 20   # nS

iapp = 100

# Rename MATLAB's "from" and "to" to avoid conflicts with Python keywords
t_from = 200
t_to = 600

tauNbar = 10  # mS
tauH = 1      # mS

thetaM = -35  # mV
thetaN = -30  # mV
sigmaM = -5   # mV
sigmaN = -5   # mV
thetaS = -20  # mV
sigmaS = -0.05  # mV

C = 100  # pF

RTF = 26.7
Ca_ex = 2.5
f = 0.1
eps = 0.0015
kca = 0.3

# Initial conditions: [n, h, ca, v]
n_i = 0.0002
h_i = 0.01
ca_i = 0.103
v_i = -72.14
init = [n_i, h_i, ca_i, v_i]

# Duration of the simulation (time in ms)
time = np.arange(0, 800.01, 0.01)

def DiffEquations(state, t):
    """
    Computes the derivatives for the model variables.
    
    Parameters:
        state : list or array
            The state variables [n, h, ca, v].
        t : float
            The current time (ms).
            
    Returns:
        list
            The derivatives [dn/dt, dh/dt, dca/dt, dv/dt].
    """
    n, h, ca, v = state
    
    # Na+ and K+ equations and currents
    minf = 1 / (1 + np.exp((v - thetaM) / sigmaM))
    ninf = 1 / (1 + np.exp((v - thetaN) / sigmaN))
    tauN = tauNbar / np.cosh((v - thetaN) / (2 * sigmaN))
    
    alphaH = 0.128 * np.exp(-(v + 50) / 18)
    betaH = 4 / (1 + np.exp(-(v + 27) / 5))
    hinf = alphaH / (alphaH + betaH)
    
    iNa = gNa * (minf ** 3) * h * (v - eNa)
    iK = gK * (n ** 4) * (v - eK)
    
    # L-Type Ca++ equations and current
    sinf = 1 / (1 + np.exp((v - thetaS) / sigmaS))
    iCaL = gCaL * (sinf ** 2) * v * (Ca_ex / (1 - np.exp((2 * v) / RTF)))
    
    # Leak current
    iL = gL * (v - eL)
    
    # Applied current: active between t_from and t_to
    if t_from <= t <= t_to:
        iap_current = iapp
    else:
        iap_current = 0

    # Differential equations
    dndt = (ninf - n) / tauN
    dhdt = (hinf - h) / tauH
    dca_dt = - f * (eps * iCaL + kca * (ca - 0.1))
    dvdt = (-iNa - iK - iCaL - iL + iap_current) / C
    
    return [dndt, dhdt, dca_dt, dvdt]

# Solve the ODE system using odeint.
solution = odeint(DiffEquations, init, time)
voltage = solution[:, 3]  # Voltage is the fourth variable

# Plot the voltage trace over time
plt.figure()
plt.plot(time, voltage, 'k-', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Voltage Trace')
plt.box(False)  # Similar to MATLAB's "box off"
plt.show()
