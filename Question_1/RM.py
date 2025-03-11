import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Global parameters (make sure these are defined before calling the ODE solver)
t_from = 150
t_to = 350

C = 1

gNa = 120  
vNa = 50
gK = 36  
vK = -77
gL = 0.3  
vL = -54.4
Iapp = 20

def DiffEquations(sol, t):
    global vNa, vK, vL, gNa, gK, gL, C, Iapp, t_from, t_to
    v, m, h, n = sol

    # Na+ Equations and Currents
    alphaM = 0.1 * ((v + 40) / (1 - np.exp(-(v + 40) / 10)))
    betaM = 4 * np.exp(-(v + 65) / 18)
    
    alphaH = 0.07 * np.exp(-(v + 65) / 20)
    betaH = 1 / (np.exp(-(v + 35) / 10) + 1)
    
    iNa = gNa * (m**3) * h * (v - vNa)
    
    # K+ Equations and Currents
    alphaN = 0.01 * ((v + 55) / (1 - np.exp(-(v + 55) / 10)))
    betaN = 0.125 * np.exp(-(v + 65) / 80)
    
    iK = gK * (n**4) * (v - vK)
    
    # Leak current
    iL = gL * (v - vL)
    
    # Applied current: only active between t_from and t_to
    if t_from <= t <= t_to:
        iap = Iapp
    else:
        iap = 0
    
    dvdt = (-iNa - iK - iL + iap) / C
    dmdt = alphaM * (1 - m) - betaM * m
    dhdt = alphaH * (1 - h) - betaH * h
    dndt = alphaN * (1 - n) - betaN * n
    
    return [dvdt, dmdt, dhdt, dndt]

# Time span and initial conditions
tspan = np.arange(0, 500.01, 0.01)  # equivalent to MATLAB's 0:0.01:500

v_i = -65
m_i = 0.1
h_i = 0.1
n_i = 0.1
init_cond = [v_i, m_i, h_i, n_i]

# Integrate the ODE
sol = odeint(DiffEquations, init_cond, tspan)
modelTrace = sol[:, 0]  # Voltage is the first element in the state vector

# Plot the voltage trace over time
plt.figure()
plt.plot(tspan, modelTrace, 'k-', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.title('Model Voltage Trace')
plt.show()
