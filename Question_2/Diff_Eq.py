import numpy as np

# Global parameters should be defined elsewhere in your code.
# For example:
# eNa, eK, eL, gNa, gK, gL, gCaL, tauNbar, tauH, Ca_ex, RTF,
# thetaM, sigmaM, thetaN, sigmaN, thetaS, sigmaS, f, iapp, eps, kca, C,
# t_from, t_to = ...

def DiffEquations(time, init):
    global eNa, eK, eL, gNa, gK, gL, gCaL, tauNbar, tauH, Ca_ex, RTF
    global thetaM, sigmaM, thetaN, sigmaN, thetaS, sigmaS, f, iapp, eps, kca, C, t_from, t_to
    
    n = init[0]
    h = init[1]
    ca = init[2]
    v = init[3]
    
    # Na+ and K+ Equations and Currents
    minf = 1 / (1 + np.exp((v - thetaM) / sigmaM))
    ninf = 1 / (1 + np.exp((v - thetaN) / sigmaN))
    tauN = tauNbar / np.cosh((v - thetaN) / (2 * sigmaN))
    
    alphaH = 0.128 * np.exp(-(v + 50) / 18)
    betaH = 4 / (1 + np.exp(-(v + 27) / 5))
    hinf = alphaH / (alphaH + betaH)
    
    iNa = gNa * (minf ** 3) * h * (v - eNa)
    iK = gK * (n ** 4) * (v - eK)
    
    # L-Type Ca++ Equations and Current
    sinf = 1 / (1 + np.exp((v - thetaS) / sigmaS))
    iCaL = gCaL * (sinf ** 2) * v * (Ca_ex / (1 - np.exp((2 * v) / RTF)))
    
    # Leak current
    iL = gL * (v - eL)
    
    # Applied current only active between t_from and t_to
    if t_from <= time <= t_to:
        iap_current = iapp
    else:
        iap_current = 0
    
    # Define the differential equations
    dndt = (ninf - n) / tauN
    dhdt = (hinf - h) / tauH
    dca_dt = - f * (eps * iCaL + kca * (ca - 0.1))
    dvdt = (-iNa - iK - iCaL - iL + iap_current) / C
    
    return [dndt, dhdt, dca_dt, dvdt]
