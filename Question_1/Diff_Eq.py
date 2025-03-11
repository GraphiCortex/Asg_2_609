import numpy as np

# Global variables must be defined somewhere in your code.
# For example:
# vNa = ...
# vK = ...
# vL = ...
# gNa = ...
# gK = ...
# gL = ...
# C = ...
# Iapp = ...
# t_from = ...   # instead of 'from'
# t_to = ...     # instead of 'to'

def DiffEquations(t, sol):
    global vNa, vK, vL, gNa, gK, gL, C, Iapp, t_from, t_to

    v = sol[0]
    m = sol[1]
    h = sol[2]
    n = sol[3]

    # Na+ Equations and Currents
    alphaM = 0.1 * ((v + 40) / (1 - np.exp(-(v + 40) / 10)))
    betaM = 4 * np.exp(-(v + 65) / 18)

    alphaH = 0.07 * np.exp(-(v + 65) / 20)
    betaH = 1 / (np.exp(-(v + 35) / 10) + 1)

    iNa = gNa * (m ** 3) * h * (v - vNa)

    # K+ Equations and Currents
    alphaN = 0.01 * ((v + 55) / (1 - np.exp(-(v + 55) / 10)))
    betaN = 0.125 * np.exp(-(v + 65) / 80)

    iK = gK * (n ** 4) * (v - vK)

    # Leak current
    iL = gL * (v - vL)

    if t >= t_from and t <= t_to:
        iap = Iapp
    else:
        iap = 0

    # Create the output vector
    output = np.zeros(4)
    output[0] = (-iNa - iK - iL + iap) / C
    output[1] = alphaM * (1 - m) - betaM * m
    output[2] = alphaH * (1 - h) - betaH * h
    output[3] = alphaN * (1 - n) - betaN * n

    return output
