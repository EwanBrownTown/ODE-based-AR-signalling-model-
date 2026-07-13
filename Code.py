import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import messagebox

# ODE function
def ar_odes(t, y, Vm, Ki, gamma_P, gamma_C, k1, k2, k3, k4, S):
    P, C, D = y

    # Calculate rates
    V1 = Vm / (1 + D / Ki)
    V2 = gamma_P * P
    V3 = k1 * S * P - k2 * C
    V4 = gamma_C * C
    V5 = k3 * C**2 - k4 * D

    # Rate of change
    dPdt = V1 - V2 - V3
    dCdt = V3 - V4 - V5
    dDdt = V5

    return [dPdt, dCdt, dDdt]