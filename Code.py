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

def read_params():
    # Read the shared parameters, returns None if something is wrong
    try:
        Vm      = float(entry_Vm.get())
        Ki      = float(entry_Ki.get())
        gamma_P = float(entry_gamma_P.get())
        gamma_C = float(entry_gamma_C.get())
        k1      = float(entry_k1.get())
        k2      = float(entry_k2.get())
        k3      = float(entry_k3.get())
        k4      = float(entry_k4.get())
    except ValueError:
        messagebox.showerror("Input error", "All fields must be numbers.")
        return None

    return (Vm, Ki, gamma_P, gamma_C, k1, k2, k3, k4)

def run_model():
    params = read_params()
    if params is None:
        return

    try:
        S       = float(entry_S.get())
        t_start = float(entry_tstart.get())
        t_end   = float(entry_tend.get())
    except ValueError:
        messagebox.showerror("Input error", "All fields must be numbers.")
        return

    if t_start >= t_end:
        messagebox.showerror("Input error", "Start time must be less than end time.")
        return

    # always solve from 0
    #t_start onwards
    time_points = np.linspace(0, t_end, 1000)
    sol = solve_ivp(ar_odes, [0, t_end], [0, 0, 0],
                    args=params + (S,),
                    t_eval=time_points)

    # results
    t = sol.t
    P = sol.y[0]
    C = sol.y[1]
    D = sol.y[2]

    mask = t >= t_start
    t = t[mask]
    P = P[mask]
    C = C[mask]
    D = D[mask]

    # Basic plot
    plt.figure()
    plt.plot(t, P, label="P (free AR)")
    plt.plot(t, C, label="C (complex)")
    plt.plot(t, D, label="D (dimer)")

    plt.annotate(f"P = {P[-1]:.3f}", xy=(t[-1], P[-1]))
    plt.annotate(f"C = {C[-1]:.3f}", xy=(t[-1], C[-1]))
    plt.annotate(f"D = {D[-1]:.3f}", xy=(t[-1], D[-1]))

    plt.xlabel("Time (hours)")
    plt.ylabel("Concentration")
    plt.title("AR Model")
    plt.legend()
    plt.show()

    # Print the final steady state values
    print("Final values:")
    print("P =", P[-1])
    print("C =", C[-1])
    print("D =", D[-1])

def sweep_S():
    params = read_params()
    if params is None:
        return

    try:
        S_min  = float(entry_Smin.get())
        S_max  = float(entry_Smax.get())
        S_step = float(entry_Sstep.get())
        t_end  = float(entry_tend.get())
    except ValueError:
        messagebox.showerror("Input error", "All fields must be numbers.")
        return

    if S_min >= S_max:
        messagebox.showerror("Input error", "S min must be less than S max.")
        return

    if S_step <= 0:
        messagebox.showerror("Input error", "S step must be positive.")
        return

    # the extra half step makes sure S_max is included
    S_values = np.arange(S_min, S_max + S_step/2, S_step)

    final_P = []
    final_C = []
    final_D = []

    # run the model once for each testosterone value
    for S in S_values:
        sol = solve_ivp(ar_odes, [0, t_end], [0, 0, 0],
                        args=params + (S,))
        final_P.append(sol.y[0][-1])
        final_C.append(sol.y[1][-1])
        final_D.append(sol.y[2][-1])

    final_P = np.array(final_P)
    final_C = np.array(final_C)
    final_D = np.array(final_D)

    # dose response plot
    plt.figure()
    plt.plot(S_values, final_P, label="P (free AR)")
    plt.plot(S_values, final_C, label="C (complex)")
    plt.plot(S_values, final_D, label="D (dimer)")
    plt.plot(S_values, final_C + final_D, "--", label="C + D (total bound)")

    plt.xlabel("S (testosterone)")
    plt.ylabel("Steady state concentration")
    plt.title("Dose Response")
    plt.legend()
    plt.show()

    print("Dose response:")
    for i in range(len(S_values)):
        print(f"S = {S_values[i]:.2f}   P = {final_P[i]:.3f}   C = {final_C[i]:.3f}   D = {final_D[i]:.3f}   C+D = {final_C[i] + final_D[i]:.3f}")

# window
root = tk.Tk()
root.title("AR Model")

# parameter inputs
tk.Label(root, text="Vm").grid(row=0, column=0)
entry_Vm = tk.Entry(root)
entry_Vm.insert(0, "1.0")
entry_Vm.grid(row=0, column=1)

tk.Label(root, text="Ki").grid(row=1, column=0)
entry_Ki = tk.Entry(root)
entry_Ki.insert(0, "1.0")
entry_Ki.grid(row=1, column=1)

tk.Label(root, text="gamma_P").grid(row=2, column=0)
entry_gamma_P = tk.Entry(root)
entry_gamma_P.insert(0, "0.5")
entry_gamma_P.grid(row=2, column=1)

tk.Label(root, text="gamma_C").grid(row=3, column=0)
entry_gamma_C = tk.Entry(root)
entry_gamma_C.insert(0, "0.5")
entry_gamma_C.grid(row=3, column=1)

tk.Label(root, text="k1").grid(row=4, column=0)
entry_k1 = tk.Entry(root)
entry_k1.insert(0, "0.1")
entry_k1.grid(row=4, column=1)

tk.Label(root, text="k2").grid(row=5, column=0)
entry_k2 = tk.Entry(root)
entry_k2.insert(0, "0.05")
entry_k2.grid(row=5, column=1)

tk.Label(root, text="k3").grid(row=6, column=0)
entry_k3 = tk.Entry(root)
entry_k3.insert(0, "0.1")
entry_k3.grid(row=6, column=1)

tk.Label(root, text="k4").grid(row=7, column=0)
entry_k4 = tk.Entry(root)
entry_k4.insert(0, "0.05")
entry_k4.grid(row=7, column=1)

tk.Label(root, text="S (testosterone)").grid(row=8, column=0)
entry_S = tk.Entry(root)
entry_S.insert(0, "1.0")
entry_S.grid(row=8, column=1)

tk.Label(root, text="Start time (hours)").grid(row=9, column=0)
entry_tstart = tk.Entry(root)
entry_tstart.insert(0, "0")
entry_tstart.grid(row=9, column=1)

tk.Label(root, text="End time (hours)").grid(row=10, column=0)
entry_tend = tk.Entry(root)
entry_tend.insert(0, "50")
entry_tend.grid(row=10, column=1)

# sweep inputs
tk.Label(root, text="S min").grid(row=11, column=0)
entry_Smin = tk.Entry(root)
entry_Smin.insert(0, "0")
entry_Smin.grid(row=11, column=1)

tk.Label(root, text="S max").grid(row=12, column=0)
entry_Smax = tk.Entry(root)
entry_Smax.insert(0, "10")
entry_Smax.grid(row=12, column=1)

tk.Label(root, text="S step").grid(row=13, column=0)
entry_Sstep = tk.Entry(root)
entry_Sstep.insert(0, "0.1")
entry_Sstep.grid(row=13, column=1)

# buttons
tk.Button(root, text="Run Model", command=run_model).grid(row=14, column=0, columnspan=2)
tk.Button(root, text="Sweep S", command=sweep_S).grid(row=15, column=0, columnspan=2)

root.mainloop()