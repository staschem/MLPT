import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================================
# Load experimental data
# ================================

file = 'filename.csv'

print(f"Reading file: {file}")
data = pd.read_csv(file, sep=',', header=None)

# Extract time and concentrations
time = data.values[:, 0]                          # Time column
concentrations = data.values[:, 1:].T             # Transpose: rows -> species

# ================================
# Define the kinetic model
# ================================

def get_concs(t: NDArray, params: Parameters, start_concs: NDArray) -> NDArray:
    """
    Solve ODEs for A ↔ B → C system and return concentrations over time.

    Parameters:
        t (array): Time points to evaluate
        params (Parameters): lmfit Parameters object with rate constants
        start_concs (array): Initial concentrations [A0, B0, C0]

    Returns:
        np.ndarray: Concentrations of A, B, C at each time point
    """
    c0 = sum(start_concs)  # Total concentration (constant)
    B_0, C_0 = start_concs[1], start_concs[2]

    # Extract rate constants
    k1 = params['k1'].value         # A -> B
    k1_inv = params['k1_inv'].value # B -> A
    k2 = params['k2'].value         # B -> C

    # Define differential equations: dB/dt and dC/dt
    def derivatives(t, concs):
        B, C = concs
        A = c0 - B - C              # From mass balance
        dB = k1 * A - k1_inv * B - k2 * B
        dC = k2 * B
        return [dB, dC]

    # Solve ODEs
    solution = solve_ivp(
        derivatives, 
        (t[0], t[-1]), 
        y0=[B_0, C_0], 
        t_eval=t, 
        method="BDF"
    )

    if not solution.success:
        raise RuntimeError("ODE solver failed to converge.")

    # Retrieve concentrations
    B, C = solution.y
    A = c0 - B - C  # Calculate A from mass balance

    return np.array([A, B, C])


def residual(params: Parameters, time: NDArray, concentrations: NDArray) -> NDArray:
    """
    Compute residuals between model and experimental data.

    Parameters:
        params (Parameters): lmfit Parameters object
        time (array): Time points
        concentrations (array): Experimental concentrations

    Returns:
        np.ndarray: Flattened array of residuals
    """
    start_concs = concentrations[:, 0]  # Initial concentrations at t=0
    model_concs = get_concs(time, params, start_concs)
    return (concentrations - model_concs).ravel()


# ================================
# Parameter fitting
# ================================

# Initialize model parameters with bounds
params = Parameters()
params.add('k1', value=5e-4, min=1e-8, max=1)
params.add('k1_inv', value=4e-4, min=1e-8, max=1)
params.add('k2', value=1e-4, min=1e-8, max=1)

# Perform least-squares optimization
result = minimize(residual, params, args=(time, concentrations), method='ampgo')

# Print fit report
print(report_fit(result))


# ================================
# Plot experimental data and fit
# ================================

# plot the data and the fit
fig = plt.figure()

alpha = 0.7
markersize = 7
custom_palette = sns.color_palette("colorblind", 10)

time = time/3600
plt.plot(time, concentrations[0]*1000, linestyle='None', color = custom_palette[0], label="Ru(H)", alpha=alpha, markersize=markersize, marker='o')
plt.plot(time, concentrations[1]*1000, linestyle='None', color = custom_palette[4], label="Ru(0)", alpha=alpha, markersize=markersize, marker='o')
plt.plot(time, concentrations[2]*1000, linestyle='None', color = custom_palette[2], label="cyclo-Ru(H)", alpha=alpha, markersize=markersize, marker='o')
plt.plot(time, np.sum(concentrations, axis=0)*1000, linestyle='None', color = custom_palette[7], label="Mass balance", alpha=alpha, markersize=markersize, marker='o')


time = np.linspace(time[0]*3600, time[-1]*3600, 500)
calculated = get_concs(time, result.params,[concentrations[0][0], concentrations[1][0], concentrations[2][0]])

alpha = 0.4
linewidth = 3

# Rescale time to hours and concentrations to mM
time = time/3600
plt.plot(time, calculated[0]*1000, color = custom_palette[0], alpha=alpha, linewidth=linewidth, label="Ru(H) fit")
plt.plot(time, calculated[1]*1000, color = custom_palette[4], alpha=alpha, linewidth=linewidth, label="Ru(0) fit")
plt.plot(time, calculated[2]*1000, color = custom_palette[2], alpha=alpha, linewidth=linewidth, label="cyclo-Ru(H) fit")


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']

plt.legend().remove()

plt.xlabel("Time (hours)", fontsize=18)
plt.ylabel("C (mM)", fontsize=18)


plt.gca().minorticks_on()
plt.tick_params(axis='x', which='major', length=7, direction='in', labelsize=12, colors='grey')
plt.tick_params(axis='y', which='major', length=7, direction='in', labelsize=12, colors='grey')
plt.tick_params(axis='x', which='minor', length=4, direction='in', colors='grey')
plt.tick_params(axis='y', which='minor', length=4, direction='in', colors='grey')

plt.xticks([0, 3, 6, 9, 12], color = 'black', fontsize=18)  
plt.yticks([0, 3, 6, 9, 12], color = 'black', fontsize=18)

plt.tick_params(width=1)

for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(1.5)
    plt.gca().spines[axis].set_color('grey')

plt.gcf().subplots_adjust(bottom=0.15)
fig.savefig(file.split('.')[0] + ".png", format='png', dpi=1000)


plt.show()