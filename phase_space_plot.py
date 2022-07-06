'''
Put stuff here that shows what the thing does
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, diff, lambdify


# Function that reads file containing initial values (& mass)
def read_file(filename):
    infile = open(filename, 'r')
    lines = infile.readlines()
    length = len(lines)
    init = np.zeros((2, len(lines)))

    for i, line in enumerate(lines):
        init[:,i] = np.array([float(line.split()[0]), float(line.split()[1])])

    infile.close()
    return init, length

# This is too abstract for Hamltions equations, analytical mechanics symbols shpuld be used (q,p), the symbols must not become a barrier to understanding
# Determining Hamilton's equations
def Equations_of_Motion(t, alphabeta):
    m = 1; b = 0.2; l = 1; g = 10; # Parameters that must be specified

    alpha, beta = alphabeta
    alpha_dot = -(b*l/m)*alpha - g*np.sin(beta)
    beta_dot = alpha

    return np.array([alpha_dot, beta_dot])


# Reading input file: Recovering initial values (IVs) & number of IVs
init, length = read_file('input.txt')


# Defining time interval
time_span = [0, 30]
t = np.linspace(0,30,1000)


# Solving the equations of motion for each set of IVs
for i in range(length):
    sol = solve_ivp(Equations_of_Motion, time_span, y0 = init[:,i], t_eval = t)

    # Plotting solutions q & p for each set of IVs
    plt.plot(sol.y[1], sol.y[0], label=rf'$\theta_0 =$ {init[1,i]}, '+
    r'$\dot{\theta}_0 =$'+f' {init[0,i]}')
    plt.legend()

plt.ylabel(r'$\dot{\theta}(t)$', fontsize=15)
plt.xlabel(r'$\theta(t)$', fontsize=15)
plt.title(r'1D pendulum with damping', fontsize=20)
plt.savefig(f'non_ham_phasespace.pdf')
plt.close()
