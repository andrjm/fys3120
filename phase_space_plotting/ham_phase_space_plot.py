# Simple 1D pendulum example

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, cos


# Defining the Hamiltonian of the system
def Hamiltonian(q, p, arg):
    m = 1; g = 10; l = 1 # Mass, gravitational acceleration & length of pendulum
    if arg == 'sympy':
        return p**2/(2*m*l**2) + m*g*l*(1-cos(q))
    elif arg == 'numpy':
        return p**2/(2*m*l**2) + m*g*l*(1-np.cos(q))


# Determining Hamilton's equations by differentiating H using SymPy
def Hamiltons_Equations(qp):
    q, p = symbols('q p') # Declaring q & p as SymPy symbols

    H = Hamiltonian(q, p, 'sympy')
    q_dot, p_dot = np.array([diff(H, p), -diff(H, q)]) # Hamilton's equations

    # Converting q_dot & p_dot from SymPy expressions to NumPy functions
    q_dot = lambdify(p, q_dot, 'numpy')
    p_dot = lambdify(q, p_dot, 'numpy')

    return np.array([q_dot(qp[1]), p_dot(qp[0])])


# Defining linspaces for coordinate values & conjugate momentum values
q = np.linspace(-2*np.pi, 2*np.pi, 101)
p = np.linspace(-10, 10, 101)


# Creating a meshgrid of the coordinate & conjugate momentum values
X, Y = np.meshgrid(q, p)
Z = Hamiltonian(X, Y, 'numpy')

# Computing derivatives
u = np.empty_like(X)
v = np.empty_like(Y)
u, v = Hamiltons_Equations(np.array([X, Y]))


# Making a stream plot of the Hamiltonian
fig, ax = plt.subplots()
strm = ax.streamplot(X, Y, u, v, color=Z, linewidth=1, cmap='viridis')
fig.colorbar(strm.lines, label=r'$H(q, p)$')
plt.xlabel(r'$q(t)$', fontsize=10)
plt.ylabel(r'$p(t)$', fontsize=10)
plt.title(r'Simple 1D pendulum', fontsize=15)
plt.tight_layout()
fig.savefig(f'ham_phasespace.pdf')
plt.close()
