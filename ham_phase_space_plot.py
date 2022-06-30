# Simple 1D pendulum example

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, cos


# Defining the Hamiltonian of the system
def Hamiltonian(q, p, arg):
    m = 1; g = 10; l = 1 # mass, gravitational acceleration & length of pendulum
    if arg == 'sympy':
        return p**2/(2*m*l**2) + m*g*l*(1-cos(q))
    elif arg == 'numpy':
        return p**2/(2*m*l**2) + m*g*l*(1-np.cos(q))


# Function determining Hamilton's equations by differentiating H
def Hamiltons_Equations(qp):
    q, p = symbols('q p') # Declaring q & p as sympy symbols

    H = Hamiltonian(q, p, 'sympy')
    q_dot, p_dot = np.array([diff(H, p), -diff(H, q)]) # Hamilton's equations

    # Converting q_dot & p_dot from sympy expressions to numpy functions
    q_dot = lambdify(p, q_dot, 'numpy')
    p_dot = lambdify(q, p_dot, 'numpy')

    return np.array([q_dot(qp[1]), p_dot(qp[0])])


# Defining linspaces for theta coordinate values & conjugate momentum values
q = np.linspace(-np.pi/2, np.pi/2, 101)
p = np.linspace(-10, 10, 101)


# Creating a meshgrid of the theta & conjugate momentum values
X, Y = np.meshgrid(q, p)
Z = Hamiltonian(X, Y, 'numpy')

# Computing derivatives
u = np.empty_like(X)
v = np.empty_like(Y)


u, v = Hamiltons_Equations(np.array([X, Y]))


# Making a stream plot of the Hamiltonian
fig, ax = plt.subplots()
strm = ax.streamplot(X, Y, u, v, color=Z, linewidth=1, cmap='viridis')
fig.colorbar(strm.lines, label=r'$H(\theta, p_{\theta})$')
plt.xlabel(r'$\theta(t)$', fontsize=15)
plt.ylabel(r'$p_{\theta}(t)$', fontsize=15)
plt.title(r'Simple 1D pendulum', fontsize=20)
plt.tight_layout()
fig.savefig(f'streamplot.pdf')
plt.close()
