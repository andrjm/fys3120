# Simple 1D pendulum example

import numpy as np
import matplotlib.pyplot as plt


# Defining the Hamiltonian of the system
def Hamiltonian(q, p):
    m = 1; g = 10; l = 1 # mass, gravitational acceleration & length of pendulum
    return p**2/(2*m*l**2) + m*g*l*(1-np.cos(q))


# Function determining Hamilton's equations by differentiating H
def Hamiltons_Equations(qp):
    m = 1; g = 10; l = 1
    q, p = qp
    q_dot = p/m/l**2
    p_dot = -m*g*l*np.sin(q)
    return np.array([q_dot, p_dot])


# Defining linspaces for theta coordinate values & conjugate momentum values
q = np.linspace(-np.pi/2, np.pi/2, 101)
p = np.linspace(-10, 10, 101)


# Creating a meshgrid of the theta & conjugate momentum values
X, Y = np.meshgrid(q, p)
Z = Hamiltonian(X, Y)


# Computing derivatives
u = np.empty_like(X)
v = np.empty_like(Y)
u, v = Hamiltons_Equations(np.array([X, Y]))


# Creating the plot
fig, ax = plt.subplots()
strm = ax.streamplot(X, Y, u, v, color=Z, linewidth=1, cmap='viridis')
fig.colorbar(strm.lines, label=r'$H(\theta, p_{\theta})$')
plt.xlabel(r'$\theta(t)$', fontsize=15)
plt.ylabel(r'$p_{\theta}(t)$', fontsize=15)
plt.title(r'Simple 1D pendulum', fontsize=20)
plt.tight_layout()
fig.savefig(f'streamplot.pdf')
plt.close()
