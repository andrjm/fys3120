# Simple 1D pendulum example

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, lambdify, cos
import be150
import bokeh.io


# Defining the Hamiltonian of the system
def Hamiltonian(theta, p, arg):
    m = 1; g = 10; l = 1 # mass, gravitational acceleration & length of pendulum
    if arg == 'sympy':
        return p**2/(2*m*l**2) + m*g*l*(1-cos(theta))
    elif arg == 'numpy':
        return p**2/(2*m*l**2) + m*g*l*(1-np.cos(theta))


# Defining linspaces for theta coordinate values & conjugate momentum values
theta = np.linspace(np.pi, 0, 101)
p = np.linspace(0, 10, 101)


# Creating a meshgrid of the theta & conjugate momentum values
X, Y = np.meshgrid(theta, p)
Z = Hamiltonian(X, Y, 'numpy')


# Function determining Hamilton's equations by differentiating H
def Hamiltons_Equations(qp):
    q, p = symbols('q p') # Declaring q & p as sympy symbols

    H = Hamiltonian(q, p, 'sympy')
    q_dot, p_dot = np.array([diff(H, p), -diff(H, q)]) # Hamilton's equations

    # Converting q_dot & p_dot from sympy expressions to numpy functions
    q_dot = lambdify(p, q_dot, 'numpy')
    p_dot = lambdify(q, p_dot, 'numpy')

    return np.array([q_dot(qp[1]), p_dot(qp[0])]) #



def flow_field():

    # Compute derivatives
    u = np.empty_like(X)
    v = np.empty_like(Y)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            u[i,j], v[i,j] = Hamiltons_Equations(np.array([X[i,j], Y[i,j]]))

    # Make stream plot
    return plt.streamplot(theta, p, u, v)




plot = flow_field()
#bokeh.io.show(plot)

# Producing the contour plot of the Hamiltonian
plt.contour(X, Y, Z, cmap='cividis', levels=80) #, shading='gouraud')  # gouraud
plt.colorbar(label=r'$H(\theta, p_{\theta})$')
plt.xlabel(r'$\theta(t)$', fontsize=15)
plt.ylabel(r'$p_{\theta}(t)$', fontsize=15)
plt.title(r'Simple 1D pendulum', fontsize=20)
plt.show()
