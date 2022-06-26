import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import diff, symbols, lambdify

input1 = input('Please enter m q_0 p_0:')
m = float(input1.split()[0]); q_0 = float(input1.split()[1]); p_0 = float(input1.split()[2])

"""
input2 = raw_input('Specify potential V:') '0.5 * m * omega ** 2 * x ** 2'
for factor in input2.split():
    try float(factor):
        f

    except ValueError:
        if factor == '*':
"""


time_span = [0, 1000]
init = [q_0, p_0]


def V(q):
    omega = 1
    return 0.5*m*omega**2*q**2

def integrand(t, qp):
    q, p = symbols('q p')
    H = p**2/(2*m) + V(q)
    q_dot, p_dot = np.array([diff(H, p), -diff(H, q)])

    q_dot = lambdify(p, q_dot, 'numpy')
    p_dot = lambdify(q, p_dot, 'numpy')

    return np.array([q_dot(qp[1]), p_dot(qp[0])])


t = np.linspace(0,15,1000)
sol = solve_ivp(integrand, time_span, y0 = init, t_eval = t)


plt.plot(t, sol.y[0], label='q')
plt.plot(t, sol.y[1], label='p')
plt.legend()
plt.show()
