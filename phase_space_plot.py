import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, diff, lambdify


# Function that reads file containing initial values (& mass)
def read_file(filename):
    infile = open(filename, 'r')
    lines = infile.readlines()
    length = len(lines) - 1
    m = float(lines[0])
    init = np.zeros((2, len(lines)-1))

    for i, line in enumerate(lines[1:]):
        init[:,i] = np.array([float(line.split()[0]), float(line.split()[1])])

    infile.close()
    return init, length, m


# Function containing the Hamiltonian as function of p's & q's
def Hamiltonian(q, p):
    omega = 1 # Parameters that must be specified
    return p**2/(2*m) + 0.5*m*omega**2*q**2


# Function determining Hamilton's equations by differentiating H
def Hamiltons_Equations(t, qp):
    q, p = symbols('q p') # Declaring q & p as sympy symbols

    H = Hamiltonian(q, p)
    q_dot, p_dot = np.array([diff(H, p), -diff(H, q)]) # Hamilton's equations

    # Converting q_dot & p_dot from sympy expressions to numpy expressions
    q_dot = lambdify(p, q_dot, 'numpy')
    p_dot = lambdify(q, p_dot, 'numpy')

    return np.array([q_dot(qp[1]), p_dot(qp[0])]) #


# Reading input file: Recovering initial values (IVs), number of IVs and mass parameter
init, length, m = read_file('input.txt')

# Defining time interval
time_span = [0, 1000]
t = np.linspace(0,15,1000)


# Solving Hamilton's equations for each set of IVs (and storing them)
solution = np.zeros((length, 2, len(t)))
for i in range(length):
    sol = solve_ivp(Hamiltons_Equations, time_span, y0 = init[:,i], t_eval = t)
    solution[i, 0] = sol.y[0]
    solution[i, 1] = sol.y[1]

    # Plotting solutions q & p for each set of IVs
    plt.plot(sol.y[0], sol.y[1], label=rf'$q_0 =$ {init[0,i]}, $p_0 =$ {init[1,i]}')
    #plt.arrow(sol.y[0, 0], sol.y[1, 0], sol.y[0, 1], sol.y[1, 1]-sol.y[1, 0], shape='full', lw=0, length_includes_head=True, head_width=.05)
    plt.legend()

plt.xlabel(r'$q(t)$ [m]')
plt.ylabel(r'$p(t)$ [kg$\cdot$m/s]')
plt.show()
