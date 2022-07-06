import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, diff, lambdify, cos
import matplotlib.ticker as mticker


# Function that reads file containing initial values
def read_file(filename):
    infile = open(filename, 'r')
    lines = infile.readlines()
    length = len(lines)
    init = np.zeros((2, len(lines)))

    for i, line in enumerate(lines):
        init[:,i] = np.array([float(line.split()[0]), float(line.split()[1])])

    infile.close()
    return init, length


# Defining the Hamiltonian of the system
def Hamiltonian(q, p, arg):
    m = 1; l = 1; g = 10; # Parameters that must be specified
    if arg == 'sympy':
        return p**2/(2*m*l**2) + m*g*l*(1-cos(q))
    elif arg == 'numpy':
        return p**2/(2*m*l**2) + m*g*l*(1-np.cos(q))


# Determining Hamilton's equations
def Equations_of_Motion(t, qp):
    q, p = symbols('q p') # Declaring q & p as sympy symbols

    H = Hamiltonian(q, p, 'sympy')
    q_dot, p_dot = np.array([diff(H, p), -diff(H, q)]) # Hamilton's equations

    # Converting q_dot & p_dot from sympy expressions to numpy functions
    q_dot = lambdify(p, q_dot, 'numpy')
    p_dot = lambdify(q, p_dot, 'numpy')

    return np.array([q_dot(qp[1]), p_dot(qp[0])])


# Adding arrow to plot
def add_arrow(line, position=None, direction='right', size=15, color='k'):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()

    start_ind = 300 #np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


# Reading input file: Recovering initial values (IVs) & number of IVs
init, length = read_file('input2.txt')


# Defining time interval
time_span = [0, 15]
t = np.linspace(0,15,1000)


# Solving the equations of motion for each set of IVs
for i in range(length):
    sol = solve_ivp(Equations_of_Motion, time_span, y0 = init[:,i], t_eval = t, method='DOP853')

    # Plotting solutions q & p for each set of IVs
    line = plt.plot(sol.y[0]/np.pi, sol.y[1], color='k')[0]
    #label=rf'$\theta_0 =$ {init[0,i]}, '+ r'$p_0 =$'+f' {init[1,i]}'
    #plt.legend()
    add_arrow(line)
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter(r'%d $\pi$'))
plt.ylabel(r'$p(t)$', fontsize=15)
plt.xlabel(r'$\theta(t)$', fontsize=15)
plt.title(r'1D pendulum', fontsize=20)
plt.savefig(f'forelesningseksempel.pdf')
plt.close()
