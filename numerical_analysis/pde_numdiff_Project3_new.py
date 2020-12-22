import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pde_numdiff_Project3_methods import solveDiffusion, solveLinAdv, solve_convdif, solve_visBurgers, get_label, \
    get_suptitle, get_RMSnorm


def plot_settings():
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)


def diffusion():
    # Initialize constants

    method = 'Crank-Nicolson'
    iv = 'parabola'
    x0 = 0
    x_end = 1
    t0 = 0
    t_end = 1
    N = 100  # space
    Ms = []

    if method == 'Explicit Euler':
        Ms = [20410, 20390, 20380, 20378]  # time
    elif method == 'Crank-Nicolson':
        Ms = [4, 10, 100, 1000]

    # Prepare for plotting
    fig = plt.figure(figsize=[15, 15])
    fig.suptitle('Solutions to the diffusion equation $u_t = u_{xx}$ for some ' +
                 '\n numbers of time steps $M$, $N$ = 100,' + f' {method}-method', fontsize=40)

    # Solve diffusion equation
    for i in range(len(Ms)):
        u, X, T, courant = solveDiffusion(N, Ms[i], x0, x_end, t0, t_end, method, iv)

        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.plot_surface(X, T, u, cmap='coolwarm', linewidth=0)
        ax.set_title(get_label(i) + f' $M =$ {Ms[i]},' + r' $\mu$ = ' + f'{courant:.5}', fontsize=28, y=1.025)
        ax.set_xlabel('$x$', fontsize=22)
        ax.set_ylabel('$t$', fontsize=22)
        if method == 'Crank-Nicolson':
            ax.view_init(30, 30)

    plt.show()


def advection():
    # Initialize constants
    N = 100
    M = 111
    a = -1
    x0 = 0
    x_end = 1
    t0 = 0
    t_end = 5

    # Solve PDE
    u, X, T, amu = solveLinAdv(N, M, a, x0, x_end, t0, t_end)

    # Plot
    degree = [[30, 120], [90, 90]]

    fig1 = plt.figure(figsize=[15, 7.5])
    fig1.suptitle(get_suptitle(amu, a, N, M), fontsize=30, y=1.025)

    for i in range(2):
        ax = fig1.add_subplot(1, 2, i + 1, projection='3d')
        ax.plot_surface(X, T, u, cmap='coolwarm', linewidth=0)
        ax.set_xlabel('$x$', fontsize=22)
        ax.set_ylabel('$t$', fontsize=22)
        ax.view_init(*degree[i])
        ax.set_title(get_label(i), fontsize=28, y=-0.125)

    plt.show()


def advection_norm():
    # 2.2

    # Initialize constants
    N = 100
    Ms = [100, 111]
    a = 1
    x0 = 0
    x_end = 1
    t0 = 0
    t_end = 5

    # Prepare for plot
    fig2 = plt.figure(figsize=[15, 7.5])
    fig2.suptitle(r'Norm $\|\|U^n\|\|_{\mathrm{RMS}}$ ' +
                  'of numerical solution $U^n$ to \n $u_t + u_x = 0$ at time ' +
                  r'$t_0 \leq t_n \leq t_{\mathrm{end}}$ for different $a \mu$', fontsize=36, y=1.09)

    # Solve PDE for different amu
    for i in range(len(Ms)):
        u, _, _, amu = solveLinAdv(N, Ms[i], a, x0, x_end, t0, t_end)

        norm = get_RMSnorm(u, N)

        ax1 = fig2.add_subplot(1, 2, i + 1)
        ax1.set_title(get_label(i) + r' $a \mu $= ' + f'{amu:.2}', fontsize=28)
        ax1.set_xlabel('$t$', fontsize=22)
        ax1.set_ylabel('$\|\|U^n\|\|$', fontsize=22)
        ax1.plot(np.linspace(t0, t_end, Ms[i] + 1), norm)

    plt.show()


def convect_diffusion1():
    # 3.

    # Initialize constants
    a = 1
    d = [0.1, a / 1000]
    N = 1000
    M = 1000
    x0 = 0
    x_end = 1
    t0 = 0
    t_end = 1
    iv = 'skewed Gaussian'

    # Prepare plot
    fig = plt.figure(figsize=[15, 7.5])
    fig.suptitle('Numerical solution to the convection-diffusion equation \n $u_t + au_x = du_{xx}$ ' +
                 'using trapezoidal rule, $g$ ' + f'{iv}', fontsize=32, y=1.025)

    # Solve covection-diffusion equation
    for i in range(len(d)):
        u, X, T, Pe = solve_convdif(a, d[i], N, M, x0, x_end, t0, t_end, iv)

        # Plot
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.plot_surface(X, T, u, cmap='coolwarm', linewidth=0)
        ax.set_title(get_label(i) + r' $\mathrm{Pe} =$ ' + f'{Pe:.2f},' + r' $\mathrm{mesh \ Pe} =$ ' +
                     f'{Pe / N:.2f}', fontsize=26)
        ax.set_xlabel('$x$', fontsize=22)
        ax.set_ylabel('$t$', fontsize=22)

    plt.show()


def convection_diffusion2():
    # Initialize constants
    a = 1
    d = a / 1000
    N = 10
    M = 1000
    x0 = 0
    x_end = 1
    t0 = 0
    t_end = 1
    iv = 'skewed Gaussian'

    # Prepare plot
    fig = plt.figure(figsize=[7.5, 7.5])
    fig.suptitle('Numerical solution to the convection-diffusion equation \n $u_t + au_x = du_{xx}$ ' +
                 'using trapezoidal rule, $g$ ' + f'{iv}', fontsize=28, y=1.025)

    # Solve covection-diffusion equation

    u, X, T, Pe = solve_convdif(a, d, N, M, x0, x_end, t0, t_end, iv)

    # Plot
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, T, u, cmap='coolwarm', linewidth=0)
    ax.set_title(get_label(0) + r' $\mathrm{Pe} =$ ' + f'{Pe:.2f},' + r' $\mathrm{mesh \ Pe} =$ ' +
                 f'{Pe / N:.2f}', fontsize=26)
    ax.set_xlabel('$x$', fontsize=22)
    ax.set_ylabel('$t$', fontsize=22)


def burgers():
    # 4.

    # Initialize constants
    d = 0.01
    N = 300
    M = 1000
    x0 = 0
    x_end = 1
    t0 = 0
    t_end = 1
    iv = 'skewed Gaussian'

    # Solve equation
    u, X, T = solve_visBurgers(d, N, M, x0, x_end, t0, t_end, iv)

    # Plot
    degree = [[30, -50], [30, 40], [90, -90]]

    fig1 = plt.figure(figsize=[22.5, 7.5])
    fig1.suptitle('Solution to (nonlinear) viscous Burgers\' equation ' +
                  f' $u_t + uu_x = {d if d != 1 else str()}$' +
                  '$u_{xx}$, \n using Lax-Wendroff scheme and trapezoidal rule', fontsize=38, y=1.005)

    for i in range(3):
        ax = fig1.add_subplot(1, 3, i + 1, projection='3d')
        ax.plot_surface(X, T, u, cmap='coolwarm', linewidth=0)
        ax.set_title(get_label(i), fontsize=28, y=-0.125)
        ax.set_xlabel('$x$', fontsize=22)
        ax.set_ylabel('$t$', fontsize=22)
        ax.view_init(*degree[i])

    plt.show()


if __name__ == '__main__':
    plot_settings()
    diffusion()
    advection()
    advection_norm()
    convect_diffusion1()
    convection_diffusion2()
    burgers()
