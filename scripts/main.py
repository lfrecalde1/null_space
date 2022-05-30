#!/usr/bin/env python3
import numpy as np
from fancy_plots import fancy_plots_1
from system_functions import f_d
from system_functions import kinematic_controller
import matplotlib.pyplot as plt


def main():
    # Time paramters
    ts = 0.1
    t_final = 20
    t = np.arange(0, t_final+ts, ts)

    # System values
    a = 0.1
    L = [a]

    # Initial states of the system
    x = 0.0
    y = 0.0
    yaw = 0*(np.pi/180)

    # Direct Kinematics
    x = x + a*np.cos(yaw)
    y = y + a*np.sin(yaw)

    # Vector of the initial states of the system
    h = np.zeros((3, t.shape[0]+1), dtype=np.float32)
    h[0, 0] = x
    h[1, 0] = y
    h[2, 0] = yaw

    # Control values vector
    u_control = np.zeros((2, t.shape[0]), dtype=np.float32)

    # Desired trajectory refernce
    hd = np.zeros((2, t.shape[0]), dtype=np.float32)
    hd[0, :] = 1*np.sin(0.5*t)
    hd[1, :] = 1*np.cos(0.5*t)
    # Desired trajectory dot
    hdp = np.zeros((2, t.shape[0]), dtype=np.float32)
    hdp[0, :] = 1*0.5*np.cos(0.5*t)
    hdp[1, :] = -1*0.5*np.sin(0.5*t)

    # Error vector definition
    he = np.zeros((2, t.shape[0]), dtype=np.float32)

    # Controler gains
    k1 = 1
    k2 = 0.5

    # Sytem simulation
    for k in range(0, t.shape[0]):
        # Error vector defitnion
        he[:, k] = hd[:, k] - h[0:2, k]
        # Control Law
        u_control[:, k] = kinematic_controller(h[:, k], hd[:, k], hdp[:, k],
                                               k1, k2, L)
        h[:, k+1] = f_d(h[:, k], u_control[:, k], L, ts)

    # System plot
    fig1, ax1 = fancy_plots_1()
    # Axis definition necesary to fancy plots
    ax1.set_xlim((t[0], t[-1]))
    control_u, = ax1.plot(t[0:u_control.shape[1]], u_control[0, :],
                          color='#00429d', lw=2, ls="-")
    control_w, = ax1.plot(t[0:u_control.shape[1]], u_control[1, :],
                          color='#97a800', lw=2, ls="-")
    ax1.set_ylabel(r"$[m/s],[rad/s]$", rotation='vertical')
    ax1.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
    ax1.legend([control_u, control_w],
               [r'$\mu$', r'$\omega$'],
               loc="best",
               frameon=True, fancybox=True, shadow=False, ncol=2,
               borderpad=0.5, labelspacing=0.5, handlelength=3,
               handletextpad=0.1, borderaxespad=0.3, columnspacing=2)
    ax1.grid(color='#949494', linestyle='-.', linewidth=0.5)
    fig1.savefig("control_values.eps", resolution=300)
    fig1.savefig("control_values.png", resolution=300)
    plt.show()
    fig2, ax2 = fancy_plots_1()
    # Axis definition necesary to fancy plots
    state_x_y, = ax2.plot(h[0, :], h[1, :],
                          color='#00429d', lw=2, ls="-")
    state_x_y_d, = ax2.plot(hd[0, :], hd[1, :],
                            color='#00429d', lw=2, ls="--")
    ax2.set_ylabel(r"$[m]$", rotation='vertical')
    ax2.set_xlabel(r"$[m]$", labelpad=5)
    ax2.legend([state_x_y, state_x_y_d],
               [r'$h$', r'$h_d$'],
               loc=1,
               frameon=True, fancybox=True, shadow=False, ncol=2,
               borderpad=0.5, labelspacing=0.5, handlelength=3,
               handletextpad=0.1, borderaxespad=0.3, columnspacing=2)
    ax2.grid(color='#949494', linestyle='-.', linewidth=0.5)
    fig2.savefig("system_states.eps", resolution=300)
    fig2.savefig("system_states.png", resolution=300)
    fig2
    plt.show()
    # System plot
    fig3, ax3 = fancy_plots_1()
    # Axis definition necesary to fancy plots
    ax1.set_xlim((t[0], t[-1]))
    error_x, = ax3.plot(t[0:he.shape[1]], he[0, :],
                        color='#00429d', lw=2, ls="-")
    error_y, = ax3.plot(t[0:he.shape[1]], he[1, :],
                        color='#97a800', lw=2, ls="-")
    ax3.set_ylabel(r"$[m]$", rotation='vertical')
    ax3.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
    ax3.legend([control_u, control_w],
               [r'$\tilde{h_x}$', r'$\tilde{h_y}$'],
               loc="best",
               frameon=True, fancybox=True, shadow=False, ncol=2,
               borderpad=0.5, labelspacing=0.5, handlelength=3,
               handletextpad=0.1, borderaxespad=0.3, columnspacing=2)
    ax3.grid(color='#949494', linestyle='-.', linewidth=0.5)
    fig3.savefig("control_error.eps", resolution=300)
    fig3.savefig("control_error.png", resolution=300)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Pres Ctrl-c to end the statement")
