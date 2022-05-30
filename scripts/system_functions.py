import numpy as np


def f_system(h, u, L):
    # Jacobian of the system
    J = jacobian_system(h, L)

    # System f
    hp = J@u
    return hp


def f_d(h, u, L, ts):
    k1 = f_system(h, u, L)
    k2 = f_system(h+(ts/2)*k1, u, L)
    k3 = f_system(h+(ts/2)*k2, u, L)
    k4 = f_system(h+(ts)*k3, u, L)
    h = h + (ts/6)*(k1+2*k2+2*k3+k4)
    return h


def jacobian_system(h, L):
    n_states = 3
    u_states = 2

    # Parameters definition
    a = L[0]

    # Internal states of the system
    yaw = h[2]

    # Jacobian of the system
    J = np.zeros((n_states, u_states), dtype=np.float32)
    J[0, 0] = np.cos(yaw)
    J[0, 1] = -a*np.sin(yaw)
    J[1, 0] = np.sin(yaw)
    J[1, 1] = a*np.cos(yaw)
    J[2, 0] = 0
    J[2, 1] = 1
    return J


def jacobian_controller(h, L):
    n_states = 2
    u_states = 2

    # Parameters definition
    a = L[0]

    # Internal states of the system
    yaw = h[2]

    # Jacobian of the system
    J = np.zeros((n_states, u_states), dtype=np.float32)
    J[0, 0] = np.cos(yaw)
    J[0, 1] = -a*np.sin(yaw)
    J[1, 0] = np.sin(yaw)
    J[1, 1] = a*np.cos(yaw)
    return J


def kinematic_controller(h, hd, hdp, k1, k2, L):
    # Error definiton
    h_xy = h[0:2]
    he = hd - h_xy
    # Controller jacobian_controller
    J = jacobian_controller(h, L)
    # Gain matrices
    K1 = k1*np.eye(2)
    K2 = k2*np.eye(2)
    # Inverse J_1
    J_1 = np.linalg.inv(J)
    K2_1 = np.linalg.inv(K2)
    u = J_1@(hdp+K2@np.tanh(K2_1@K1@he))
    return u


def dynamic_values():
    x = np.zeros((7, 1), dtype=np.float32)
    x[0, 0] = 0.189663950707076
    x[1, 0] = 0.164747439046170
    x[2, 0] = 0.985794879357231
    x[3, 0] = 6.499003771150739e-05
    x[4, 0] = 3.552425533589924e-05
    x[5, 0] = 0.014529342845266
    x[6, 0] = 0.914880354023648
    return x


def f_dynamic(v, vref, chi):
    # Parameters definition
    x = chi
    # Velocities definiton
    w = v[1]
    # Inertial matrix definition
    M = np.zeros((2, 2), dtype=np.float32)
    M[0, 0] = x[0, 0]
    M[0, 1] = 0.0
    M[1, 0] = 0.0
    M[1, 1] = x[1, 0]
    # Centrifugal Forces
    C = np.zeros((2, 2), dtype=np.float32)
    C[0, 0] = x[2, 0]
    C[0, 1] = x[3, 0] + x[4, 0]*w
    C[1, 0] = x[5, 0]*w
    C[1, 1] = x[6, 0]
    # Matrix invertion section
    M_1 = np.linalg.inv(M)
    vp = -M_1@C@v+M_1@vref
    return vp


def f_dynamic_d(v, vref, chi, ts):
    k1 = f_dynamic(v, vref, chi)
    k2 = f_dynamic(v+(ts/2)*k1, vref, chi)
    k3 = f_dynamic(v+(ts/2)*k2, vref, chi)
    k4 = f_dynamic(v+(ts)*k3, vref, chi)
    v = v + (ts/6)*(k1+2*k2+2*k3+k4)
    return v
