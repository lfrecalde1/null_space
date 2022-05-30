import numpy as np


def distance(h, obstacles):
    # Positions of the system
    xc = h[0]
    yc = h[1]
    # obstacles positions
    xo = obstacles[0]
    yo = obstacles[1]
    # Constants values, we need these values to define the function positive
    ax = 2
    ay = 2
    n = 4
    # Auxiliar values
    aux_x = ((xc-xo)**n)/ax
    aux_y = ((yc-yo)**n)/ay
    value = (-aux_x - aux_y)
    return value


def potential_field(h, obstacles):
    # Auxiliar values
    j = obstacles.shape[1]
    # Potential fiel vector
    V = []
    for k in range(0, j):
        aux = np.exp(distance(h, obstacles[:, k]))
        V.append(aux)
    V_vector = np.array(V)
    return V_vector
