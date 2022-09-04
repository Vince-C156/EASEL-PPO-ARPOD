import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def in_LOS(chaserPos):
    """
    Testing if chaser point satisfies pyramid constraint inequality formulation. The region is a pyramid bounded by four planes along the x axis.
    switch x and z positions if desired LOS points up or down.

    Ax <= b

    A : [[sin(theta1/2) cos(theta1/2), 0],
         [sin(theta1/2) -cos(theta1/2), 0],
         [sin(theta2/2) 0, cos(theta2/2)],
         [sin(theta2/2) 0, -cos(theta2/2)],
                                          ]
    x : [[x],
         [y],
         [z],]

    b : [[0],
         [0],
         [0],
         [0]]

    1. np matrix A
    2. pass current state position
    3. return true or false
    """

    theta1 = 0.5
    theta2 = 0.5

    A = np.asmatrix([[np.sin(theta1/2), np.cos(theta1/2), 0], 
          [np.sin(theta1/2), -np.cos(theta1/2), 0],
          [np.sin(theta2/2), 0, np.cos(theta2/2)],
          [np.sin(theta2/2), 0, -np.cos(theta2/2)]])

    x = np.asmatrix([ [chaserPos[0],chaserPos[1],chaserPos[2]] ]).T

    b = np.asmatrix([ [0.0, 0.0, 0.0, 0.0] ]).T

    Ax = A @ x

    values = np.less_equal(Ax, b)


    if np.all(values):
        return True
    else:
        return False


pos = np.asarray([200.0, -3000.0, 1000.0])

print("POS", pos)

in_LOS(pos)