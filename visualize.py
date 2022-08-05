import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_path(X, Y, Z):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot3D(X, Y, Z, 'blue')
    ax.scatter3D(X, Y, Z, c=Z, cmap='Greens', s=5)
    ax.scatter(0, 0, 0, c='red', marker='D', s=500, label='target')
    ax.scatter(X[0], Y[0], Z[0], c='yellow', marker='X', s=400, label='inital position')
    ax.scatter(X[-1], Y[-1], Z[-1], c='orange', marker='D', s=300, label='end position')
    ax.set_zlim([-5000, 5000]) 
    ax.set_xlim([-5000, 5000]) 
    ax.set_ylim([-5000, 5000])

    plt.legend(loc="upper right")
    plt.show()

    print("VISUALIZING RUN") 

