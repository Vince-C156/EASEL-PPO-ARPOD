import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pyramid_LOS import pyramid

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x, y, z = np.array([[-10,0,0],[0,-10,0],[0,0,-10]])
u, v, w = np.array([[20,0,0],[0,20,0],[0,0,20]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
ax.grid(False)

region = pyramid(1.52, 1.52, ax)

region.plot_LOS()

#plt.xlim([0, 1])
#plt.ylim([0, 1])



plt.show()
