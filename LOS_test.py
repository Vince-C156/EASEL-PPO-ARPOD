import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

"""
ax+by+cz=d
a,b,c,d = 1,2,3,4

x = np.linspace(-1,1,10)
y = np.linspace(-1,1,10)

X,Y = np.meshgrid(x,y)
Z = (d - a*X - b*Y) / c

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z)
"""

#sin(15/2)x+cos(15/2)y+ 1z = z+0

#eq = lambda x, y : (np.sin(theta1 / 2) * x) + (np.cos(theta1 / 2) * y)
#[ [ [x[i,j], y[i,j], eq(x[i,j], y[i,j])] for j in range(x.shape[1]) ] for i in range(x.shape[0]) ]

def f1(x,y,theta1):
    y = np.sin(theta2/2) * x
    y *= -np.cos(theta2/2)
    return y


def f2(x,y,theta1):
    y = np.sin(theta2/2) * x
    y *= np.cos(theta2/2)
    return y

def f3(x,y,theta2):
    #z = ( (np.sin(theta2 / 2) * x) / (np.cos(theta2 / 2)*-1.0) )
    z = np.sin(theta2/2) * x
    z *= -np.cos(theta2/2)
    return z

def f4(x,y, theta2):
    z = np.sin(theta2/2) * x
    z *= np.cos(theta2/2)
    return z

theta1 = 1.52
theta2 = 1.52

x = np.linspace(-4,4,100)
y = np.linspace(-4,4,100)
z = np.linspace(-4,4,100)

X,Y = np.meshgrid(x,y)
X,Z = np.meshgrid(x,z)

print(X.shape)
print(X)
Y1 = f1(X,Y,theta1)
Y2 = f2(X,Y,theta1)
Z3 = f3(X,Y,theta2)
Z4 = f4(X,Y,theta2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x, y, z = np.array([[-4,0,0],[0,-4,0],[0,0,-4]])
u, v, w = np.array([[8,0,0],[0,8,0],[0,0,8]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
ax.grid(False)

#plt.xlim([0, 1])
#plt.ylim([0, 1])

surf = ax.plot_surface(X, Y1, Z,color='yellow',alpha=0.5)
surf = ax.plot_surface(X, Y2, Z,color='lightseagreen',alpha=0.5)
surf = ax.plot_surface(X, Y, Z3,color='red',alpha=0.5)
surf = ax.plot_surface(X, Y, Z4,color='blue',alpha=0.5)

plt.show()
