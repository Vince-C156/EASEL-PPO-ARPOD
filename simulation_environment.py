import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

eng = matlab.engine.start_matlab()
R = 1000
#x = matlab.single([1000, -1000, 1000,-10,-10,-10])
x = matlab.double(np.random.randn(6) * 100000)
u0 = matlab.double([1.0, 1.0, 1.0])
Tspan = 10

X, Y, Z = [], [], []

for i in range(100):
    x = eng.ARPOD_Benchmark.nextStep(x, u0, matlab.single(1), matlab.single(1), nargout=1)[0]
    u0 = matlab.double(np.random.randn(3)*100) 
    X.append(x[0])
    Y.append(x[1])
    Z.append(x[2])

#print(X, Y, Z)
fig = plt.figure()
#plt.zlim([-10000, 10000])
ax = plt.axes(projection='3d')
#ax.scatter(0,0,0,c='red',marker='D',s=500)
ax.plot3D(X, Y, Z, 'blue')
ax.scatter3D(X, Y, Z, c=Z, cmap='Greens', s=200)
ax.scatter(0,0,0,c='red',marker='D',s=500, label='target')
ax.scatter(X[0], Y[0], Z[0], c='yellow', marker='X', s=400, label='inital position')
ax.scatter(X[-1], Y[-1], Z[-1], c='orange', marker='D', s=400)
ax.set_zlim([-10000, 10000])
ax.set_xlim([-10000, 10000])
ax.set_ylim([-10000, 10000])
plt.legend(loc="upper right")   
plt.show()
print(x)
print(type(x))
#print(ts)
#print(trajs)
print("Hello world")
