import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pyramid_LOS import pyramid


def plot_target(ax, center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, target_Z=np.meshgrid(theta, z)
    target_X = radius*np.cos(theta_grid) + center_x
    target_Y = radius*np.sin(theta_grid) + center_y
    
    ax.plot_surface(target_X, target_Y, target_Z, color='red', linewidth=0.5,
                       rstride=20, cstride=10, alpha=0.5)

    ax.plot_surface(target_X, -target_Y, target_Z, color='red', linewidth=0.5,
                   rstride=20, cstride=10, alpha=0.5)

def plot_obstacles(ax, info_items, step):
    upper_lim = np.asmatrix([ [15.0, 15.0, 15.0] ]).T
    lower_lim = np.asmatrix([ [-15.0, -15.0, -15.0] ]).T
    for id, value in info_items:
        obstacle_data_array = np.array(value)
        positions = obstacle_data_array[:,0]
        
        print("OBSTACLE POSITIONS")
        print(positions)
        axes = obstacle_data_array[:,1]
        
        print("ALL Xc")
        Xc, Yc, Zc = positions[:step+1,0], positions[:step+1,1], positions[:step+1,2]
        
        end_xc, end_yc, end_zc = Xc[step], Yc[step], Zc[step]
        center = np.asarray([end_xc, end_yc, end_zc]).reshape(3,1)
        end_axes = axes[-1]
        
        a, b, c = end_axes[0], end_axes[1], end_axes[2]
        
        P = np.asarray([[a, 0, 0], [0, b, 0], [0, 0, c]], dtype=np.float64).reshape((3,3))
        U, s, rotation = np.linalg.svd(P)
        rx, ry, rz = 1/np.sqrt(s)
        
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        
        XC = np.outer(np.cos(u), np.sin(v))
        YC = np.outer(np.sin(u), np.sin(v))
        ZC = np.outer(np.ones_like(u), np.cos(v))
        
        circle = np.stack((XC, YC, ZC), 0)     #.reshape(3,-1) + center
        xTPinv = P @ circle.reshape(3,-1) 
        xTPinv += center
        ellipsoid = xTPinv.reshape(3, *XC.shape)
        print(ellipsoid.shape)
        """
        for i in range(len(XC)):
            for j in range(len(XC)):
                [XC[i,j],YC[i,j],ZC[i,j]] = np.dot([XC[i,j],YC[i,j],ZC[i,j]], rotation) + center
        
        """

        values = np.less(center, upper_lim) + np.less(center, lower_lim)

        if np.any(values):
            surf = ax.plot_surface(*ellipsoid, rstride=1, cstride=1, color='g', alpha=1)
        
        ax.plot3D(Xc, Yc, Zc, 'red', linewidth=0.5)

        #ax.scatter3D(Xc, Yc, Zc, c=Zc, cmap='Reds', s=5)
        
        
        max_radius = max(rx, ry, rz)
        for axis in 'xyz':
            getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
            
def plot_path(fig, ax, X, Y, Z, info, theta1, theta2, step):
    """
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111, projection='3d')
    """

    #ax, center_x,center_y,radius,height_z

    plot_obstacles(ax, info.items(),step)


    """

    region = pyramid(theta1, theta2, ax)

    region.plot_LOS()
        
    """
    ax.plot3D(X[:step], Y[:step], Z[:step], 'blue')
    ax.scatter(X[0], Y[0], Z[0], c='yellow', marker='X', s=25, label='inital position')
    #ax.scatter(X[step], Y[step], Z[step], c='orange', marker='D', s=50, label='end position')
    ax.scatter(X[-1], Y[-1], Z[-1], c='orange', marker='D', s=50, label='end position')
    ax.set_zlim([-15, 15]) 
    ax.set_xlim([-15, 15]) 
    ax.set_ylim([-15, 15])

    plt.draw()
    plt.pause(0.001)

    print("VISUALIZING RUN")

#f'{5:005}

