import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



class pyramid():

    def __init__(self, theta1, theta2, ax):
        self.theta1 = theta1
        self.theta2 = theta2
        self.ax = ax

        self.A = np.asmatrix([[np.sin(self.theta1/2), np.cos(self.theta1/2), 0], 
            [np.sin(self.theta1/2), -np.cos(self.theta1/2), 0],
            [np.sin(self.theta2/2), 0, np.cos(self.theta2/2)],
            [np.sin(self.theta2/2), 0, -np.cos(self.theta2/2)]])

        self.b = np.asmatrix([ [0.0, 0.0, 0.0, 0.0] ]).T

        x_span = np.linspace(-10,10,100)
        y_span = np.linspace(-10,10,100)
        z_span = np.linspace(-10,10,100)

        constrained_span = [[x_span, y_span, z_span] for n in range(len(x_span)) if self.is_constrained([x_span[n], y_span[n], z_span[n]])]

        self.X,self.Y = np.meshgrid(x_span,y_span)
        self.X,self.Z = np.meshgrid(x_span,z_span)

        self.Y1 = np.array([])
        self.Y2 = np.array([])
        self.Z3 = np.array([])
        self.Z4 = np.array([])

    def plane1(self):
        self.Y1 = np.sin(self.theta1/2) * self.X
        self.Y1 *= -np.cos(self.theta1/2)


    def plane2(self):
        self.Y2 = np.sin(self.theta1/2) * self.X
        self.Y2 *= np.cos(self.theta1/2)

    def plane3(self):
        #z = ( (np.sin(theta2 / 2) * x) / (np.cos(theta2 / 2)*-1.0) )

        self.Z3 = np.sin(self.theta2/2) * self.X
        self.Z3 *= -np.cos(self.theta2/2)

    def plane4(self):
        self.Z4 = np.sin(self.theta2/2) * self.X
        self.Z4 *= np.cos(self.theta2/2)

    def is_constrained(self, Point):
        """
        Point = [xpos ypos zpos]
        """
        x = np.asmatrix([ [Point[0], Point[1], Point[2]] ]).T

        Ax = self.A @ x
        values = np.less_equal(Ax, self.b)

        if np.all(values):
            return True
        else:
            return False

    def plot_LOS(self):

        ax = self.ax

        self.plane1()
        self.plane2()
        self.plane3()
        self.plane4()

        x, y, z = np.array([[-10,0,0],[0,-10,0],[0,0,-10]])
        u, v, w = np.array([[10,0,0],[0,10,0],[0,0,10]])
        ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
        
        """
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        """
        #ax.grid(False)

        #(M, N) for ‘ij’ indexing

        points = []

        p1_X = []
        p1_Y1 = []

        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                if is_constrained([self.X[i,j],self.Y1[i,j],self.Z[i,j]]):
                    points.append(self.X[i,j], self.Y1[i,j], self.Z[i,j])

        #[(i,j) for i in range(self.X.shape[0]) for j in range(self.X.shape[0]) if is_constrained([self.X[i,j],self.Y1[i,j],self.Z[i,j]])]
        # treat xv[i,j], yv[i,j]

        self.X self.Y1 self.Z

        surf = ax.plot_surface(self.X, self.Y1, self.Z,color='yellow',alpha=0.5)
        surf = ax.plot_surface(self.X, self.Y2, self.Z,color='lightseagreen',alpha=0.5)
        surf = ax.plot_surface(self.X, self.Y, self.Z3,color='red',alpha=0.5)
        surf = ax.plot_surface(self.X, self.Y, self.Z4,color='blue',alpha=0.5)

