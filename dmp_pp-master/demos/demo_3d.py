from dmp import dmp_cartesian as dmp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_curve(X,Y):
    """
    Plots a 3D curve from a 3xN array.

    Parameters:
        X (numpy.ndarray): A 2D numpy array with three columns representing x, y, and z coordinates.
    """
    # Create a new figure
    fig = plt.figure()

    # Add an Axes3D instance to the figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color='b', linestyle='-', linewidth=3,label='Parametric curve')
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='r', linestyle='--', linewidth=3, label='DMP curve')

    # Add labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Line Plot')

    # Show legend
    ax.legend()

    # Show the plot
    plt.show()


myK = 10000.0
alpha_s = 4.0

dmp_rescaling = dmp.DMPs_cartesian (n_dmps = 3, n_bfs = 50, dt=0.01, K = myK, rescale = 'rotodilatation', alpha_s = alpha_s, tol = 0.001)





t = np.linspace(0, np.pi, 1000)
# X = np.transpose(np.array([t * t, t * t, t*t]))
X = np.transpose(np.array([t * np.cos(2 * t), t * np.sin(2 * t), t * t]))
dmp_rescaling.imitate_path (x_des = X, t_des = t)
tn=np.pi
# dmp_rescaling.x_0=np.array([1.0,1.0,1.0])
# dmp_rescaling.x_goal=np.array([tn * np.cos(2 * tn+np.pi/2), tn * np.sin(2 * tn+np.pi/2), tn * tn])+np.array([1.0,1.0,1.0])
x_track, _, _, _ = dmp_rescaling.rollout()

plot_3d_curve(X,x_track)

# x_error=X-x_track
# print("x_error",x_error)