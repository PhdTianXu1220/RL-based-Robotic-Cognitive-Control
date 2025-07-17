import sys

# Path to the directory containing the package
# package_path = '/home/tianxu/Documents/DMP-python/Dual_Arm_New'
#
# # Add this path to sys.path
# if package_path not in sys.path:
#     sys.path.append(package_path)

from Robot_Sim.robots.kinova_robotiq_new import Kinova_Robotiq
# from Robot_Sim.robots.ur5_robotiq import UR5_Robotiq

import pybullet as pb
import pybullet_data
import time
import pdb
import json
import numpy as np
import matplotlib.pyplot as plt

from dmp import dmp_cartesian as dmp
import numpy as np
import matplotlib.pyplot as plt


# Define the rotation (90 degrees around Z-axis)
rotation_euler = [0, 0, 3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion = pb.getQuaternionFromEuler(rotation_euler)

# Define the rotation (90 degrees around Z-axis)
rotation_euler2 = [0, 0, -3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion2 = pb.getQuaternionFromEuler(rotation_euler2)

bias_kinova=np.array([0.0,0.0,0.022])
bias_ur5e=np.array([0,0,0])
height_bias=np.array([0,0,-0.05])
table_bias=np.array([0.0,0.0,-0.2])
execution_bias = np.array([-0.00109593, - 0.0001968, 0.0])

contact_flag=False
re_grip_flag=False
re_grip_wait_count=0
loose_thres=0.05
step_dist=0.03
step_dist_return=0.03
close_thres=0.03
object_dist_thres=0.1
reach_step=0
close_step=0
initial_step_num=5

max_step =500
grip_open_steps=30
grip_close_steps=20

iter_flag=False
dmp_step_num=10

myK = 10000.0
alpha_s = 4.0

# number of skill
skill_num=3

# create a dictionary for instances
dmps={f"DMP{i}":dmp.DMPs_cartesian (n_dmps = 3, n_bfs = 50, dt=0.01, K = myK, rescale = 'rotodilatation', alpha_s = alpha_s, tol = 0.01) for i in range(1, skill_num+1)}

# dmp_rescaling = dmp.DMPs_cartesian (n_dmps = 3, n_bfs = 50, dt=0.01, K = myK, rescale = 'rotodilatation', alpha_s = alpha_s, tol = 0.001)

def plot_joint_torque(data,savename='torque.png'):
    # Zip is used to unpack and then repack each row into columns
    transposed_data = list(zip(*data))

    # Index array for the x-axis
    index_data = list(range(0, len(data)))

    # Plot each column data
    for i, column in enumerate(transposed_data, start=1):
        plt.plot(index_data, column, label=f'Column {i}', marker='o', linestyle='-')

    # Adding titles and labels
    plt.title('Plot of All Columns Over Index')
    plt.xlabel('Index')
    plt.ylabel('Values')
    # Adding a legend to distinguish the lines
    plt.legend()
    plt.savefig(savename, dpi=300)
    # Display the plot
    plt.show()

# 0: lift up triangle skill
# 1: move skill
# 2: lift and move skill

def tmp_skill_lib(skill_ID):
    if skill_ID==0:
        start1 = [0.28667097385077456, -0.48847984909271797, 0.5940973561794143 + 0.022]
        mid1 = [0.28667097385077456 - 0.25, -0.48847984909271797, 0.5940973561794143 + 0.022 + 0.15]
        end1 = [0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.022]

        tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
        tra_go = np.array([np.linspace(start1[i], mid1[i], 100) for i in range(3)]).T
        tra_go2 = np.array([np.linspace(mid1[i], end1[i], 100) for i in range(3)]).T

        tra1 = np.vstack((tra_pre, tra_go, tra_go2))
        return tra1
    elif skill_ID==1:
        start1=[0.28667097385077456, -0.48847984909271797, 0.5940973561794143+0.022]
        end1=[0.28667097385077456-0.5, -0.48847984909271797, 0.5940973561794143+0.022]

        tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
        tra_go = np.array([np.linspace(start1[i], end1[i], 200) for i in range(3)]).T

        tra1 = np.vstack((tra_pre, tra_go))
        return tra1
    elif skill_ID==2:
        start1 = [0.28667097385077456, -0.48847984909271797, 0.5940973561794143 + 0.022]
        mid1 = [0.28667097385077456 , -0.48847984909271797, 0.5940973561794143 + 0.022 + 0.08]
        mid2 = [0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.022 + 0.08]
        end1 = [0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.022]

        tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
        tra_go = np.array([np.linspace(start1[i], mid1[i], 20) for i in range(3)]).T
        tra_go2 = np.array([np.linspace(mid1[i], mid2[i], 160) for i in range(3)]).T
        tra_putdown = np.array([np.linspace(mid2[i], end1[i], 20) for i in range(3)]).T

        tra1 = np.vstack((tra_pre, tra_go, tra_go2,tra_putdown))
        return tra1
    else:
        print("no such skill")
        return -1

def plot_3d_curve(X,Y):
    plt.rcParams['font.family'] = 'Nimbus Roman'
    # plt.rcParams['font.size'] = 20
    plt.figure(figsize=(8, 6))
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

def plot_3d_curve_new(X,Y,Z):
    plt.rcParams['font.family'] = 'Nimbus Roman'
    plt.rcParams['font.size'] = 20
    fig=plt.figure(figsize=(10, 10))
    """
    Plots a 3D curve from a 3xN array.

    Parameters:
        X (numpy.ndarray): A 2D numpy array with three columns representing x, y, and z coordinates.
    """
    # Create a new figure
    # fig = plt.figure()

    # Add an Axes3D instance to the figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color='k', linestyle='-', linewidth=3,label='Reference Skill Curve')
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='r', linestyle='--', linewidth=3, label='invariant DMP Fit Curve')
    ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], color='b', linestyle='--', linewidth=3, label='invariant DMP Task Curve')

    # Add labels and title
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # ax.set_title('3D Line Plot')

    # Show legend
    ax.legend()
    plt.legend(fontsize=20)
    # plt.xticks(fontsize=16)  # Set font size of the tick labels on x-axis
    # plt.yticks(fontsize=16)
    # plt.zticks(fontsize=16)

    # plt.savefig('invariant DMP skill', dpi=600, format='png')
    plt.savefig('DMP skill Demo.png', dpi=600, format='png')

    # Show the plot
    plt.show()


def reset_dmp():
    num_keys=len(dmps)
    for i in range(1,num_keys+1):
        key_ = f"DMP{i}"
        dmps[key_].reset_state()





# Convert kinematic skills to dynamic skills
for i in range(1, skill_num+1):
    key=f"DMP{i}"
    X=tmp_skill_lib(i-1)
    dmps[key].imitate_path(x_des=X)



X1=tmp_skill_lib(0)
X2=tmp_skill_lib(1)
X3=tmp_skill_lib(2)

x_track1, _, _, _ = dmps[f"DMP{1}"].rollout()
x_track2, _, _, _ = dmps[f"DMP{2}"].rollout()

x_track3, _, _, _ = dmps[f"DMP{3}"].rollout()
reset_dmp()
dmps[f"DMP{3}"].x_goal=np.array([0.28667097385077456 - 0.1, -0.48847984909271797+0.05, 0.5940973561794143 + 0.1])
dmps[f"DMP{3}"].x_0=dmps[f"DMP{3}"].x_0+np.array([0.1,0.1,0.])
x_track3_new,_, _, _ = dmps[f"DMP{3}"].rollout()

# dmps[f"DMP{2}"].x_goal=np.array([0.28667097385077456, -0.48847984909271797, 0.5940973561794143-0.17])+np.array([-0.06,0.06,0])
# dmps[f"DMP{2}"].x_0=np.array([0.28667097385077456, -0.48847984909271797, 0.5940973561794143-0.17])+np.array([-0.2,0.2,0])
# x_track2, _, _, _ = dmps[f"DMP{2}"].rollout()
# x_track2_new=[]



# plot_3d_curve(X1,x_track1)
# plot_3d_curve(X2,x_track2)
plot_3d_curve_new(X3,x_track3,x_track3_new)


# with open('joint_kinova_f2f.json', 'w') as json_file:
#     json.dump(joint_angle_list_Kinova, json_file)
#
# with open('joint_UR5e_f2f.json', 'w') as json_file:
#     json.dump(joint_angle_list_UR5e, json_file)


