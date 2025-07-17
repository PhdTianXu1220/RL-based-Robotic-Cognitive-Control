import sys

# Path to the directory containing the package
package_path = '/home/tianxu/Documents/DMP-python/Dual_Arm_New'

# Add this path to sys.path
if package_path not in sys.path:
    sys.path.append(package_path)

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


def reset_dmp():
    num_keys=len(dmps)
    for i in range(1,num_keys+1):
        key_ = f"DMP{i}"
        dmps[key_].reset_state()

def init_global_params():
    global reach_step, close_step
    reach_step=0
    close_step=0


def go_catch_object(target_position):
    global reach_step,close_step
    print("reach_step",reach_step,"close_step",close_step)
    finish_flag=False

    l = robot1._getLinkState(robot1.end_effector_index)
    end_effector_pos1 = np.array(l[0])

    # execution_bias = np.array([0, 0, 0.0])
    base_position_grip = np.array(target_position)
    if base_position_grip[2] < end_effector_zbias:
        base_position_grip[2] = end_effector_zbias
    re_grip_vec=np.array(base_position_grip) - np.array(end_effector_pos1)
    # dist_target_ = np.linalg.norm(np.array(base_position_grip) - np.array(end_effector_pos1))
    dist_target_ = np.linalg.norm(re_grip_vec)
    print("dist_target_",dist_target_)

    if (reach_step>grip_open_steps) and (dist_target_<close_thres):
        print("ready to grip")
        # pdb.set_trace()
        joint_angle = robot1._calculateIK(np.array(base_position_grip) + [0.0, 0.0, -0.0], rotation_ee)
        if close_step<=grip_close_steps:
            robot1._setRobotiqPosition(0.23)
            close_step+=1
        else:
            finish_flag = True


        return joint_angle, finish_flag

    if reach_step<=grip_open_steps:
        robot1._setRobotiqPosition(0.0)
        joint_angle = robot1._calculateIK(np.array(end_effector_pos1) + [0.0, 0.0, -0.0], rotation_ee)
        reach_step+=1
        # robot1._resetJointStateforce(joint_angle)
    else:
        if np.linalg.norm(dist_target_) < step_dist_return:
            joint_angle = robot1._calculateIK(np.array(base_position_grip) + [0.0, 0.0, -0.0], rotation_ee)
        else:
            next_pos = end_effector_pos1 + step_dist_return * re_grip_vec / np.linalg.norm(re_grip_vec)+ execution_bias
            if next_pos[2]<end_effector_zbias:
                next_pos[2]=end_effector_zbias+0.01

            # next_pos = end_effector_pos1 + step_dist_return * re_grip_vec / np.linalg.norm(re_grip_vec)+ execution_bias
            joint_angle = robot1._calculateIK(np.array(next_pos) + [0.0, 0.0, -0.0], rotation_ee)
            print("go to", next_pos)

        robot1._resetJointStateforce(joint_angle)

    return joint_angle,finish_flag



# Convert kinematic skills to dynamic skills
for i in range(1, skill_num+1):
    key=f"DMP{i}"
    X=tmp_skill_lib(i-1)
    dmps[key].imitate_path(x_des=X)



X1=tmp_skill_lib(0)
X2=tmp_skill_lib(1)
X3=tmp_skill_lib(2)

# x_track1, _, _, _ = dmps[f"DMP{1}"].rollout()
# x_track2, _, _, _ = dmps[f"DMP{2}"].rollout()
# dmps[f"DMP{3}"].x_goal=np.array([0.28667097385077456 - 0.2, -0.48847984909271797, 0.5940973561794143 + 0.1])
# dmps[f"DMP{3}"].x_0=dmps[f"DMP{3}"].x_0+np.array([0.1,0.1,0])
# x_track3, _, _, _ = dmps[f"DMP{3}"].rollout()
# x_track3_new=[]

# dmps[f"DMP{2}"].x_goal=np.array([0.28667097385077456, -0.48847984909271797, 0.5940973561794143-0.17])+np.array([-0.06,0.06,0])
# dmps[f"DMP{2}"].x_0=np.array([0.28667097385077456, -0.48847984909271797, 0.5940973561794143-0.17])+np.array([-0.2,0.2,0])
# x_track2, _, _, _ = dmps[f"DMP{2}"].rollout()
# x_track2_new=[]



# plot_3d_curve(X1,x_track1)
# plot_3d_curve(X2,x_track2)
# plot_3d_curve(X2,x_track2)
iter_flag=False
initial_catch_flag=False
t=0


end_effector_zbias=0.5940973561794143 + 0.022+table_bias[2]
end_effector_target=[0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.02]

tra1=tmp_skill_lib(0)


# run steps
step_num=len(tra1)


pb.connect(pb.GUI)
# Load robot with accurate dynamics
# pb.setPhysicsEngineParameter(numSolverIterations=150, numSubSteps=10)

pb.setGravity(0, 0, -9.8)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
plane_id = pb.loadURDF("plane.urdf")

robot1 = Kinova_Robotiq()
# robot2 = UR5_Robotiq()





# block_start_position =np.array(tra1[0])+bias_kinova
block_start_position =np.array([0.28667097385077456, -0.48847984909271797, 0.5940973561794143-0.17])
# block_start_position =np.array([0.28667097385077456, -0.48847984909271797, 0.5940973561794143])
# block_start_position2 = np.array(tra2[0])+bias_ur5e
# block_start_position3 =np.array(tra3[tra1_connect_time])+bias_kinova++height_bias
# block_start_position4 = np.array(tra4[tra2_connect_time])+bias_ur5e++height_bias

# block_target_position=[-0.18100409137299697, -0.49399412541519766, 0.4171282952898786]
# block_bias=np.array([-0.06,0.06,0])
# the initial range of block should be in
# block_bias=np.array([0.0,0.1,0])
# Define the lower and upper bounds for each dimension
ini_low_bounds = np.array([-0.1, -0.1, 0])
ini_high_bounds = np.array([0, 0.1, 0])

# Generate a random array with these bounds
ini_random_array = np.random.uniform(low=ini_low_bounds, high=ini_high_bounds)
block_bias=ini_random_array
# block_bias=np.array([-0.5,0.2,0])

# Define the lower and upper bounds for each dimension
goal_low_bounds = np.array([-0.5, -0.1, 0])
goal_high_bounds = np.array([-0.3, 0.2, 0])

# Generate a random array with these bounds
goal_random_array = np.random.uniform(low=goal_low_bounds, high=goal_high_bounds)
target_bias=goal_random_array



block_start_orientation = pb.getQuaternionFromEuler([0, 0, 0])
block_id = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/magnetic.urdf", block_start_position+table_bias+block_bias, block_start_orientation)
table_id = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/table.urdf", basePosition=np.array([0, -0.4, -0.22])+table_bias,useFixedBase=True)
# block_id2 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/magnetic.urdf", block_start_position2, block_start_orientation)
# block_id3 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/cube1.urdf", block_start_position3, block_start_orientation)
# block_id4 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/cube2.urdf", block_start_position4, block_start_orientation)
# Change color of the base link (linkIndex = -1 for the base)
pb.changeVisualShape(block_id, linkIndex=-1, rgbaColor=[1, 0, 0, 1])

# Set friction for object1
pb.changeDynamics(block_id, -1, lateralFriction=1)
pb.changeDynamics(block_id, -1, rollingFriction=0.6,spinningFriction=0.2)
pb.changeDynamics(block_id, linkIndex=-1, mass=1)  # Change the mass of the first link to 5 kg
# pb.changeDynamics(block_id, linkIndex=-1, mass=10)  # Change the mass of the first link to 5 kg


# Set friction for object2
pb.changeDynamics(table_id, -1, lateralFriction=0.4)
pb.changeDynamics(table_id, -1, rollingFriction=1)

robot1.initialize(base_pos=[0.29, -0.8, 0.025], base_ori=rotation_quaternion)
# robot2.initialize(base_pos=[-0.0635, 0.0, 0.067], base_ori=rotation_quaternion2)

pb.setTimeStep(1/100)
# robot1._setRobotiqPosition(0.22)
# robot2._setRobotiqPosition(0.02)

# print("key config1 robot1",np.array(tra1[0]))
# print("key config1 robot2",np.array(tra2[0]))


# print("key config2 robot1",np.array(tra1[tra1_connect_time]))
# print("key config2 robot2",np.array(tra2[tra2_connect_time]))


# print("key config3 robot1",np.array(tra1[-1]))
# print("key config3 robot2",np.array(tra2[-1]))

pdb.set_trace()

rotation_ee = [0,0,0.707,0.707]

joint_angle_list_Kinova=[]
joint_angle_list_UR5e=[]

manipulator_torque_time=[]
gripper_torque_time=[]
gripper_signal_time=[]

DMP_execution_step=1

reset_dmp()

# block_target_position=np.array([0.28667097385077456 - 0.5, -0.48847984909271797 + 0.2, 0.5940973561794143]) + table_bias
block_target_position=np.array([0.28667097385077456, -0.48847984909271797 , 0.5940973561794143]) + table_bias+target_bias

for i in range(max_step):
    # reset_dmp()
    key = f"DMP{1}"
    # dmps[key].x_goal = np.array([0.28667097385077456 - 0.2, -0.48847984909271797 + 0.2, 0.5940973561794143 + 0.022+0.01])+table_bias

    dmps[key].x_goal = block_target_position+bias_kinova


    # after each step, clean global params values

    # initial catchup function
    if i<initial_step_num:
        pb.stepSimulation()
        time.sleep(0.1)
        print("wait")
        continue
    elif not (initial_catch_flag):
        if i == initial_step_num:
            print("initial first position")
            # x_track_s = dmps[key].x_0

            # initialize parameters
            init_global_params()

            base_position, base_orientation = pb.getBasePositionAndOrientation(block_id)
            base_position=np.array(base_position)
            base_position[2]=end_effector_zbias
            _, step_finish_flag = go_catch_object(base_position)
        else:
            print("go and catch",base_position)
            _, step_finish_flag = go_catch_object(np.array(base_position))

        if step_finish_flag==True:
            l = robot1._getLinkState(robot1.end_effector_index)
            end_effector_pos1 = np.array(l[0])
            end_effector_pos1_modified=end_effector_pos1+execution_bias
            end_effector_pos1_modified[2]=end_effector_zbias
            dmps[key].x_0 = end_effector_pos1_modified
            pb.stepSimulation()
            time.sleep(0.1)
            initial_catch_flag=True
            print("catch finish, time steps:",i)
        else:
            pb.stepSimulation()
            time.sleep(0.1)
            continue



    # pdb.set_trace()
    print("start dmp execution")
    for k in range(dmp_step_num):
        if i<110:
            x_track_s, _, _ = dmps[key].step(tau=1)
        else:
            x_track_s, _, _ = dmps[key].step(tau=1)
        print("x_track_s", x_track_s)
    joint_angle1 = robot1._calculateIK(np.array(x_track_s) + [0.0, 0.0, -0.0], rotation_ee)
    robot1._resetJointStateforce(joint_angle1)
    pb.stepSimulation()
    time.sleep(0.1)

    # l = robot1._getLinkState(robot1.end_effector_index)
    # end_effector_pos1 = np.array(l[0])
    # execution_bias = np.array([-0.00109593, - 0.0001968, 0.0])


    err_abs = np.linalg.norm(x_track_s - dmps[key].x_goal)
    err_rel = err_abs / (np.linalg.norm(dmps[key].x_goal - dmps[key].x_0) + 1e-14)
    print("err_rel",err_rel,"err_abs",err_abs)
    # iter_flag = ((i >= dmps[key].cs.timesteps) and err_rel <= dmps[key].tol)

    # manipulator_torque, gripper_torque = robot1._getJointStateTorque()
    # gripper_signal = np.sum(np.abs(np.array(gripper_torque)))







    # t += 1

    # iter_flag = ((t >= dmps[key].cs.timesteps) and err_rel <= dmps[key].tol)


    # if i==1:
    #     pdb.set_trace()
    # robot1._setRobotiqPosition(0.22)
    #
    # if not re_grip_flag:
    #     if i ==0:
    #         joint_angle1 = robot1._calculateIK(np.array(dmps[key].x_0)+[0.0,0.0,-0.0], rotation_ee)
    #     else:
    #         for k in range(dmp_step_num):
    #             x_track_s, _, _ = dmps[key].step(tau=5)
    #             print("x_track_s", x_track_s)
    #         joint_angle1 = robot1._calculateIK(np.array(x_track_s) + [0.0, 0.0, -0.0], rotation_ee)
    #
    # elif re_grip_flag and re_grip_wait_count<grip_open_steps:
    #     robot1._setRobotiqPosition(0.0)
    #     joint_angle1 = robot1._calculateIK(next_pos, rotation_ee)
    #     re_grip_wait_count+=1
    #     print("here",next_pos)
    #     # pdb.set_trace()
    #
    # elif re_grip_flag and grip_open_steps<=re_grip_wait_count<grip_open_steps+grip_close_steps:
    #     robot1._setRobotiqPosition(0.22)
    #     joint_angle1 = robot1._calculateIK(next_pos, rotation_ee)
    #     re_grip_wait_count += 1
    # elif re_grip_wait_count>=grip_open_steps+grip_close_steps:
    #     print("continue moving")
    #     joint_angle1 = robot1._calculateIK(next_pos, rotation_ee)

    robot1._resetJointStateforce(joint_angle1)
    manipulator_torque, gripper_torque = robot1._getJointStateTorque()
    gripper_signal = np.sum(np.abs(np.array(gripper_torque)))

    l = robot1._getLinkState(robot1.end_effector_index)
    end_effector_pos1=np.array(l[0])
    execution_bias=np.array([-0.00109593, - 0.0001968,  0.0])

    if gripper_signal<loose_thres:
        if contact_flag:
            print("block slip out")
            contact_flag=False
            pdb.set_trace()

            # if re_grip_flag==False:
            #     re_grip_flag=True
                #

    if gripper_signal>loose_thres:
        contact_flag=True
        print("hold the block")
    # print("bias:",execution_bias)
    # pdb.set_trace()


    base_position, base_orientation = pb.getBasePositionAndOrientation(block_id)
    print("block_position", base_position)
    dist_target=np.linalg.norm(np.array(base_position)-np.array(block_target_position))
    print("object dist error:",dist_target)
    if dist_target<object_dist_thres:
        print("target achieved")
        break

    err_abs = np.linalg.norm(x_track_s - dmps[key].x_goal)
    err_rel = err_abs / (np.linalg.norm(dmps[key].x_goal - dmps[key].x_0) + 1e-14)
    # iter_flag = ((i >= dmps[key].cs.timesteps) and err_rel <= dmps[key].tol)
    iter_flag = ((i >= 10) and err_rel <= dmps[key].tol)

    if iter_flag:
        print("DMP skill end")
        print("object dist error:",dist_target)
        break




    # joint_angle2 = robot2._calculateIK(np.array(tra2[i])+[0.0,0.0,0.0], rotation_ee)

    # joint_angle1[3]=joint_angle1[3]-2*np.pi
    # joint_angle1[5] = joint_angle1[5] - 2 * np.pi





    if gripper_signal<loose_thres:
        if contact_flag:
            print("block slip out")
            contact_flag=False

            if re_grip_flag==False:
                re_grip_flag=True
                # pdb.set_trace()

    if gripper_signal>loose_thres:
        contact_flag=True
        print("hold the block")

    if re_grip_flag and re_grip_wait_count < grip_open_steps+grip_close_steps:
        base_position_grip = np.array(base_position)
        base_position_grip[2] = end_effector_zbias
        re_grip_vec = base_position_grip - end_effector_pos1-execution_bias
        next_pos = base_position_grip

        if np.linalg.norm(re_grip_vec) < step_dist_return:
            next_pos = base_position_grip
            next_pos[2] = end_effector_zbias
        else:
            next_pos = end_effector_pos1 + step_dist_return * re_grip_vec / np.linalg.norm(re_grip_vec)+execution_bias
            print("go back to",next_pos)
    if re_grip_flag and re_grip_wait_count >= grip_open_steps+grip_close_steps:
        re_grip_vec = end_effector_target - end_effector_pos1
        # next_pos = end_effector_target

        if np.linalg.norm(re_grip_vec) < step_dist:
            next_pos = end_effector_target
        else:
            next_pos = end_effector_pos1 + step_dist * re_grip_vec / np.linalg.norm(re_grip_vec)
        next_pos[2]=end_effector_zbias

        print("go to target",next_pos)

    manipulator_torque_time.append(manipulator_torque)
    gripper_torque_time.append(gripper_torque)
    gripper_signal_time.append([gripper_signal])
    print("gripper_signal", gripper_signal)
    # pdb.set_trace()
    # pb.stepSimulation()
    # robot2._resetJointState(joint_angle2)
    # pb.stepSimulation()
    # pb.stepSimulation()
    # pb.resetBasePositionAndOrientation(block_id, np.array(tra1[i]) + bias_kinova,
    #                                    block_start_orientation)
    # pb.resetBasePositionAndOrientation(block_id2, np.array(tra2[i]) + bias_ur5e,
    #                                    block_start_orientation)
    # if i>=tra1_connect_time:
    #     pb.resetBasePositionAndOrientation(block_id3, np.array(tra1[i]) + bias_kinova+height_bias,
    #                                        block_start_orientation)


    # if i>=tra2_connect_time:
    #     pb.resetBasePositionAndOrientation(block_id4, np.array(tra2[i]) + bias_ur5e+height_bias,
    #                                        block_start_orientation)

    pb.stepSimulation()
    time.sleep(0.1)
    # joint_angle_list_Kinova.append(joint_angle1)
    # joint_angle_list_UR5e.append(joint_angle2)


print("total step:",i)

plot_joint_torque(manipulator_torque_time,'mani_torq_6.png')
plot_joint_torque(gripper_torque_time,'grip_torq_6.png')
plot_joint_torque(gripper_signal_time,'grip_signal_torq_6.png')

# with open('joint_kinova_f2f.json', 'w') as json_file:
#     json.dump(joint_angle_list_Kinova, json_file)
#
# with open('joint_UR5e_f2f.json', 'w') as json_file:
#     json.dump(joint_angle_list_UR5e, json_file)


