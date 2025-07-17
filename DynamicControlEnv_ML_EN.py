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

from gymnasium import Env
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Tuple

from dmp import dmp_cartesian as dmp
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)

# Define the rotation (90 degrees around Z-axis)
rotation_euler = [0, 0, 3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion = pb.getQuaternionFromEuler(rotation_euler)

# Define the rotation (90 degrees around Z-axis)
rotation_euler2 = [0, 0, -3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion2 = pb.getQuaternionFromEuler(rotation_euler2)

rotation_ee = [0,0,0.707,0.707]

bias_kinova=np.array([0.0,0.0,0.022])
bias_ur5e=np.array([0,0,0])
height_bias=np.array([0,0,-0.05])
table_bias=np.array([0.0,0.0,-0.2])
execution_bias = np.array([-0.00109593, - 0.0001968, 0.0])
end_effector_zbias=0.5940973561794143 + 0.022+table_bias[2]
# end_effector_target=[0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.02]

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
grip_open_steps=25
grip_close_steps=25

iter_flag=False
dmp_step_num=10

sleep_time_set=0.1

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




# Convert kinematic skills to dynamic skills
for i in range(1, skill_num+1):
    key=f"DMP{i}"
    X=tmp_skill_lib(i-1)
    dmps[key].imitate_path(x_des=X)



class ControlModuleEnv(Env):
    def __init__(self,GUI_flag=False):
        super(ControlModuleEnv, self).__init__()
        self.reach_step=0
        self.close_step=0
        self.execution_time=0

        self.all_step_count=0
        self.env_end_flag=False
        self.initial_catch_flag=False
        self.contact_flag=False
        self.start_dmp_flag=False
        self.task_success=False
        self.energy=0
        self.dt=0.01
        self.torque_thres=35

        self.action_space = Box(low=np.array([1.0]), high=np.array([10.0]), dtype=np.float32)


        self.GUI_flag = GUI_flag

        if self.GUI_flag:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)

        pb.setGravity(0, 0, -9.8)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        pb.setTimeStep(1 / 100)
        plane_id = pb.loadURDF("plane.urdf")

        self.robot1 = Kinova_Robotiq()

        # robot2 = UR5_Robotiq()


        block_start_position = np.array([0.28667097385077456, -0.48847984909271797, 0.5940973561794143 - 0.17])
        block_start_orientation = pb.getQuaternionFromEuler([0, 0, 0])

        # Define the lower and upper bounds for each dimension
        ini_low_bounds = np.array([-0.1, -0.1, 0])
        ini_high_bounds = np.array([0, 0.1, 0])

        # Generate a random array with these bounds
        ini_random_array = np.random.uniform(low=ini_low_bounds, high=ini_high_bounds)
        block_bias = ini_random_array
        # block_bias=np.array([-0.5,0.2,0])

        # Define the lower and upper bounds for each dimension
        goal_low_bounds = np.array([-0.5, -0.1, 0])
        goal_high_bounds = np.array([-0.3, 0.2, 0])

        # Generate a random array with these bounds
        goal_random_array = np.random.uniform(low=goal_low_bounds, high=goal_high_bounds)
        target_bias = goal_random_array

        mass_low_bounds=0.05
        mass_high_bounds=2.0

        friction_low_bounds=0.1
        friction_high_bound=1

        self.block_start_position_random=block_start_position + table_bias + block_bias
        self.block_target_position_random = np.array(
            [0.28667097385077456, -0.48847984909271797, 0.5940973561794143]) + table_bias + target_bias


        self.mass=np.random.uniform(low=mass_low_bounds, high=mass_high_bounds)
        self.friction=np.random.uniform(low=friction_low_bounds, high=friction_high_bound) # recommended 0.4


        self.block_id = pb.loadURDF("/home/tianxu/Documents/Dynamic Skill Learning/Robot_Sim/urdf/object/magnetic.urdf",self.block_start_position_random
                               , block_start_orientation)
        self.table_id = pb.loadURDF("/home/tianxu/Documents/Dynamic Skill Learning/Robot_Sim/urdf/object/table.urdf",
                               basePosition=np.array([0, -0.4, -0.22]) + table_bias, useFixedBase=True)

        # Change color of the base link (linkIndex = -1 for the base)
        pb.changeVisualShape(self.block_id, linkIndex=-1, rgbaColor=[1, 0, 0, 1])

        # Set friction for object1
        pb.changeDynamics(self.block_id, -1, lateralFriction=1)
        pb.changeDynamics(self.block_id, -1, rollingFriction=0.6, spinningFriction=0.2)
        pb.changeDynamics(self.block_id, linkIndex=-1, mass=self.mass)  # Change the mass of the first link to 5 kg
        # pb.changeDynamics(block_id, linkIndex=-1, mass=10)  # Change the mass of the first link to 5 kg

        # Set friction for object2
        pb.changeDynamics(self.table_id, -1, lateralFriction=self.friction)
        pb.changeDynamics(self.table_id, -1, rollingFriction=1)

        self.robot1.initialize(base_pos=[0.29, -0.8, 0.025], base_ori=rotation_quaternion)
        # robot2.initialize(base_pos=[-0.0635, 0.0, 0.067], base_ori=rotation_quaternion2)
        self.robot1._setRobotiqPosition(0.22)
        reset_dmp()

        #


    def go_catch_object(self,target_position):
        print("reach_step", self.reach_step, "close_step", self.close_step)
        finish_flag = False

        l = self.robot1._getLinkState(self.robot1.end_effector_index)
        end_effector_pos1 = np.array(l[0])

        # execution_bias = np.array([0, 0, 0.0])
        base_position_grip = np.array(target_position)
        if base_position_grip[2] < end_effector_zbias:
            base_position_grip[2] = end_effector_zbias
        re_grip_vec = np.array(base_position_grip) - np.array(end_effector_pos1)
        # dist_target_ = np.linalg.norm(np.array(base_position_grip) - np.array(end_effector_pos1))
        dist_target_ = np.linalg.norm(re_grip_vec)
        print("dist_target_", dist_target_)

        if (self.reach_step > grip_open_steps) and (dist_target_ < close_thres):
            print("ready to grip")
            # pdb.set_trace()
            joint_angle = self.robot1._calculateIK(np.array(base_position_grip) + [0.0, 0.0, -0.0], rotation_ee)
            if self.close_step <= grip_close_steps:
                self.robot1._setRobotiqPosition(0.23)
                self.close_step += 1
            else:
                finish_flag = True

            return joint_angle, finish_flag

        if self.reach_step <= grip_open_steps:
            self.robot1._setRobotiqPosition(0.0)
            joint_angle = self.robot1._calculateIK(np.array(end_effector_pos1) + [0.0, 0.0, -0.0], rotation_ee)
            self.reach_step += 1
            # robot1._resetJointStateforce(joint_angle)
        else:
            if np.linalg.norm(dist_target_) < step_dist_return:
                joint_angle = self.robot1._calculateIK(np.array(base_position_grip) + [0.0, 0.0, -0.0], rotation_ee)
            else:
                next_pos = end_effector_pos1 + step_dist_return * re_grip_vec / np.linalg.norm(
                    re_grip_vec) + execution_bias
                if next_pos[2] < end_effector_zbias:
                    next_pos[2] = end_effector_zbias + 0.01

                # next_pos = end_effector_pos1 + step_dist_return * re_grip_vec / np.linalg.norm(re_grip_vec)+ execution_bias
                joint_angle = self.robot1._calculateIK(np.array(next_pos) + [0.0, 0.0, -0.0], rotation_ee)
                print("go to", next_pos)

            self.robot1._resetJointStateforce(joint_angle)

        return joint_angle, finish_flag

    def init_global_params(self):
        self.reach_step = 0
        self.close_step = 0



    def step(self,action):
        global reach_step, close_step

        key = f"DMP{int(action[0].round())}"
        tau=action[1]
        dmps[key].x_goal = self.block_target_position_random + bias_kinova
        reward = -1

        info={}

        l = self.robot1._getLinkState(self.robot1.end_effector_index)
        end_effector_pos1 = np.array(l[0])
        end_effector_ori1=np.array(l[1])
        joint_angle_m=self.robot1._getJointStateAngle()



        obs = np.concatenate((np.array([self.mass_obs]),np.array([self.friction_obs]),np.array([action[0]]),end_effector_pos1, end_effector_ori1,joint_angle_m, np.array([dmps[key].cs.s]),
                              np.array(self.block_start_position_random), np.array(self.block_target_position_random)))

        if self.all_step_count>=max_step:
            self.env_end_flag=True
            reward+=-100

            print("reach step limitation")
            return obs,reward,self.task_success,info


        if self.all_step_count < initial_step_num:
            self.all_step_count+=1
            pb.stepSimulation()
            time.sleep(sleep_time_set)

            print("wait initialize")
            return obs,reward,self.task_success,info

        elif not (self.initial_catch_flag) :
            if self.all_step_count == initial_step_num:
                print("initial first position")
                # x_track_s = dmps[key].x_0

                # initialize parameters
                self.init_global_params()

                base_position, base_orientation = pb.getBasePositionAndOrientation(self.block_id)
                base_position = np.array(base_position)
                base_position[2] = end_effector_zbias
                self.base_position=base_position
                _, step_finish_flag = self.go_catch_object(self.base_position)
            else:
                print("go and catch", self.base_position)
                _, step_finish_flag = self.go_catch_object(np.array(self.base_position))

            pb.stepSimulation()
            time.sleep(sleep_time_set)
            self.all_step_count += 1

            if step_finish_flag == True:
                l = self.robot1._getLinkState(self.robot1.end_effector_index)
                end_effector_pos1 = np.array(l[0])
                end_effector_pos1_modified = end_effector_pos1 + execution_bias
                end_effector_pos1_modified[2] = end_effector_zbias
                dmps[key].x_0 = end_effector_pos1_modified
                self.initial_catch_flag = True

                print("catch finish, time steps:", i)

            return obs,reward,self.task_success,info



            # pdb.set_trace()

        if self.initial_catch_flag:
            self.start_dmp_flag=True
            self.execution_time+=0.01
            print("start dmp execution")
            for k in range(dmp_step_num):
                # if i < 110:
                #     x_track_s, _, _ = dmps[key].step(tau=1)
                # else:
                x_track_s, _, _ = dmps[key].step(tau=tau)
                print("x_track_s", x_track_s)
            joint_angle1 = self.robot1._calculateIK(np.array(x_track_s) + [0.0, 0.0, -0.0], rotation_ee)
            self.robot1._resetJointStateforce(joint_angle1)

            pb.stepSimulation()
            time.sleep(sleep_time_set)


            err_abs = np.linalg.norm(x_track_s - dmps[key].x_goal)
            err_rel = err_abs / (np.linalg.norm(dmps[key].x_goal - dmps[key].x_0) + 1e-14)
            print("err_rel", err_rel, "err_abs", err_abs)

            self.robot1._resetJointStateforce(joint_angle1)
            manipulator_torque, gripper_torque = self.robot1._getJointStateTorque()
            manipulator_power=self.robot1._getJointStatePower()
            self.energy+=manipulator_power*self.dt

            gripper_signal = np.sum(np.abs(np.array(gripper_torque)))

            print("gripper_signal", gripper_signal)

            l = self.robot1._getLinkState(self.robot1.end_effector_index)
            end_effector_pos1 = np.array(l[0])
            # execution_bias = np.array([-0.00109593, - 0.0001968, 0.0])

            if gripper_signal < loose_thres:
                if self.contact_flag:
                    print("block slip out")
                    self.contact_flag = False
                    self.env_end_flag=True
                    reward+=-20

                    # if re_grip_flag==False:
                    #     re_grip_flag=True
                    #

            if gripper_signal > loose_thres:
                self.contact_flag = True
                print("hold the block")
            # print("bias:",execution_bias)
            # pdb.set_trace()

            torque_flag = np.all(np.abs(manipulator_torque) < self.torque_thres)
            print("manipulator torque,",np.abs(manipulator_torque) ,"judge",torque_flag)

            if (not torque_flag) and (self.execution_time>0.1):
                self.env_end_flag = True
                reward+=-100
                print("out of max torque limitation")

            base_position, base_orientation = pb.getBasePositionAndOrientation(self.block_id)
            print("block_position", base_position)
            dist_target = np.linalg.norm(np.array(base_position) - np.array(self.block_target_position_random))
            print("object dist error:", dist_target)
            if dist_target < object_dist_thres:
                print("target achieved")
                print("energy consumption:",self.energy)
                self.env_end_flag = True
                self.task_success=True
                reward+=200



            err_abs = np.linalg.norm(x_track_s - dmps[key].x_goal)
            err_rel = err_abs / (np.linalg.norm(dmps[key].x_goal - dmps[key].x_0) + 1e-14)
            # iter_flag = ((i >= dmps[key].cs.timesteps) and err_rel <= dmps[key].tol)
            # iter_flag = ((i >= 10) and err_rel <= dmps[key].tol)
            iter_flag =  err_rel <= dmps[key].tol

            if iter_flag and (not self.env_end_flag):
                print("DMP skill end")
                print("execution time:",self.execution_time)
                print("object dist error:", dist_target)
                print("energy consumption:", self.energy)
                self.env_end_flag = True
                self.task_success = True
                reward += 200

            l = self.robot1._getLinkState(self.robot1.end_effector_index)
            end_effector_pos1 = np.array(l[0])
            end_effector_ori1 = np.array(l[1])
            joint_angle_m = self.robot1._getJointStateAngle()

            obs = np.concatenate(
                (np.array([self.mass_obs]),np.array([self.friction_obs]),np.array([action[0]]), end_effector_pos1, end_effector_ori1, joint_angle_m, np.array([dmps[key].cs.s]),
                 np.array(self.block_start_position_random), np.array(self.block_target_position_random)))

        return obs,reward,self.task_success,info


    def reset(self):
        self.reach_step = 0
        self.close_step = 0
        self.execution_time = 0

        self.all_step_count=0
        self.env_end_flag=False
        self.initial_catch_flag=False
        self.contact_flag=False
        self.start_dmp_flag=False
        self.task_success=False
        self.energy=0
        self.dt=0.01


        if pb.getConnectionInfo()['isConnected']:
            pb.disconnect()

        if self.GUI_flag:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)

        pb.setGravity(0, 0, -9.8)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        pb.setTimeStep(1 / 100)
        plane_id = pb.loadURDF("plane.urdf")

        self.robot1 = Kinova_Robotiq()

        # robot2 = UR5_Robotiq()

        block_start_position = np.array([0.28667097385077456, -0.48847984909271797, 0.5940973561794143 - 0.17])
        block_start_orientation = pb.getQuaternionFromEuler([0, 0, 0])

        # Define the lower and upper bounds for each dimension
        ini_low_bounds = np.array([-0.1, -0.1, 0])
        ini_high_bounds = np.array([0, 0.1, 0])

        # Generate a random array with these bounds
        ini_random_array = np.random.uniform(low=ini_low_bounds, high=ini_high_bounds)
        block_bias = ini_random_array
        # block_bias=np.array([-0.5,0.2,0])

        # Define the lower and upper bounds for each dimension
        goal_low_bounds = np.array([-0.5, -0.1, 0])
        goal_high_bounds = np.array([-0.3, 0.2, 0])

        # Generate a random array with these bounds
        goal_random_array = np.random.uniform(low=goal_low_bounds, high=goal_high_bounds)
        target_bias = goal_random_array

        mass_low_bounds = 0.05
        mass_high_bounds = 1.0

        friction_low_bounds = 0.1
        friction_high_bound = 0.5

        self.block_start_position_random = block_start_position + table_bias + block_bias
        self.block_target_position_random = np.array(
            [0.28667097385077456, -0.48847984909271797, 0.5940973561794143]) + table_bias + target_bias

        self.mass = np.random.uniform(low=mass_low_bounds, high=mass_high_bounds)
        self.friction = np.random.uniform(low=friction_low_bounds, high=friction_high_bound)  # recommended 0.4

        mass_error_thres=0.02
        friction_error_thres=0.1
        self.mass_obs=self.mass+np.random.normal(loc=0.0, scale=mass_error_thres/0.7979)
        self.friction_obs=self.friction+np.random.normal(loc=0.0, scale=friction_error_thres/0.7979)

        self.block_id = pb.loadURDF(
            "/home/tianxu/Documents/Dynamic Skill Learning/Robot_Sim/urdf/object/magnetic.urdf",
            self.block_start_position_random
            , block_start_orientation)
        self.table_id = pb.loadURDF("/home/tianxu/Documents/Dynamic Skill Learning/Robot_Sim/urdf/object/table.urdf",
                                    basePosition=np.array([0, -0.4, -0.22]) + table_bias, useFixedBase=True)

        # Change color of the base link (linkIndex = -1 for the base)
        pb.changeVisualShape(self.block_id, linkIndex=-1, rgbaColor=[1, 0, 0, 1])

        # Set friction for object1
        pb.changeDynamics(self.block_id, -1, lateralFriction=1)
        pb.changeDynamics(self.block_id, -1, rollingFriction=0.6, spinningFriction=0.2)
        pb.changeDynamics(self.block_id, linkIndex=-1, mass=self.mass)  # Change the mass of the first link to 5 kg
        # pb.changeDynamics(block_id, linkIndex=-1, mass=10)  # Change the mass of the first link to 5 kg

        # Set friction for object2
        pb.changeDynamics(self.table_id, -1, lateralFriction=self.friction)
        pb.changeDynamics(self.table_id, -1, rollingFriction=1)

        self.robot1.initialize(base_pos=[0.29, -0.8, 0.025], base_ori=rotation_quaternion)
        self.robot1._setRobotiqPosition(0.22)
        # robot2.initialize(base_pos=[-0.0635, 0.0, 0.067], base_ori=rotation_quaternion2)
        reset_dmp()

        l = self.robot1._getLinkState(self.robot1.end_effector_index)
        end_effector_pos1 = np.array(l[0])
        end_effector_ori1 = np.array(l[1])
        print(end_effector_ori1)
        joint_angle_m = self.robot1._getJointStateAngle()


        # obs = np.concatenate((np.array([0]),np.array(end_effector_pos1) , np.array(end_effector_ori1), np.array(joint_angle_m),
        #                       np.array(dmps[key].cs.s),
        #                       np.array(self.block_start_position_random), np.array(self.block_target_position_random)))

        obs = np.concatenate((np.array([self.mass_obs]),np.array([self.friction_obs]),np.array([0.0]),end_effector_pos1,end_effector_ori1, joint_angle_m, np.array([dmps[key].cs.s]),np.array(self.block_start_position_random), np.array(self.block_target_position_random)))

        # obs = np.concatenate((np.array([self.mass]),np.array([self.friction]),np.array(self.block_start_position_random), np.array(self.block_target_position_random)))
        info = {}
        return obs,info















