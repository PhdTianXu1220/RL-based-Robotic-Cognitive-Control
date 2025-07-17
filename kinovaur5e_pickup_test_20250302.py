from Robot_Sim.robots.kinova_robotiq import Kinova_Robotiq
from Robot_Sim.robots.ur5_robotiq_new import UR5_Robotiq
#self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
import pybullet as pb
import pybullet_data
import time
import pdb
import json
import numpy as np

# with open('tra_kinova_f2f.json', 'r') as json_file:
#     tra1 = json.load(json_file)
#
# # with open('tra_UR5e_f2f.json', 'r') as json_file:
# #     tra2 = json.load(json_file)
#
# with open('tra_kinova_f2f.json', 'r') as json_file:
#     tra3 = json.load(json_file)
#
# # with open('tra_UR5e_f2f.json', 'r') as json_file:
# #     tra4 = json.load(json_file)
#
#
# tra1_connect_time = min(range(len(tra3)), key=lambda i: tra3[i][2])
# # tra2_connect_time = min(range(len(tra4)), key=lambda i: tra4[i][2])

#create traj array for linear move
t_des = np.linspace(0.0, 1.0, 200)

start1=[0.28667097385077456, -0.48847984909271797, 0.5940973561794143+0.025]
end1=[0.28667097385077456, -0.48847984909271797, 0.5940973561794143+0.025+0.1]

tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
tra_go = np.array([np.linspace(start1[i], end1[i], 200) for i in range(3)]).T

tra1 = np.vstack((tra_pre, tra_go))

start2=[0.28667097385077456-0.3, -0.48847984909271797, 0.5940973561794143-0.19]
end2=[0.28667097385077456-0.3, -0.48847984909271797, 0.5940973561794143-0.19+0.1]

tra2_pre = np.array([np.linspace(start2[i], start2[i], 20) for i in range(3)]).T
tra2_go = np.array([np.linspace(start2[i], end2[i], 200) for i in range(3)]).T
tra2 = np.vstack((tra2_pre, tra2_go))

# # create traj array for lift up
# t_des = np.linspace(0.0, 1.0, 200)
#
# start1=[0.28667097385077456, -0.48847984909271797, 0.5940973561794143+0.02]
# mid1=[0.28667097385077456-0.25, -0.48847984909271797, 0.5940973561794143+0.02+0.15]
# end1=[0.28667097385077456-0.5, -0.48847984909271797, 0.5940973561794143+0.02]
#
# tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
# tra_go = np.array([np.linspace(start1[i], mid1[i], 100) for i in range(3)]).T
# tra_go2 = np.array([np.linspace(mid1[i], end1[i], 100) for i in range(3)]).T
#
# tra1 = np.vstack((tra_pre, tra_go,tra_go2))

# run steps
step_num=len(tra1)


pb.connect(pb.GUI)

pb.setGravity(0, 0, -9.8)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
plane_id = pb.loadURDF("plane.urdf")

robot1 = Kinova_Robotiq()
robot2 = UR5_Robotiq()

# Define the rotation (90 degrees around Z-axis)
rotation_euler = [0, 0, 3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion = pb.getQuaternionFromEuler(rotation_euler)

# Define the rotation (90 degrees around Z-axis)
rotation_euler2 = [0, 0, -3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion2 = pb.getQuaternionFromEuler(rotation_euler2)

bias_kinova=np.array([0.0,0.0,-0.155-0.02])
bias_ur5e=np.array([0,0,0])
height_bias=np.array([0,0,-0.05])

# block_start_position =np.array(tra1[0])+bias_kinova
rod_start_position =[0.28667097385077456-0.15, -0.48847984909271797, 0.5940973561794143-0.17]
block_start_position =[0.28667097385077456, -0.48847984909271797, 0.5940973561794143-0.17]
block_start_position2 =[0.28667097385077456-0.3, -0.48847984909271797, 0.5940973561794143-0.17]
# block_start_position2 = np.array(tra2[0])+bias_ur5e
# block_start_position3 =np.array(tra3[tra1_connect_time])+bias_kinova++height_bias
# block_start_position4 = np.array(tra4[tra2_connect_time])+bias_ur5e++height_bias


block_start_orientation = pb.getQuaternionFromEuler([0, 0, 0])
block_id = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/rod.urdf", rod_start_position, block_start_orientation)
# block_id2 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/magnetic.urdf", block_start_position2, block_start_orientation)
table_id = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/table.urdf", basePosition=[0, -0.4, -0.22],useFixedBase=True)
# block_id2 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/magnetic.urdf", block_start_position2, block_start_orientation)
# block_id3 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/cube1.urdf", block_start_position3, block_start_orientation)
# block_id4 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/cube2.urdf", block_start_position4, block_start_orientation)
# Change color of the base link (linkIndex = -1 for the base)
pb.changeVisualShape(block_id, linkIndex=-1, rgbaColor=[1, 0, 0, 1])

# Set friction for object1
# 2 arm success pickup
# pb.changeDynamics(block_id, -1, lateralFriction=0.5)
# pb.changeDynamics(block_id, linkIndex=-1, mass=1)  # Change the mass of the first link to 5 kg

pb.changeDynamics(block_id, -1, lateralFriction=1)
pb.changeDynamics(block_id, linkIndex=-1, mass=0.001)  # Change the mass of the first link to 5 kg

# pb.changeDynamics(block_id2, -1, lateralFriction=10)
# pb.changeDynamics(block_id2, linkIndex=-1, mass=0.1)  # Change the mass of the first link to 5 kg


# Set friction for object2
pb.changeDynamics(table_id, -1, lateralFriction=0.2)

robot1.initialize(base_pos=[0.29, -0.8, 0.025], base_ori=rotation_quaternion)
robot2.initialize(base_pos=[0.09, -1, 0.067], base_ori=rotation_quaternion)

pb.setTimeStep(1/100)
# 2 arm success pickup
# robot1._setRobotiqPosition(0.22)
# robot2._setRobotiqPosition(0.025)


robot1._setRobotiqPosition(0.22)
robot2._setRobotiqPosition(0.0)

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



for i in range(step_num):

    # if i==1:
    #     pdb.set_trace()
    # robot1._setRobotiqPosition(0.22)


    joint_angle1 = robot1._calculateIK(np.array(tra1[i])+[0.0,0.0,-0.0], rotation_ee)
    joint_angle2 = robot2._calculateIK(np.array(tra2[i])+[0.0,0.0,0.0], rotation_ee)

    print("joint_angle2",np.array(joint_angle2)/np.pi*180)

    # joint_angle1[3]=joint_angle1[3]-2*np.pi
    # joint_angle1[5] = joint_angle1[5] - 2 * np.pi

    robot1._resetJointStateforce(joint_angle1)
    robot2._resetJointStateforce(joint_angle2)
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


# print(i)

# with open('joint_kinova_f2f.json', 'w') as json_file:
#     json.dump(joint_angle_list_Kinova, json_file)
#
# with open('joint_UR5e_f2f.json', 'w') as json_file:
#     json.dump(joint_angle_list_UR5e, json_file)


