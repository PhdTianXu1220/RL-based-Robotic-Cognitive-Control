from Robot_Sim.robots.kinova_robotiq_new import Kinova_Robotiq
# from Robot_Sim.robots.ur5_robotiq import UR5_Robotiq
#self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
import pybullet as pb
import pybullet_data
import time
import pdb
import json
import numpy as np



# with open('kinova_tra_SAC.json', 'r') as json_file:
#     tra1 = json.load(json_file)
#
# with open('ur5e_tra_SAC.json', 'r') as json_file:
#     tra2 = json.load(json_file)

with open('Kinova_pos_pre.json', 'r') as json_file:
    tra1_pre = json.load(json_file)

with open('Kinova_ori_pre.json', 'r') as json_file:
    ori1_pre = json.load(json_file)

with open('Kinova_pos.json', 'r') as json_file:
    tra1 = json.load(json_file)

with open('Kinova_ori.json', 'r') as json_file:
    ori1 = json.load(json_file)


step_num_pre=len(tra1_pre)
step_num=len(tra1)


pb.connect(pb.GUI)

pb.setGravity(0, 0, 0)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
plane_id = pb.loadURDF("plane.urdf")

robot1 = Kinova_Robotiq()



# Define the rotation (90 degrees around Z-axis)
# rotation_euler2 = [0, 0, 0]  # Roll, Pitch, Yaw (in radians)
# rotation_quaternion2 = pb.getQuaternionFromEuler(rotation_euler2)
# Define the rotation (90 degrees around Z-axis)
rotation_euler = [0, 0, 3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion = pb.getQuaternionFromEuler(rotation_euler)

robot1.initialize(base_pos=[0.29, -0.8, 0.025], base_ori=rotation_quaternion)



pb.setTimeStep(1/100)

pdb.set_trace()

rotation_ee = [0,0,0.707,0.707]
joint_angle_list_UR5e=[]


for i in range(step_num_pre):
    print(i)
    joint_angle1 = robot1._calculateIK(np.array(tra1_pre[i])+[0.0,0.0,-0.055], ori1_pre[i])
    robot1._resetJointStateforce(joint_angle1)
    pb.stepSimulation()

    time.sleep(0.1)
    joint_angle_list_UR5e.append(joint_angle1)

pdb.set_trace()

for i in range(step_num):
    print(i)
    joint_angle1 = robot1._calculateIK(np.array(tra1[i])+[0.0,0.0,-0.055], ori1[i])
    robot1._resetJointStateforce(joint_angle1)
    pb.stepSimulation()

    time.sleep(0.1)
    joint_angle_list_UR5e.append(joint_angle1)


print("time step num:",step_num)

with open('Kinova_demo_joint_skill.json', 'w') as json_file:
    json.dump(joint_angle_list_UR5e, json_file)

print("data length",len(joint_angle_list_UR5e))

