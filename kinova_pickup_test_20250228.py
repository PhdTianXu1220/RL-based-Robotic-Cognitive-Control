import sys

# Path to the directory containing the package
# package_path = '/home/tianxu/Documents/DMP-python/Dual_Arm_New'
#
# # Add this path to sys.path
# if package_path not in sys.path:
#     sys.path.append(package_path)

from Robot_Sim.robots.kinova_robotiq_new import Kinova_Robotiq
# from Robot_Sim.robots.ur5_robotiq import UR5_Robotiq
#self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
import pybullet as pb
import pybullet_data
import time
import pdb
import json
import numpy as np
import matplotlib.pyplot as plt

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

# 0: lift up skill
# 1: move skill

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

        tra1 = np.vstack(( tra_go, tra_go2,tra_putdown))
        return tra1

    else:
        print("no such skill")
        return -1



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
# t_des = np.linspace(0.0, 1.0, 200)
#
# start1=[0.28667097385077456, -0.48847984909271797, 0.5940973561794143+0.02]
# end1=[0.28667097385077456-0.5, -0.48847984909271797, 0.5940973561794143+0.02]
#
# tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
# tra_go = np.array([np.linspace(start1[i], end1[i], 200) for i in range(3)]).T
#
# tra1 = np.vstack((tra_pre, tra_go))

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
end_effector_zbias=0.5940973561794143 + 0.022
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

# Define the rotation (90 degrees around Z-axis)
rotation_euler = [0, 0, 3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion = pb.getQuaternionFromEuler(rotation_euler)

# Define the rotation (90 degrees around Z-axis)
rotation_euler2 = [0, 0, -3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion2 = pb.getQuaternionFromEuler(rotation_euler2)

bias_kinova=np.array([0.0,0.0,-0.155-0.02])
bias_ur5e=np.array([0,0,0])
height_bias=np.array([0,0,-0.05])

close_thres=0.03

# block_start_position =np.array(tra1[0])+bias_kinova
block_start_position =[0.28667097385077456, -0.48847984909271797, 0.5940973561794143-0.17]
# block_start_position2 = np.array(tra2[0])+bias_ur5e
# block_start_position3 =np.array(tra3[tra1_connect_time])+bias_kinova++height_bias
# block_start_position4 = np.array(tra4[tra2_connect_time])+bias_ur5e++height_bias
block_target_position=[-0.18100409137299697, -0.49399412541519766, 0.4171282952898786]

block_start_orientation = pb.getQuaternionFromEuler([0, 0, 0])
block_id = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/magnetic.urdf", block_start_position, block_start_orientation)
table_id = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/table.urdf", basePosition=[0, -0.4, -0.22],useFixedBase=True)
# block_id2 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/magnetic.urdf", block_start_position2, block_start_orientation)
# block_id3 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/cube1.urdf", block_start_position3, block_start_orientation)
# block_id4 = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/cube2.urdf", block_start_position4, block_start_orientation)
# Change color of the base link (linkIndex = -1 for the base)
pb.changeVisualShape(block_id, linkIndex=-1, rgbaColor=[1, 0, 0, 1])

# Set friction for object1
pb.changeDynamics(block_id, -1, lateralFriction=0.5)
pb.changeDynamics(block_id, -1, rollingFriction=0.6,spinningFriction=0.1)
pb.changeDynamics(block_id, linkIndex=-1, mass=0.1)  # Change the mass of the first link to 5 kg
# pb.changeDynamics(block_id, linkIndex=-1, mass=10)  # Change the mass of the first link to 5 kg


# Set friction for object2
pb.changeDynamics(table_id, -1, lateralFriction=0.3)
pb.changeDynamics(table_id, -1, rollingFriction=1)

robot1.initialize(base_pos=[0.29, -0.8, 0.025], base_ori=rotation_quaternion)
# robot2.initialize(base_pos=[-0.0635, 0.0, 0.067], base_ori=rotation_quaternion2)

pb.setTimeStep(1/100)
robot1._setRobotiqPosition(0.22)
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
contact_flag=False
re_grip_flag=False
re_grip_wait_count=0
loose_thres=0.05
step_dist=0.03
step_dist_return=0.03

max_step =500
grip_open_steps=30
grip_close_steps=20

for i in range(max_step):

    # if i==1:
    #     pdb.set_trace()
    # robot1._setRobotiqPosition(0.22)

    if not re_grip_flag:
        joint_angle1 = robot1._calculateIK(np.array(tra1[i])+[0.0,0.0,-0.0], rotation_ee)
    elif re_grip_flag and re_grip_wait_count<grip_open_steps:
        robot1._setRobotiqPosition(0.0)
        joint_angle1 = robot1._calculateIK(next_pos, rotation_ee)
        re_grip_wait_count+=1
        print("here",next_pos)
        # pdb.set_trace()

    elif re_grip_flag and grip_open_steps<=re_grip_wait_count<grip_open_steps+grip_close_steps:
        robot1._setRobotiqPosition(0.22)
        joint_angle1 = robot1._calculateIK(next_pos, rotation_ee)
        re_grip_wait_count += 1
    elif re_grip_wait_count>=grip_open_steps+grip_close_steps:
        print("continue moving")
        joint_angle1 = robot1._calculateIK(next_pos, rotation_ee)

    robot1._resetJointStateforce(joint_angle1)
    manipulator_torque, gripper_torque = robot1._getJointStateTorque()
    gripper_signal = np.sum(np.abs(np.array(gripper_torque)))

    l = robot1._getLinkState(robot1.end_effector_index)
    end_effector_pos1=np.array(l[0])
    execution_bias=np.array([-0.00109593, - 0.0001968,  0.0])

    # print("bias:",execution_bias)
    # pdb.set_trace()


    base_position, base_orientation = pb.getBasePositionAndOrientation(block_id)
    print("block_position", base_position)
    dist_target=np.linalg.norm(np.array(base_position)-np.array(block_target_position))
    if dist_target<close_thres:
        print("target achieved")
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


print(i)

plot_joint_torque(manipulator_torque_time,'mani_torq_6.png')
plot_joint_torque(gripper_torque_time,'grip_torq_6.png')
plot_joint_torque(gripper_signal_time,'grip_signal_torq_6.png')

# with open('joint_kinova_f2f.json', 'w') as json_file:
#     json.dump(joint_angle_list_Kinova, json_file)
#
# with open('joint_UR5e_f2f.json', 'w') as json_file:
#     json.dump(joint_angle_list_UR5e, json_file)


