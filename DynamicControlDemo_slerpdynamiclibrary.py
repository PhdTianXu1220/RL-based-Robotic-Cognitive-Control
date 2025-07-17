from Robot_Sim.robots.kinova_robotiq_new import Kinova_Robotiq
# from Robot_Sim.robots.ur5_robotiq import UR5_Robotiq
#self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
import pybullet as pb
import pybullet_data
import time
import pdb
import json
import numpy as np

from mapping_v2 import mapping,sclerp_dual_quaternion
from scipy.spatial.transform import Rotation as R, Slerp
from slerp_interpolate_utilities import slerp_projection,slerp_geodesic,rotation_matrix_from_vectors,transform_trajectory,interpolate_traj

# position library
start1 = [0.28667097385077456, -0.48847984909271797, 0.5940973561794143 + 0.022]
mid1 = [0.28667097385077456 - 0.25, -0.48847984909271797, 0.5940973561794143 + 0.022 + 0.15]
end1 = [0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.022]

tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
tra_go = np.array([np.linspace(start1[i], mid1[i], 100) for i in range(3)]).T
tra_go2 = np.array([np.linspace(mid1[i], end1[i], 100) for i in range(3)]).T

tra= np.vstack((tra_pre, tra_go, tra_go2))

#orientation library
rotation_euler = [0, 3.14159 / 4, 3.14159 / 4]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion = pb.getQuaternionFromEuler(rotation_euler)
quat_end=np.array(rotation_quaternion)

ori1_new=[]

# Define the rotation (90 degrees around Z-axis)
# rotation_euler2 = [0, 0, 0]  # Roll, Pitch, Yaw (in radians)
# rotation_quaternion2 = pb.getQuaternionFromEuler(rotation_euler2)
# Define the rotation (90 degrees around Z-axis)
rotation_euler = [0, 0, 3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion = pb.getQuaternionFromEuler(rotation_euler)
quat_start=np.array(rotation_quaternion)
# *Step 2: Use SLERP
# SLERP requires [w, x, y, z] for scipy, so reorder
# r_start = R.from_quat([*quat_start[:3], quat_start[3]])
# r_end = R.from_quat([*quat_end[:3], quat_end[3]])

# Step 2: Create Rotation objects
r_start = R.from_quat(quat_start)
r_end = R.from_quat(quat_end)

# Step 3: Setup SLERP
key_times = [0, 1]  # interpolation range
key_rots = R.concatenate([r_start, r_end])
slerp1 = Slerp(key_times, key_rots)

ori1_num=len(tra)
half_index=round(ori1_num/2)

# Step 4: Query interpolated quaternions
times = np.linspace(0, 1, half_index)
interp_rots = slerp1(times)
interp_quats = interp_rots.as_quat()

interp_quats1=interp_quats

# Step 3: Setup SLERP
key_times = [0, 1]  # interpolation range
key_rots = R.concatenate([r_end, r_start])
slerp2 = Slerp(key_times, key_rots)

# Step 4: Query interpolated quaternions
times = np.linspace(0, 1, ori1_num-half_index)
interp_rots = slerp2(times)

interp_quats = interp_rots.as_quat()
# interp_quats=interp_quats[1:]

print("rotation quat",slerp2(1.0).as_quat())

interp_quats2=interp_quats
ori1_new = np.vstack([interp_quats1, interp_quats2])

task_start_pos=np.array([0.28667097385077456, -0.48847984909271797, 0.5940973561794143 + 0.022])
task_start_ori=quat_start
task_end_pos=np.array([0.28667097385077456 - 0.3, -0.48847984909271797-0.2, 0.5940973561794143 + 0.022+0.02])
task_end_ori=slerp2(0.7)
print(type(task_end_ori))

t_star, q3 = slerp_projection(quat_end, quat_start, task_end_ori.as_quat())
print("t_star",t_star)

tra_orientation_pos=interpolate_traj(tra,0.5+t_star/2)
tra_tmp=tra
tra_tmp[-1]=np.array(tra_orientation_pos)

transformed_tra = transform_trajectory(tra_tmp, task_start_pos, task_end_pos)


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

bias=np.array([0.0,0.0,-0.255])

for i in range(len(tra)):
    print(i)
    joint_angle1 = robot1._calculateIK(np.array(tra[i])+bias, ori1_new[i])
    # joint_angle1 = robot1._calculateIK(np.array(transformed_tra[i]) + bias, ori1_new[i])
    robot1._resetJointStateforce(joint_angle1)
    pb.stepSimulation()
    l = robot1._getLinkState(robot1.end_effector_index)
    end_effector_pos1 = np.array(l[0])

    dist_target = np.linalg.norm(np.array(end_effector_pos1) - np.array(tra_orientation_pos)-bias)

    # if dist_target<0.05:
    #     print("dist_target",dist_target)
    #     break
    if i==round(len(tra)*0.65):
        print("dist_target", dist_target)
        break


    time.sleep(0.1)


pdb.set_trace()



