import os
import numpy as np
import gym
import pybullet as pybullet
import pybullet_data
import time

physics_clientid = pybullet.connect(pybullet.GUI)

pybullet.setGravity(0, 0, 0)

basepath = os.path.abspath(os.getcwd())

print("basepath is ", basepath)

robot = "arm.urdf"
work_cell = "ge_cell.urdf"
env = pybullet.loadURDF(work_cell, useFixedBase=1, physicsClientId=physics_clientid)

r2000_i = pybullet.loadURDF(robot, [-0.193519, 0.474939, 0.66437], [0, 0, 0, 1], useFixedBase=1,
                            flags=pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_MERGE_FIXED_LINKS)
r2000_i_j0 = 0.0
r2000_i_j1 = 0.0
r2000_i_j2 = 0.0
r2000_i_j3 = 0.0
r2000_i_j4 = 0.0
r2000_i_j5 = 0.0
# r2000_i_j4 = -1.57
# r2000_i_j5 = -1.57
joint_positions_r2000_i = np.array([r2000_i_j0, r2000_i_j1, r2000_i_j2, r2000_i_j3, r2000_i_j4, r2000_i_j5])
print("joint_positions = ", joint_positions_r2000_i)

for joint_index_r2000_i in range(0, 6):
    pybullet.resetJointState(r2000_i, joint_index_r2000_i, joint_positions_r2000_i[joint_index_r2000_i])

jointsLL = [0., 0., 0., 0., 0., 0.]

jointsUL = [0., 0., 0., 0., 0., 0.]

for i in range(6):
    jointsInfo = pybullet.getJointInfo(r2000_i, i)
    jointsLL[i] = jointsInfo[8]
    jointsUL[i] = jointsInfo[9]

print("jointsLL = ", jointsLL)
print("jointsUL = ", jointsUL)


##
def quaternion_mult(q, r):
    return [r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
            r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
            r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
            r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]]


def rotation_by_quaternion(point, q):
    r = [0] + point
    q_conj = [q[0], -1 * q[1], -1 * q[2], -1 * q[3]]
    return quaternion_mult(quaternion_mult(q, r), q_conj)[1:]


##
reachable_position = []
count = []
reachable_index = 0
for episode in range(100000):
    print("episode = ", episode)
    joints_pos_orig = [0., 0., 0., 0., 0., 0.]
    for i in range(6):
        joints_pos_orig[i] = np.random.uniform(jointsLL[i], jointsUL[i])

    for joint_index_r2000_i in range(0, 6):
        pybullet.resetJointState(r2000_i, joint_index_r2000_i, joints_pos_orig[joint_index_r2000_i])

    contact_points = pybullet.getClosestPoints(r2000_i, env, 0.0)
    # print("contact_info = ", contact_points)
    # print("number_of_contact_points = ", len(contact_points) - 1)

    if len(contact_points) - 1 == 0:
        rch_state = pybullet.getLinkState(r2000_i, 5, computeLinkVelocity=0)
        rp = np.array(rch_state[0])
        ro = np.array(rch_state[1])
        ro = [ro[0], ro[1], ro[2]]
        rp = [rp[0], rp[1], rp[2]]
        target_orientation = pybullet.getQuaternionFromEuler(ro)
        unit_vector_rot = rotation_by_quaternion([1, 0, 0], target_orientation)
        tip_position = np.array(rp) - np.array(unit_vector_rot) * 0.265
        tip_position = np.round(tip_position, 1)
        tip_position = [tip_position[0], tip_position[1], tip_position[2]]
        if tip_position not in reachable_position:
            reachable_position.append(tip_position)
            count.append(1)
            reachable_index += 1
        else:
            count[reachable_position.index(tip_position)] += 1

np.savetxt("reachable_position.txt", reachable_position, fmt="%.2f", delimiter=",")
np.savetxt("count.txt", count, fmt="%.2f", delimiter=",")

with open('count.txt', 'r') as f:
    counter = [[float(num) for num in line.split(',')] for line in f]
# print("mas_counter = ", max(counter))
with open('reachable_position.txt', 'r') as fl:
    reachable_position = [[float(num) for num in line.split(',')] for line in fl]
print("lenth = ", len(reachable_position))

reachability_map_visual = []
for i in range(len(reachable_position)):
    print(i)
    if counter[i][0] >= 9:
        counter[i][0] = 9

    if counter[i][0] <= 5:
        red = 0.0
    elif counter[i][0] == 6:
        red = 0.5
    else:
        red = 1.0

    if counter[i][0] <= 3:
        blue = 1.0
    elif counter[i][0] == 4:
        blue = 0.5
    else:
        blue = 0.0

    if counter[i][0] <= 1 or counter[i][0] >= 9:
        green = 0.0
    elif 3 <= counter[i][0] <= 7:
        green = 1.0
    else:
        green = 0.5

    reachability_map = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                  radius=0.1,
                                                  rgbaColor=[red, green, blue, 0.5],
                                                  specularColor=[red, 0.0, blue],
                                                  visualFramePosition=np.array(reachable_position[i]))
    reachability_map_visual.append(pybullet.createMultiBody(baseVisualShapeIndex=reachability_map,
                                                            physicsClientId=physics_clientid))

time.sleep(500)
