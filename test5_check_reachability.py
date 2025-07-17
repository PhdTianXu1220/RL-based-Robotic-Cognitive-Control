import os
import math
import numpy as np
import gym
import pybullet as pybullet
import pybullet_data
import time
from numpy import linalg as LA
from tip_ee import ee_to_tip, tip_to_ee

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
r2000_i_j4 = -1.57
r2000_i_j5 = -1.57
joint_positions_r2000_i = np.array([r2000_i_j0, r2000_i_j1, r2000_i_j2, r2000_i_j3, r2000_i_j4, r2000_i_j5])
for joint_index_r2000_i in range(0, 6):
    pybullet.resetJointState(r2000_i, joint_index_r2000_i, joint_positions_r2000_i[joint_index_r2000_i])

time.sleep(3)

ee_state = pybullet.getLinkState(r2000_i, 5, computeLinkVelocity=0)
ee_position = ee_state[0]
ee_orientation = ee_state[1]

print("ee_position = ", ee_position)
print("ee_orientation= ", ee_orientation)
com_trn = pybullet.getLinkState(r2000_i, 5, computeLinkVelocity=0)[2]
mpos = [r2000_i_j0, r2000_i_j1, r2000_i_j2, r2000_i_j3, r2000_i_j4, r2000_i_j5]
jacobian = pybullet.calculateJacobian(r2000_i, 5, [0, 0, 0], mpos,
                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

jacobian_tr = np.array(jacobian[0])
jacobian_rot = np.array(jacobian[1])
jacobian_matrix = np.concatenate((jacobian_tr, jacobian_rot), axis=0)
mani = np.linalg.det(jacobian_matrix)
initial_reachability = math.sqrt(abs(mani))
print("reach_initial", math.sqrt(abs(mani)))

contact_points = pybullet.getClosestPoints(r2000_i, env, 0.0)
print("contact_info = ", contact_points)
print("number_of_contact_points = ", len(contact_points) - 1)

if len(contact_points) - 1 != 0:
    for i in range(len(contact_points) - 1):
        print("contact_position = ", contact_points[i + 1][5])


def quaternion_mult(q, r):
    return [r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
            r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
            r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
            r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]]


def rotation_by_quaternion(point, q):
    r = [0] + point
    q_conj = [q[0], -1 * q[1], -1 * q[2], -1 * q[3]]
    return quaternion_mult(quaternion_mult(q, r), q_conj)[1:]


def normalized_vector(vector):
    mag2 = sum(n * n for n in vector)
    mag = math.sqrt(mag2)
    normalize_vector = [n / mag for n in vector]
    return normalize_vector


def quaternion_from_angle(vector, theta):
    vector = normalized_vector(vector)
    x, y, z = vector
    w = math.cos(theta / 2.)
    x = x * math.sin(theta / 2.)
    y = y * math.sin(theta / 2.)
    z = z * math.sin(theta / 2.)
    return [w, x, y, z]


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


unit_vector_rot = rotation_by_quaternion([1, 0, 0], np.array(ee_orientation))
tip_position = np.array(ee_position) - np.array(unit_vector_rot) * 0.265

print("tip_position = ", tip_position)

# visual_tip = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
#                                         radius=0.1,
#                                         rgbaColor=[0, 0, 1, 0.5],
#                                         specularColor=[0.4, 0.0, 0.4],
#                                         visualFramePosition=np.array(tip_position))
# tip_create = pybullet.createMultiBody(baseVisualShapeIndex=visual_tip,
#                                       physicsClientId=physics_clientid)

# tip_position = [0.17, 0., 0.97]
# target_orientation = pybullet.getQuaternionFromEuler([0, np.pi / 4, -np.pi / 2])
# # target_position = link_state[0]
# # target_orientation = link_state[1]
#
# unit_vector_rot = rotation_by_quaternion([1, 0, 0], target_orientation)
# print("rotated_unit_vector = ", unit_vector_rot)
#
# target_position = np.array(tip_position) + np.array(unit_vector_rot) * 0.265
# ik = pybullet.calculateInverseKinematics(r2000_i, 5, target_position, target_orientation)
# ik = np.array(ik)
# print("ik_jointangles = ", ik)
#
# for joint_index_r2000_i in range(0, 6):
#     pybullet.resetJointState(r2000_i, joint_index_r2000_i, ik[joint_index_r2000_i])

# visual_tip = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
#                                         radius=0.1,
#                                         rgbaColor=[1, 0, 0, 0.5],
#                                         specularColor=[0.4, 0.0, 0.4],
#                                         visualFramePosition=np.array(tip_position))
# tip_create = pybullet.createMultiBody(baseVisualShapeIndex=visual_tip,
#                                       physicsClientId=physics_clientid)

# time.sleep(500)
jointsLL = [0., 0., 0., 0., 0., 0.]

jointsUL = [0., 0., 0., 0., 0., 0.]

for i in range(6):
    jointsInfo = pybullet.getJointInfo(r2000_i, i)
    jointsLL[i] = jointsInfo[8]
    jointsUL[i] = jointsInfo[9]

tray_dimension = [0.228600, 0.304800, 0.044941]
tray_center = [0.177178, -0.00300, 0.878144]
# turn_table = np.pi/2 - angle_between([0.177178, -0.00300, 0], [0.368229, -0.473057, 0]) + np.pi/2
turn_table = 5 * np.pi / (2 * 9)
pitch_lim = [np.pi / 4, np.pi / 2]
yaw_lim = [-np.pi / 2, 0.0]

specific_pith = np.pi / 4
specific_yaw = -np.pi / 2
angle_counter = 0
for pitch_angle in np.arange(pitch_lim[0], pitch_lim[1], 0.1):
    for yaw_angle in np.arange(yaw_lim[0], yaw_lim[1], 0.1):
        angle_counter += 1
# specific_pith = 10
# specific_yaw = 10
desired_z = 0.10
brush_len = 0.265
ik_solution_tolerance = 0.01
## Calculate the dimension of the searching regionvoxel_position

serach_xlim = [0.0, 0.0]
serach_xlim[0] = tray_center[0] - tray_dimension[0] / 2
serach_xlim[1] = tray_center[0] + tray_dimension[0] / 2
serach_ylim = [0.0, 0.0]
serach_ylim[0] = tray_center[1] - tray_dimension[1] / 2
serach_ylim[1] = tray_center[1] + tray_dimension[1] / 2
serach_zlim = [0.0, 0.0]
serach_zlim[0] = tray_center[2] + tray_dimension[2] / 2
serach_zlim[1] = tray_center[2] + tray_dimension[2] / 2 + desired_z

print(serach_xlim)
print(serach_ylim)
print(serach_zlim)
reachabilitys = []
average_manipulabilities = []
computing_time = []
# for divide in range(10):
start = time.time()
# turn_table = divide * np.pi / (2 * 9)
turn_table = 1 * np.pi / (2 * 9)
specific_yaw = -np.pi / 4
point_count = 0
manipulability = []
average_manipulabilities = []
reachability_counter = 0

for x_position in np.arange(serach_xlim[0], serach_xlim[1], 0.02):
    for y_position in np.arange(serach_ylim[0], serach_ylim[1], 0.02):
        for z_position in np.arange(serach_zlim[0], serach_zlim[1], 0.02):
            voxel_position = [x_position, y_position, z_position]
            if turn_table != 0:
                vector_to_center_of_tray = [x_position - tray_center[0], y_position - tray_center[1], 0]
                rotate_axis = [0, 0, 1]
                rotate_angle = turn_table
                vector_to_center_of_tray_turn = rotation_by_quaternion(vector_to_center_of_tray,
                                                                       quaternion_from_angle(rotate_axis,
                                                                                             rotate_angle))
                voxel_position = [tray_center[0] + vector_to_center_of_tray_turn[0],
                                  tray_center[1] + vector_to_center_of_tray_turn[1],
                                  voxel_position[2]]
                # specific_yaw = -np.pi / 2 + rotate_angle
            if specific_pith == 10:
                ik_count = 0
                manipulability = []
                point_count += 1
                for pitch_angle in np.arange(pitch_lim[0], pitch_lim[1], 0.1):
                    for yaw_angle in np.arange(yaw_lim[0], yaw_lim[1], 0.1):
                        voxel_orientation = pybullet.getQuaternionFromEuler([0, pitch_angle, yaw_angle])
                        # voxel_orientation_q = [voxel_orientation[3], voxel_orientation[0], voxel_orientation[1],
                        #                        voxel_orientation[2]]
                        # unit_vector_rot = rotation_by_quaternion([1, 0, 0], voxel_orientation_q)
                        # voxel_ee_position = np.array(voxel_position) - np.array(unit_vector_rot) * 0.265
                        voxel_ee_position = tip_to_ee(voxel_position, voxel_orientation, brush_len)
                        voxel_ik = pybullet.calculateInverseKinematics(r2000_i, 5, voxel_ee_position,
                                                                       voxel_orientation,
                                                                       jointsLL, jointsUL, maxNumIterations=10000)
                        for joint_index_r2000_i in range(0, 6):
                            pybullet.resetJointState(r2000_i, joint_index_r2000_i,
                                                     voxel_ik[joint_index_r2000_i])

                        voxel_ee_fk = np.array(pybullet.getLinkState(r2000_i, 5, computeLinkVelocity=0)[0])
                        # voxel_fk = voxel_ee_fk + np.array(unit_vector_rot) * 0.265
                        voxel_fk = ee_to_tip(voxel_ee_fk, voxel_orientation, brush_len)
                        distance = LA.norm(voxel_fk - np.array(voxel_position))
                        contact_points = pybullet.getClosestPoints(r2000_i, env, 0.0)
                        if distance <= ik_solution_tolerance:
                            if len(contact_points) - 2 <= 0:
                                ik_count += 1

                        com_trn = pybullet.getLinkState(r2000_i, 5, computeLinkVelocity=0)[2]
                        mpos = [voxel_ik[0], voxel_ik[1], voxel_ik[2], voxel_ik[3], voxel_ik[4], voxel_ik[5]]
                        jacobian = pybullet.calculateJacobian(r2000_i, 5, [0, 0, 0], mpos,
                                                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                        jacobian_tr = np.array(jacobian[0])
                        jacobian_rot = np.array(jacobian[1])
                        jacobian_matrix = np.concatenate((jacobian_tr, jacobian_rot), axis=0)
                        mani = math.sqrt(abs(np.linalg.det(jacobian_matrix)))
                        manipulability.append(mani)
                mean_manipulabiltiy = np.average(manipulability)
                average_manipulabilities.append(mean_manipulabiltiy)

                print("point ", point_count)
                print("ik_count = ", ik_count)
                print("manipulability_average = ", mean_manipulabiltiy)

                if ik_count / angle_counter <= 0.4:
                    blue = 1.0
                elif 0.4 < ik_count / angle_counter < 0.5:
                    blue = 0.5
                else:
                    blue = 0.0

                if ik_count / angle_counter <= 0.1 or ik_count / angle_counter >= 0.9:
                    green = 0.0
                elif 0.3 <= ik_count / angle_counter <= 0.7:
                    green = 1.0
                else:
                    green = 0.5

                if ik_count / angle_counter <= 0.5:
                    red = 0.0
                elif 0.5 < ik_count / angle_counter <= 0.6:
                    red = 0.5
                elif ik_count / angle_counter >= 0.9:
                    if mean_manipulabiltiy / initial_reachability <= 0.5:
                        red = 0.0
                    elif mean_manipulabiltiy / initial_reachability <= 0.6:
                        red = 0.5
                    else:
                        red = 1.0

                    if mean_manipulabiltiy / initial_reachability <= 0.4:
                        blue = 1.0
                    elif mean_manipulabiltiy / initial_reachability < 0.5:
                        blue = 0.5
                    else:
                        blue = 0.0

                    if mean_manipulabiltiy / initial_reachability <= 0.1 or mean_manipulabiltiy / initial_reachability >= 0.9:
                        green = 0.0
                    elif 0.3 <= mean_manipulabiltiy / initial_reachability <= 0.7:
                        green = 1.0
                    else:
                        green = 0.5
                else:
                    red = 1.0

                checking_position = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                               radius=0.01,
                                                               rgbaColor=[red, green, blue, 0.5],
                                                               specularColor=[0.4, 0.0, 0.4],
                                                               visualFramePosition=np.array([x_position, y_position,
                                                                                             z_position]))
                checking_position_create = pybullet.createMultiBody(baseVisualShapeIndex=checking_position,
                                                                    physicsClientId=physics_clientid)
                reset_jointp = joint_positions_r2000_i
                reset_jointp[0] = -np.pi / 2 + turn_table
                for joint_index_r2000_i in range(0, 6):
                    pybullet.resetJointState(r2000_i, joint_index_r2000_i,
                                             joint_positions_r2000_i[joint_index_r2000_i])
            else:
                voxel_orientation = pybullet.getQuaternionFromEuler([0, specific_pith, specific_yaw])
                # voxel_orientation_q = [voxel_orientation[3], voxel_orientation[0], voxel_orientation[1],
                #                        voxel_orientation[2]]
                # unit_vector_rot = rotation_by_quaternion([1, 0, 0], voxel_orientation_q)
                # voxel_ee_position = np.array(voxel_position) - np.array(unit_vector_rot) * 0.265
                voxel_ee_position = tip_to_ee(voxel_position, voxel_orientation, brush_len)
                voxel_ik = pybullet.calculateInverseKinematics(r2000_i, 5, voxel_ee_position, voxel_orientation,
                                                               jointsLL, jointsUL, maxNumIterations=10000)
                voxel_ik = np.array(voxel_ik)
                for joint_index_r2000_i in range(0, 6):
                    pybullet.resetJointState(r2000_i, joint_index_r2000_i,
                                             voxel_ik[joint_index_r2000_i])

                voxel_ee_fk = np.array(pybullet.getLinkState(r2000_i, 5, computeLinkVelocity=0)[0])
                com_trn = pybullet.getLinkState(r2000_i, 5, computeLinkVelocity=0)[2]
                mpos = [voxel_ik[0], voxel_ik[1], voxel_ik[2], voxel_ik[3], voxel_ik[4], voxel_ik[5]]
                jacobian = pybullet.calculateJacobian(r2000_i, 5, [0, 0, 0], mpos,
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                jacobian_tr = np.array(jacobian[0])
                jacobian_rot = np.array(jacobian[1])
                jacobian_matrix = np.concatenate((jacobian_tr, jacobian_rot), axis=0)
                mani = math.sqrt(abs(np.linalg.det(jacobian_matrix)))

                # voxel_fk divide * np.pi / (2 * 9)= voxel_ee_fk + np.array(unit_vector_rot) * 0.265
                voxel_fk = ee_to_tip(voxel_ee_fk, voxel_orientation, brush_len)
                distance = LA.norm(voxel_fk - np.array(voxel_position))
                contact_points = pybullet.getClosestPoints(r2000_i, env, 0.0)

                if distance <= ik_solution_tolerance and len(contact_points) - 1 <= 0:
                    if mani / initial_reachability <= 0.5:
                        red = 0.0
                    elif mani / initial_reachability <= 0.6:
                        red = 0.5
                    else:
                        red = 1.0

                    if mani / initial_reachability <= 0.4:
                        blue = 1.0
                    elif mani / initial_reachability < 0.5:
                        blue = 0.5
                    else:
                        blue = 0.0

                    if mani / initial_reachability <= 0.1 or mani / initial_reachability >= 0.9:
                        green = 0.0
                    elif 0.3 <= mani / initial_reachability <= 0.7:
                        green = 1.0
                    else:
                        green = 0.5
                else:
                    mani = 0
                    red = 0
                    green = 0
                    blue = 1
                print(mani)
                manipulability.append(mani)
                if mani != 0:
                    reachability_counter += 1

                checking_position = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE, radius=0.01,
                                                               rgbaColor=[red, green, blue, 0.5],
                                                               specularColor=[0.4, 0.0, 0.4],
                                                               visualFramePosition=np.array(voxel_position))
                checking_position_create = pybullet.createMultiBody(baseVisualShapeIndex=checking_position,
                                                                    physicsClientId=physics_clientid)
                reset_jointp = joint_positions_r2000_i
                reset_jointp[0] = -np.pi / 2 + turn_table
                for joint_index_r2000_i in range(0, 6):
                    pybullet.resetJointState(r2000_i, joint_index_r2000_i,
                                             joint_positions_r2000_i[joint_index_r2000_i])

print("mean_manipulability", np.average(manipulability))
print("rechability portion", reachability_counter / len(manipulability))
reachabilitys.append(reachability_counter / len(manipulability))
average_manipulabilities.append(np.average(manipulability))
end = time.time()
computing_time.append(end - start)

print(reachabilitys)
print(average_manipulabilities)
time.sleep(500)
