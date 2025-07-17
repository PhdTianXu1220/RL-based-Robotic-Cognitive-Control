# FUNCTION DEFINITIONS
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# from screw_linear_interpolation import sclerp_dual_quaternion
pi = np.pi


# Function to convert from euler angles to quaternions
def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qw, qx, qy, qz]


# Function to write the quaternion in the cos(theta/2)+usin(theta/2) form
def extract_from_unitquaternion(q):
    costheta2 = q[0]
    sintheta2 = (1 - costheta2 ** 2) ** 0.5
    u = [q[1] / sintheta2, q[2] / sintheta2, q[3] / sintheta2]
    return [costheta2, sintheta2, u]


# Function to convert  quaternions to euler angles
def quaternion_to_euler(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))
    X = X / 180 * pi

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    Y = Y / 180 * pi

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    Z = Z / 180 * pi

    return [X, Y, Z]


# Function for vector to quaternion
def vector_to_quaternion(v):
    q = [0, v[0], v[1], v[2]]
    return q


# Function for quaternion product
def quaternion_multiply(q1, q2):
    w0 = q1[0]
    x0 = q1[1]
    y0 = q1[2]
    z0 = q1[3]

    w1 = q2[0]
    x1 = q2[1]
    y1 = q2[2]
    z1 = q2[3]

    wp = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    xp = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    yp = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    zp = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    return [wp, xp, yp, zp]


# Function to form a dual-quaternion in a format of [[qr],[qd]]
def dual_quaternion_form(qrot, qtra):
    t = np.array(quaternion_multiply(qtra, qrot)) * 0.5
    return [qrot, [t[0], t[1], t[2], t[3]]]


# Function for dual-quaternion product
def dual_quaternion_multiply(dq1, dq2):
    qr = quaternion_multiply(dq1[0], dq2[0])
    qd = np.array(quaternion_multiply(dq1[1], dq2[0])) + np.array(quaternion_multiply(dq1[0], dq2[1]))

    return [qr, [qd[0], qd[1], qd[2], qd[3]]]


# Function to calculate the conjugate of a given quaternion
def quaternion_conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]


# Function to calculate the conjugate of a given dual-quaternion
def dual_quaternion_conjugate(dq):
    return [[dq[0][0], -dq[0][1], -dq[0][2], -dq[0][3]], [dq[1][0], -dq[1][1], -dq[1][2], -dq[1][3]]]


# Function to subtract one vector from another
def vector_subtraction(v1, v2):
    return [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]


# Function to add one vector to another
def vector_add(v1, v2):
    return [v2[0] + v1[0], v2[1] + v1[1], v2[2] + v1[2]]


# Function to get the length of a 3d vector
def vector_length(v):
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


# Function to normalize a vector
def vector_norm(v):
    vector_len = vector_length(v)
    return [v[0] / vector_len, v[1] / vector_len, v[2] / vector_len]


# Function to calculate the cross product of two vectors
def vector_cross(v1, v2):
    return [(v1[1] * v2[2] - v1[2] * v2[1]), (v1[2] * v2[0] - v1[0] * v2[2]), (v1[0] * v2[1] - v1[1] * v2[0])]


# Function to calculate the dot product of two vectors
def vector_dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


# Function to convert dual-quaternion to screw parameters
def screw_parameters(dq):
    wr = dq[0][0]
    wd = dq[1][0]
    vr = dq[0][1:]
    vd = dq[1][1:]

    theta = np.arccos(wr) * 2
    if theta == 0:
        l = vector_norm(vd)
        m = [0, 0, 0]
        d = vector_length(vd) * 2
    else:
        l = np.array(vr) * (1 / np.sin(theta / 2))
        d = -1 * np.array(wd) * 2 / np.sin(theta / 2)
        m = np.array(vd) * (1 / np.sin(theta / 2)) - 0.5 * d * (1 / np.tan(theta / 2)) * l

    return [theta, l, d, m]


# Function to calculate dual-quaternion power
def dual_quaternion_power(dq, tau):
    parameters = screw_parameters(dq)
    theta = parameters[0]
    l = parameters[1]
    d = parameters[2]
    m = parameters[3]
    dqr = vector_to_quaternion(np.array(l) * np.sin(0.5 * theta * tau))
    dqr[0] = np.cos(0.5 * theta * tau)
    dqd = vector_to_quaternion(
        np.array(m) * np.sin(0.5 * theta * tau) + np.array(l) * np.cos(0.5 * theta * tau) * 0.5 * tau * d)
    dqd[0] = np.sin(0.5 * theta * tau) * (-0.5 * tau * d)
    return [dqr, dqd]


# Function to determine the interpolated point between two dual_quaternion
def sclerp_dual_quaternion(dq1, dq2, tau):
    d = dual_quaternion_multiply(dual_quaternion_conjugate(dq1), dq2)
    dtau = dual_quaternion_power(d, tau)
    ctau = dual_quaternion_multiply(dq1, dtau)
    return ctau


# # Another way to calculate the fractional power of a dual-quaternion
# def dual_quaternion_tau(dq, tau):
#     theta = np.arccos(extract_from_unitquaternion(dq[0])[0])
#     u = extract_from_unitquaternion(dq[0])[2]
#     p = np.array(quaternion_multiply(dq[1], quaternion_conjugate(dq[0]))) * 2
#     p = p[2:stop]
#     d = vector_dot(p, [1, 1, 1])
#     m = np.array(vector_cross(p, u) + np.array(vector_subtraction(np.array(u) * d * tau, p))) * (1 / np.tan(theta))
#     dr = [np.cos(tau * theta), np.array(extract_from_unitquaternion(dq[0])[2]) * np.sin(tau * theta)]
#     dd = [-1 / 2 * d * tau * np.sin(tau * theta),
#           np.array(m * np.sin(tau * theta)) + np.array(np.array(u) * 1 / 2 * d * tau * np.cos(tau * theta))]
#     return [dr, dd]


# Function to create a rotation matrix between two vectors
def rotation_matrix(v1, v2):
    costheta = (vector_dot(v1, v2) / (vector_length(v1) * vector_length(v2)))
    sintheta = (vector_length(vector_cross(v1, v2)) / (vector_length(v1) * vector_length(v2)))
    w = vector_cross(v1, v2)
    w = [w[0] / vector_length(w), w[1] / vector_length(w), w[2] / vector_length(w)]
    return [[((w[0] ** 2) * (1 - costheta) + costheta), (w[0] * w[1] * (1 - costheta) - w[2] * sintheta),
             (w[0] * w[2] * (1 - costheta) + w[1] * sintheta)],
            [(w[0] * w[1] * (1 - costheta) + w[2] * sintheta), ((w[1] ** 2) * (1 - costheta) + costheta),
             (w[1] * w[2] * (1 - costheta) + w[0] * sintheta)],
            [(w[0] * w[2] * (1 - costheta) + w[1] * sintheta), (w[1] * w[2] * (1 - costheta) + w[0] * sintheta),
             ((w[2] ** 2) * (1 - costheta) + costheta)]]


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions


def mapping(waypoints, orientations, startPosition, endPosition, orientations_goal, inter_number):
    dr1 = []
    # Convert orientations from euler angles to quaternions
    for i in range(len(orientations)):
        dr1.append(euler_to_quaternion(orientations[i][0], orientations[i][1], orientations[i][2]))

    # Calculate the delta between each configuration and the end configuration
    deltadr1 = []
    for i in range(len(dr1) - 1):
        new_deltadr1 = quaternion_multiply(quaternion_conjugate(dr1[i]), dr1[len(dr1) - 1])
        deltadr1.append(new_deltadr1)

    # Calculate the new orientations based on the goal end orientation and the deltas
    orientations_new = []
    for i in range(len(deltadr1)):
        orientations_new.append(
            quaternion_multiply(euler_to_quaternion(orientations_goal[0], orientations_goal[1], orientations_goal[2]),
                                quaternion_conjugate(deltadr1[i])))
    orientations_new.append(euler_to_quaternion(orientations_goal[0], orientations_goal[1], orientations_goal[2]))

    # Rotate and scale the path to new goal and start positions
    gsvector = vector_subtraction(waypoints[len(waypoints) - 1], waypoints[0])
    gsvector_new = vector_subtraction(endPosition, startPosition)
    rot_mat = rotation_matrix_from_vectors(gsvector, gsvector_new)

    waypoints_new = []
    waypoints_new.append(startPosition)
    for i in range(len(waypoints) - 1):
        vectori = vector_subtraction(waypoints[i], waypoints[i + 1])
        waypoints_new.append(vector_add(waypoints_new[i],
                                        np.array(rot_mat.dot(vectori)) * vector_length(gsvector_new) / vector_length(
                                            gsvector)))

    dual_a = []
    for i in range(len(waypoints_new)):
        at = vector_to_quaternion(waypoints_new[i])
        ar = orientations_new[i]
        dual_a.append(dual_quaternion_form(ar, at))

    finetune_dual = []
    for i in range(len(dual_a) - 1):
        for tau in np.arange(0, 1, 1 / inter_number):
            finetune_dual.append(sclerp_dual_quaternion(dual_a[i], dual_a[i + 1], tau))
    finetune_dual.append(dual_a[-1])

    task_space_output = []

    for i in range(len(finetune_dual)):
        qr = finetune_dual[i][0]
        qd = finetune_dual[i][1]
        qt = np.array(quaternion_multiply(qd, quaternion_conjugate(qr))) * 2
        fine_tune_position = [float(qt[1]), float(qt[2]), float(qt[3])]
        fine_tune_orientation = quaternion_to_euler(qr)
        fine_tune_orientation = [float(fine_tune_orientation[0]), float(fine_tune_orientation[1]),
                                 float(fine_tune_orientation[2])]
        fine_tune_all = fine_tune_position + fine_tune_orientation
        task_space_output.append(fine_tune_all)
        print(task_space_output[i])

    return task_space_output


# MAIN BODY

# # Inputed waypoints for the demonstrated path
# waypoints = [[-0.2, -0.1, 0.0],
#              [-0.2, -0.1, 0.2],
#              [-0.2, 0.1, 0]]
# np.savetxt("waypoints.txt", waypoints, fmt="%f", delimiter=",")
# orientations = [[0, pi / 2, 0],
#                 [0, pi / 2, 0],
#                 [0, pi / 4, 0]]
# demonstration = [[-0.2, -0.1, 0.0, 0, pi / 2, 0], [-0.2, -0.1, 0.2, 0, pi / 2, 0], [-0.2, 0.1, 0, 0, pi / 4, 0]]
# np.savetxt("demonstration.txt", demonstration, fmt="%f", delimiter=",")
#
#
# with open('demonstration.txt', 'r') as f:
#     l = [[float(num) for num in line.split(',')] for line in f]
# print(l)
# waypoints = []
# orientations = []
# for i in range(len(l)):
#     waypoints.append(l[i][0:3])
#     orientations.append(l[i][3:])
#
# fig1 = plt.figure()
# ax = fig1.add_subplot(projection='3d')
# x1data = []
# y1data = []
# z1data = []
# for j in range(len(waypoints)):
#     x1data.append(waypoints[j][0])
#     y1data.append(waypoints[j][1])
#     z1data.append(waypoints[j][2])
#
# ax.scatter3D(x1data, y1data, z1data)
#
# # The desired start and end waypoints
# startPosition = [-0.2, -0.1, 0.0, 0, pi/2, 0]
# endPosition = [-0.3, 0.2, 0, 0, pi/4, 0]
# newtask = [startPosition, endPosition]
# np.savetxt("newtask.txt", newtask, fmt="%f", delimiter=",")
# with open('newtask.txt', 'r') as fk:
#     li = [[float(num) for num in line.split(',')] for line in fk]
#
# startPosition = []
# startOrientation = []
# goalPosition = []
# goalOrientation = []
# startPosition = li[0][0:3]
# startOrientation = li[0][3:]
# goalPosition = li[1][0:3]
# goalOrientation = li[1][3:]
# inter_number = 20
#
# result = mapping(waypoints, orientations, startPosition, goalPosition, goalOrientation, inter_number)
#
# fig2 = plt.figure()
# ax = fig2.add_subplot(projection='3d')
# xdata = []
# ydata = []
# zdata = []
# for i in range(len(result)):
#     xdata.append(result[i][0])
#     ydata.append(result[i][1])
#     zdata.append(result[i][2])
#
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
#
# # Brushing Demo
# with open('brushing_task1.txt', 'r') as bt:
#     task = [[float(num) for num in line.split(',')] for line in bt]
# print("task = ", task)
#
# with open('brushing_demo1.txt', 'r') as bd:
#     demo = [[float(num) for num in line.split(',')] for line in bd]
# print("demo= ", demo)
#
# # inter_number = 20
# # brush_new = []
# # for iteration in range(len(task) - 1):
# #     startPosition = []
# #     startOrientation = []
# #     goalPosition = []
# #     goalOrientation = []
# #     waypoints = []
# #     orientations = []
# #     for i in range(iteration, iteration + 2):
# #         waypoints.append(demo[i][0:3])
# #         orientations.append(demo[i][3:])
# #
# #     print("waypoints = ", waypoints)
# #     print("orientations = ", orientations)
# #     startPosition = task[iteration][0:3]
# #     startOrientation = task[iteration][3:]
# #     goalPosition = task[iteration + 1][0:3]ee_euler
# #     goalOrientation = task[iteration + 1][3:]
# #     output = []
# #     output = mapping(waypoints, orientations, startPosition, goalPosition, goalOrientation, inter_number)
# #     print("output = ", len(output))
# #     brush_new.append(output)
# #
# # brush_result = np.vstack((brush_new[0],brush_new[1]))
# # # print()
# # # np.savetxt("file1.txt", Array, fmt="%2,4f", delimiter=",")
# # np.savetxt("file2.txt", brush_result, fmt="%f", delimiter=",")
# #
# # fig3 = plt.figure()
# # ax3 = fig3.add_subplot(projection='3d')
# # xdata = []
# # ydata = []
# # zdata = []
# # for i in range(len(brush_result)):
# #     xdata.append(brush_result[i][0])
# #     ydata.append(brush_result[i][1])
# #     zdata.append(brush_result[i][2])
# #
# # ax3.scatter3D(xdata, ydata, zdata)
# plt.show()
