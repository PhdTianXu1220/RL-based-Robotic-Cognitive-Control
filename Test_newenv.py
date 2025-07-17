from datetime import datetime
import os, shutil
import argparse
import torch
import gymnasium as gym
import time
import pdb
import pybullet as pb
import pybullet_data
from Robot_Sim.robots.kinova_robotiq_new import Kinova_Robotiq

from utils_PPO import str2bool, Action_adapter, Reward_adapter, evaluate_policy
from PPO import PPO_agent

from DynamicControlEnv_ML_EN_new import ControlModuleEnv
import numpy as np
from JudgeModule_load import RewardPredictor,action_select

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def build_regressor(J_lin, J_ang, a, alpha, omega):
    n = J_lin.shape[1]
    Y = np.zeros((n, 10))

    print("start build regress")
    # Mass term
    for i in range(n):
        Y[i, 0] = J_lin[:, i] @ a

    # First moment terms (m * r)
    Sa = skew(a)
    for i in range(n):
        Y[i, 1:4] = Sa @ J_lin[:, i]

    # Inertia terms
    I_basis = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),  # Ixx
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # Iyy
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),  # Izz
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),  # Ixy
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),  # Ixz
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])  # Iyz
    ]
    for j, B in enumerate(I_basis):
        Iw = B @ omega
        for i in range(n):
            Y[i, 4 + j] += J_ang[:, i] @ (B @ alpha + np.cross(omega, Iw))

    return Y

def build_phi_mass_com(J_lin, a):
    """
    Build reduced regressor Phi for estimating:
    [m, m*r_x, m*r_y, m*r_z]
    """

    print("start build phi")
    n = J_lin.shape[1]
    phi = np.zeros((n, 4))
    S_a = skew(a)
    for i in range(n):
        J_col = J_lin[:, i]
        phi[i, 0] = J_col @ a
        phi[i, 1:4] = J_col @ S_a
    return phi

q_list, q_dot_list, q_ddot_list, tau_list,q_arm_list, q_dot_arm_list, q_ddot_arm_list = [], [], [], [],[],[],[]


env = ControlModuleEnv(GUI_flag=True)
s, info = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
skill_select=2

payload_mass=env.mass

pdb.set_trace()
while True:
    act = [10.0]
    act_env = np.concatenate((np.array([skill_select]), np.array(act)))
    # act_env=1
    s_next, r,_,_ = env.step(act_env)

    time.sleep(0.1)

    if env.identi_flag:
        q_list.append(env.q)
        q_dot_list.append(env.q_dot)
        tau_list.append(env.tau)

        q_arm_list.append(env.q_arm)
        q_dot_arm_list.append(env.q_dot_arm)

    if env.env_end_flag:
        break
env.close()
if pb.getConnectionInfo()['isConnected']:
    pb.disconnect()

pb.connect(pb.DIRECT)

q_list.pop(0)
q_dot_list.pop(0)
tau_list.pop(0)
q_arm_list.pop(0)
q_dot_arm_list.pop(0)

# print(q_list)
# print(np.shape(q_dot_list))
# print(np.shape(tau_list))
# print(np.shape(q_arm_list))
# print(np.shape(q_dot_arm_list))

pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -9.8)

arm_joint_indices=[1,2,3,4,5,6,7]
rotation_euler = [0, 0, 3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion = pb.getQuaternionFromEuler(rotation_euler)

robot = Kinova_Robotiq()
robot.initialize(base_pos=[0.29, -0.8, 0.025], base_ori=rotation_quaternion)

collect_data_step=len(q_list)

print("collect_data_step",collect_data_step)



# Approximate q_ddot via finite difference
q_ddot_list = [np.zeros_like(q_list[0])] + [(q_dot_list[i + 1] - q_dot_list[i]) * 100 for i in
                                            range(len(q_dot_list) - 1)]

q_ddot_arm_list = [np.zeros_like(q_arm_list[0])] + [(q_dot_arm_list[i + 1] - q_dot_arm_list[i]) * 100 for i in
                                            range(len(q_dot_arm_list) - 1)]

# Build full least squares system
Y_total = []
phi_all = []
tau_res_total = []

full_model_estimate=[]
reduce_model_estimate=[]


print("start identification")

solve_data_step=collect_data_step-25




for t in range(25,solve_data_step):
    q, q_dot, q_ddot, tau_meas = q_list[t], q_dot_list[t], q_ddot_list[t], tau_list[t]
    q_dot_,q_ddot_=q_dot_arm_list[t], q_ddot_arm_list[t]

    tau_model = pb.calculateInverseDynamics(robot.id, q.tolist(), q_dot.tolist(), q_ddot.tolist())
    tau_res = np.array(tau_meas) - np.array(tau_model)

    # Jacobian
    J_lin, J_ang = pb.calculateJacobian(robot.id, robot.end_effector_index, [0,0,0], q.tolist(), q_dot.tolist(), q_ddot.tolist())
    J_lin = np.array(J_lin)
    J_ang = np.array(J_ang)

    J_lin = np.array(J_lin)[:, arm_joint_indices]  # only use arm joint columns
    J_ang = np.array(J_ang)[:, arm_joint_indices]  # only use arm joint columns

    # q_dot_=[q_dot[i] for i in arm_joint_indices]
    # q_ddot_=[q_ddot[i] for i in arm_joint_indices]

    # # EE linear/rotational accel
    # a = J_lin @ q_ddot - np.array([0,0,-9.8])
    # omega = J_ang @ q_dot
    # alpha = J_ang @ q_ddot
    print("here now")
    print("q_dot_.shape",q_dot_.shape)
    print("q_ddot_.shape", q_ddot_.shape)
    print("J_lin.shape", J_lin.shape)
    print("J_ang.shape", J_ang.shape)

    # EE linear/rotational accel
    a = J_lin @ q_ddot_ - np.array([0,0,-9.8])
    omega = J_ang @ q_dot_
    alpha = J_ang @ q_ddot_

    Y_t = build_regressor(J_lin, J_ang, a, alpha, omega)
    phi_t = build_phi_mass_com(J_lin, a)

    tau_res = tau_res[arm_joint_indices]
    # phi_t = phi_t[arm_joint_indices, :]
    # Y_t = Y_t[arm_joint_indices, :]

    Y_total.append(Y_t)
    phi_all.append(phi_t)
    tau_res_total.append(tau_res)

print("start solve")
# Stack and solve
Phi = np.vstack(phi_all)
Y_stack = np.vstack(Y_total)
tau_stack = np.vstack(tau_res_total).reshape(-1, 1)
# tau_stack = np.vstack(tau_res_total)
print("Y_stack.shape",Y_stack.shape)
print("tau_stack",tau_stack.shape)

# lambda_reg = 1e-2

theta_est, _, _, _ = np.linalg.lstsq(Y_stack, tau_stack, rcond=None)
# theta_est = np.linalg.inv(Y_stack.T @ Y_stack + lambda_reg * np.eye(Y_stack.shape[1])) @ Y_stack.T @ tau_stack

# Decode estimates
m_est = theta_est[0][0]
com_est = theta_est[1:4].flatten() / m_est
inertia_est = theta_est[4:].flatten()

full_model_estimate.append(m_est)


print(f"\nfull model")
print(f"Estimated mass: {m_est:.4f} kg")
print(f"Estimated CoM offset (m): {com_est}")
print(f"Estimated inertia components: {inertia_est}")
print(f"Actual payload mass: {payload_mass} kg")

theta_est, _, _, _ = np.linalg.lstsq(Phi, tau_stack, rcond=None)

# Decode
m_est = theta_est[0][0]
r_est = theta_est[1:4].flatten() / m_est

reduce_model_estimate.append(m_est)

print(f"\nreduced model")
print(f"Estimated mass: {m_est:.4f} kg")
print(f"Estimated CoM offset (m): {r_est}")
print(f"Actual payload mass: {payload_mass} kg")