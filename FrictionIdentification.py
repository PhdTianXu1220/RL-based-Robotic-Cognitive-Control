import pybullet as pb
import pybullet_data
import numpy as np
from numpy.linalg import pinv
from Robot_Sim.robots.kinova_robotiq_new import Kinova_Robotiq
import time
import pdb

def tmp_skill_lib(skill_ID):
    if skill_ID==0:
        start1 = [0.28667097385077456, -0.48847984909271797, 0.5940973561794143 + 0.022]
        # mid1 = [0.28667097385077456 - 0.25, -0.48847984909271797, 0.5940973561794143 + 0.022 + 0.15]
        mid1 = [0.28667097385077456 - 0.2, -0.48847984909271797, 0.5940973561794143 + 0.022 + 0.1]

        end1 = [0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.022]

        tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
        tra_go = np.array([np.linspace(start1[i], mid1[i], 400) for i in range(3)]).T
        tra_go2 = np.array([np.linspace(mid1[i], end1[i], 100) for i in range(3)]).T

        tra1 = tra_go
        return tra1
    elif skill_ID==1:
        start1=[0.28667097385077456, -0.48847984909271797, 0.5940973561794143+0.022]
        end1=[0.28667097385077456-0.3, -0.48847984909271797, 0.5940973561794143+0.022]

        # tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
        tra_go = np.array([np.linspace(start1[i], end1[i], 400) for i in range(3)]).T

        # tra1 = np.vstack((tra_pre, tra_go))
        tra1 = tra_go
        return tra1

    elif skill_ID==2:
        start1 = [0.28667097385077456, -0.48847984909271797, 0.5940973561794143 + 0.022]
        mid1 = [0.28667097385077456 , -0.48847984909271797, 0.5940973561794143 + 0.022 + 0.08]
        mid2 = [0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.022 + 0.08]
        end1 = [0.28667097385077456 - 0.5, -0.48847984909271797, 0.5940973561794143 + 0.022]

        tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
        tra_go = np.array([np.linspace(start1[i], mid1[i], 40) for i in range(3)]).T
        tra_go2 = np.array([np.linspace(mid1[i], mid2[i], 320) for i in range(3)]).T
        tra_putdown = np.array([np.linspace(mid2[i], end1[i], 40) for i in range(3)]).T

        tra1 = np.vstack(( tra_go, tra_go2,tra_putdown))
        return tra1

    else:
        print("no such skill")
        return -1

# Setup PyBullet
pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0, 0, -9.8)



plane = pb.loadURDF("plane.urdf")


# ee_link = 7

# controlled_joints=[1, 2, 3, 4, 5, 6, 7]
arm_joint_indices=[1,2,3,4,5,6,7]

block_start_position =[0.28667097385077456, -0.48847984909271797, 0.5940973561794143-0.17]
block_target_position=[-0.18100409137299697, -0.49399412541519766, 0.4171282952898786]

block_start_orientation = pb.getQuaternionFromEuler([0, 0, 0])
block_id = pb.loadURDF("/home/tianxu/Documents/Dynamic Skill Learning/Robot_Sim/urdf/object/magnetic.urdf", block_start_position, block_start_orientation)
table_id = pb.loadURDF("/home/tianxu/Documents/Dynamic Skill Learning/Robot_Sim/urdf/object/table.urdf", basePosition=[0, -0.4, -0.22],useFixedBase=True)

rotation_euler = [0, 0, 3.14159 / 2]  # Roll, Pitch, Yaw (in radians)
rotation_quaternion = pb.getQuaternionFromEuler(rotation_euler)

robot= Kinova_Robotiq()
robot.initialize(base_pos=[0.29, -0.8, 0.025], base_ori=rotation_quaternion)

rotation_ee = [0,0,0.707,0.707]

loose_thres=0.05

num_joints = pb.getNumJoints(robot.id)
controlled_joints = [i for i in range(num_joints) if pb.getJointInfo(robot.id, i)[2] != pb.JOINT_FIXED]
n_dof = len(controlled_joints)


# Attach object
payload_mass = 1
pb.changeDynamics(block_id, -1, mass=payload_mass)
pb.changeDynamics(block_id, -1, lateralFriction=1)
pb.changeDynamics(block_id, -1, rollingFriction=0.6,spinningFriction=0.2)

friction_param=0.3

pb.changeDynamics(table_id, -1, lateralFriction=friction_param)
pb.changeDynamics(table_id, -1, rollingFriction=1)

com_offset = np.array([0.0, 0.0, 0.2])


pb.stepSimulation()

# Attach a known payload to the end-effector

# Define straight-line EE trajectory in x-direction
steps = 400
tra = tmp_skill_lib(1)

q_list, q_dot_list, q_ddot_list, tau_list,q_arm_list, q_dot_arm_list, q_ddot_arm_list = [], [], [], [],[],[],[]

pdb.set_trace()

for i in range(30):
    robot._setRobotiqPosition(0.2)
    # robot._setRobotiqPosition(0.0)
    pb.stepSimulation()
    time.sleep(0.1)

for t in range(steps):
    joint_angle1 = robot._calculateIK(np.array(tra[t]) + [0.0, 0.0, -0.0], rotation_ee)
    robot._resetJointStateforce(joint_angle1)

    pb.stepSimulation()
    time.sleep(0.1)

    print("time step",t)

    manipulator_torque, gripper_torque = robot._getJointStateTorque()
    gripper_signal = np.sum(np.abs(np.array(gripper_torque)))

    if gripper_signal < loose_thres:
        print("block slip out")
        break

    js = pb.getJointStates(robot.id, controlled_joints)
    q = np.array([s[0] for s in js])
    q_dot = np.array([s[1] for s in js])
    tau = np.array([s[3] for s in js])

    js = pb.getJointStates(robot.id, arm_joint_indices)
    q_arm = np.array([s[0] for s in js])
    q_dot_arm = np.array([s[1] for s in js])

    q_list.append(q)
    q_dot_list.append(q_dot)
    tau_list.append(tau)

    q_arm_list.append(q_arm)
    q_dot_arm_list.append(q_dot_arm)

collect_data_step=t
print("collect_data_step",collect_data_step)

q_ddot_list = [np.zeros_like(q_list[0])] + [(q_dot_list[i+1] - q_dot_list[i]) * 100 for i in range(len(q_dot_list)-1)]
q_ddot_arm_list = [np.zeros_like(q_arm_list[0])] + [(q_dot_arm_list[i + 1] - q_dot_arm_list[i]) * 100 for i in
                                            range(len(q_dot_arm_list) - 1)]

solve_data_step=collect_data_step-100

if solve_data_step-30<100:
    solve_data_step=120

F_ext_list = []

for t in range(30,solve_data_step):
    q = q_list[t]
    q_dot = q_dot_list[t]
    q_ddot = q_ddot_list[t]
    tau_meas = tau_list[t]



    tau_nominal = pb.calculateInverseDynamics(robot.id, q.tolist(), q_dot.tolist(), q_ddot.tolist())
    tau_res = np.array(tau_meas) - np.array(tau_nominal)

    tau_res = tau_res[arm_joint_indices]

    # Jacobian
    J_lin, J_ang= pb.calculateJacobian(robot.id, robot.end_effector_index, [0, 0, 0], q.tolist(), q_dot.tolist(), q_ddot.tolist())
    J_lin = np.array(J_lin)[:, arm_joint_indices]  # only use arm joint columns
    J_ang = np.array(J_ang)[:, arm_joint_indices]  # only use arm joint columns

    J_full = np.vstack([np.array(J_lin), np.array(J_ang)])

    q_ddot = q_ddot[arm_joint_indices]

    # Net wrench from known payload
    a_com = np.array(J_lin) @ q_ddot
    f_lin = payload_mass * a_com
    f_ang = payload_mass * np.cross(com_offset, a_com)
    F_net = np.concatenate([f_lin, f_ang])

    # Robot wrench
    F_robot = pinv(J_full.T) @ tau_res

    # External wrench
    F_ext = F_net - F_robot
    F_ext_list.append(F_ext)

# Average over time
F_ext_avg = np.mean(F_ext_list, axis=0)


labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
for name, val in zip(labels, F_ext_avg):
    print(f"{name}: {val:.4f} N or Nm")

Fx, Fy, Fz = F_ext_avg[0], F_ext_avg[1], F_ext_avg[2]
f_friction = np.sqrt(Fx**2 + Fy**2)
f_normal = abs(Fz)

mu_est = f_friction / f_normal if f_normal > 1e-5 else np.nan
print(f"\nEstimated Friction Coefficient μ: {mu_est:.4f}")
print(f"True Friction Coefficient μ: {friction_param:.4f}")

pb.disconnect()
