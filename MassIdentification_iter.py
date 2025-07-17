import pybullet as pb
import pybullet_data
import time
import numpy as np
import pdb
from Robot_Sim.robots.kinova_robotiq_new import Kinova_Robotiq
import matplotlib.pyplot as plt

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
        tra_go = np.array([np.linspace(start1[i], mid1[i], 40) for i in range(3)]).T
        tra_go2 = np.array([np.linspace(mid1[i], mid2[i], 320) for i in range(3)]).T
        tra_putdown = np.array([np.linspace(mid2[i], end1[i], 40) for i in range(3)]).T

        tra1 = np.vstack(( tra_go, tra_go2,tra_putdown))
        return tra1

    elif skill_ID==3:
        start1=[0.28667097385077456, -0.48847984909271797, 0.5940973561794143+0.022]
        end1=[0.28667097385077456, -0.48847984909271797, 0.5940973561794143+0.022+0.1]

        tra_pre = np.array([np.linspace(start1[i], start1[i], 20) for i in range(3)]).T
        tra_go = np.array([np.linspace(start1[i], end1[i], 400) for i in range(3)]).T

        tra1 = np.vstack((tra_pre, tra_go))
        return tra1

    else:
        print("no such skill")
        return -1

def plot_multiple_arrays(*arrays):
    plt.figure(figsize=(8, 6))  # Set the size of the plot
    plt.rcParams['font.family'] = 'Nimbus Roman'
    # array_legend_list=["payload mass","full model","reduced model"]
    array_legend_list = ["True Value", "Estimated Value"]
    color_list=['black','red','green']
    line_type_list=['-','--','-.']

    # Loop through each array in the provided inputs
    for index, array in enumerate(arrays):
        mass_list_range = np.arange(0.2, 3.9, 0.2)
        label_name=array_legend_list[index]
        color=color_list[index]
        linestyle=line_type_list[index]
        plt.plot(mass_list_range,array, label=label_name,linewidth=2.5,color=color, linestyle=linestyle)

    # Adding title and labels
    plt.title('Mass Identification',fontsize=28)
    plt.xlabel('True Payload Mass',fontsize=24)
    plt.ylabel('Estimate Payload Mass',fontsize=24)



    # Adding a legend to explain which line corresponds to which array
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)  # Set font size of the tick labels on x-axis
    plt.yticks(fontsize=20)  # Set font size of the tick labels on y-axis

    plt.savefig('mass identification lift.png', dpi=600, format='png')

    # Show the plot
    plt.show()

def plot_multiple_arrays_error(*arrays):
    plt.figure(figsize=(8, 6))  # Set the size of the plot
    plt.rcParams['font.family'] = 'Nimbus Roman'
    # array_legend_list=["payload mass","full model","reduced model"]
    # array_legend_list = ["payload mass", "full model", "reduced model"]
    color_list=['black','red','green']
    line_type_list=['-','--','-.']

    # Loop through each array in the provided inputs
    for index, array in enumerate(arrays):
        if index>0:
            mass_list_range = np.arange(0.2, 3.9, 0.2)
            # label_name=array_legend_list[index]
            color=color_list[index]
            linestyle=line_type_list[index]
            # plt.plot(mass_list_range,array, label=label_name,linewidth=2.5,color=color, linestyle=linestyle)
            plt.plot(mass_list_range, np.abs(array-mass_list_range), linewidth=2.5, color="black", linestyle=linestyle)

    # Adding title and labels
    plt.title('Mass Identification',fontsize=28)
    plt.xlabel('True Payload Mass',fontsize=24)
    plt.ylabel('Payload Mass Estimation Error',fontsize=24)



    # Adding a legend to explain which line corresponds to which array
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)  # Set font size of the tick labels on x-axis
    plt.yticks(fontsize=20)  # Set font size of the tick labels on y-axis

    plt.savefig('mass identification lift error.png', dpi=600, format='png')

    # Show the plot
    plt.show()

mass_list_range = np.arange(0.2, 3.9, 0.2)

full_model_estimate=[]
reduce_model_estimate=[]

for iter_num in range(len(mass_list_range)):
    # Setup PyBullet
    # pb.connect(pb.GUI)
    pb.connect(pb.DIRECT)

    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.8)



    plane = pb.loadURDF("plane.urdf")


    # ee_link = 7

    # controlled_joints=[1, 2, 3, 4, 5, 6, 7]
    arm_joint_indices=[1,2,3,4,5,6,7]

    block_start_position =[0.28667097385077456, -0.48847984909271797, 0.5940973561794143-0.17]
    block_target_position=[-0.18100409137299697, -0.49399412541519766, 0.4171282952898786]

    block_start_orientation = pb.getQuaternionFromEuler([0, 0, 0])
    block_id = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/magnetic.urdf", block_start_position, block_start_orientation)
    table_id = pb.loadURDF("/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/object/table.urdf", basePosition=[0, -0.4, -0.22],useFixedBase=True)

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
    payload_mass = mass_list_range[iter_num]
    pb.changeDynamics(block_id, -1, mass=payload_mass)
    pb.changeDynamics(block_id, -1, lateralFriction=1)
    # pb.createConstraint(robot.id, robot.end_effector_index, block_id, -1, pb.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
    # pb.createConstraint(
    #     parentBodyUniqueId=robot.id,
    #     parentLinkIndex=robot.end_effector_index,
    #     childBodyUniqueId=block_id,
    #     childLinkIndex=-1,
    #     jointType=pb.JOINT_FIXED,
    #     jointAxis=[0, 0, 0],
    #     parentFramePosition=[0, 0, 0],         # attach at tool center
    #     childFramePosition=[0, 0, 0.2]        # 10 cm offset in child's frame
    # )
    pb.stepSimulation()

    # Simulate motion
    steps = 400
    q_list, q_dot_list, q_ddot_list, tau_list,q_arm_list, q_dot_arm_list, q_ddot_arm_list = [], [], [], [],[],[],[]


    tra1=tmp_skill_lib(3)

    # pdb.set_trace()

    for i in range(30):
        robot._setRobotiqPosition(0.23)
        # robot._setRobotiqPosition(0.0)
        pb.stepSimulation()
        time.sleep(0.001)


    for t in range(steps):
        joint_angle1 = robot._calculateIK(np.array(tra1[t]) + [0.0, 0.0, -0.0], rotation_ee)
        robot._resetJointStateforce(joint_angle1)

        pb.stepSimulation()
        time.sleep(0.001)

        print("time step",t)


        js = pb.getJointStates(robot.id, controlled_joints)
        q = np.array([s[0] for s in js])
        q_dot = np.array([s[1] for s in js])
        tau = np.array([s[3] for s in js])


        js = pb.getJointStates(robot.id, arm_joint_indices)
        q_arm = np.array([s[0] for s in js])
        q_dot_arm = np.array([s[1] for s in js])

        print(q,q_dot,tau)

        manipulator_torque, gripper_torque = robot._getJointStateTorque()
        gripper_signal = np.sum(np.abs(np.array(gripper_torque)))

        if gripper_signal < loose_thres:
            print("block slip out")
            break

        q_list.append(q)
        q_dot_list.append(q_dot)
        tau_list.append(tau)

        q_arm_list.append(q_arm)
        q_dot_arm_list.append(q_dot_arm)

    collect_data_step=t
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



    print("start identification")

    solve_data_step=collect_data_step-100

    if solve_data_step-40<100:
        solve_data_step=140

    for t in range(40,solve_data_step):
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

    # lambda_reg = 1e-4
    # theta_est = np.linalg.inv(Phi.T @ Phi + lambda_reg * np.eye(Phi.shape[1])) @ Phi.T @ tau_stack
    #
    # print(Phi.T @ Phi)
    #
    # cond_num = np.linalg.cond(Phi.T @ Phi)
    # print(f"Condition number: {cond_num:.2e}")
    #
    # # Decode
    # m_est = theta_est[0][0]
    # r_est = theta_est[1:4].flatten() / m_est
    #
    # print(f"\nEstimated mass: {m_est:.4f} kg")
    # print(f"Estimated CoM offset (m): {r_est}")
    # print(f"Actual payload mass: {payload_mass} kg")

    pb.disconnect()

plot_multiple_arrays(mass_list_range,full_model_estimate)
plot_multiple_arrays_error(mass_list_range,full_model_estimate)