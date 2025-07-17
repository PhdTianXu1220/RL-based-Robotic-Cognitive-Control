import os
import copy
import math
import numpy as np
import numpy.random as npr
from collections import deque, namedtuple
from attrdict import AttrDict
from threading import Thread
import pdb

import pybullet as pb
import pybullet_data

import bulletarm
import time
from bulletarm.pybullet.robots.robot_base import RobotBase
from bulletarm.pybullet.utils import constants

jointInfo = namedtuple("jointInfo",
                       ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"])
jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

class Kinova_Robotiq(RobotBase):
  '''

  '''
  def __init__(self):
    super(Kinova_Robotiq, self).__init__()
    # Setup arm and gripper variables
    self.max_forces = [150, 150, 150, 28, 28, 28, 30, 30]
    # self.max_forces = [50, 50, 30, 10, 10, 10, 10, 10]
    self.gripper_close_force = [30] * 2
    self.gripper_open_force = [30] * 2
    # self.end_effector_index = 12
    self.end_effector_index = 7

    # self.home_positions = [0., -1.8137723211649728, 0.44935611885263743, 2.4806174787185227, 1.594027951370597, 2.855598461956419, -1.8932106991701396, 0.6227237490435629, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
   # physical experiment start position
   #  self.home_positions = [0., 23.585, 0.222, 155.804, 268.954,
   #                         1.4, 271.631, 0.528, 0., 0., 0., 0., 0., 0., 0., 0.,
   #                         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    # home_position new for 20250228
    self.home_positions = [0., 23.585, 0.222, 155.804, 268.954-360,
                           1.4, 271.631-360, 0.528, 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
   #  self.home_positions=np.array(self.home_positions)/180*3.14159

    # home_position new for 20250314
    # self.home_positions = [0., 308.159, 60.581, 141.069, 293.726-360,
    #                        38.49, 293.839-360, 0, 0., 0., 0., 0., 0., 0., 0., 0.,
    #                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    # physical experiment end position
    # self.home_positions = [0., 308.159, 60.581, 141.069, 293.726,
    #                        38.49, 293.839, 269.89, 0., 0., 0., 0., 0., 0., 0., 0.,
    #                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    self.home_positions = np.array(self.home_positions) / 180 * 3.14159

    self.home_positions_joint = self.home_positions[1:8]

    self.gripper_joint_limit = [0.0, 0.36]
    self.gripper_joint_names = list()
    self.gripper_joint_indices = list()

    ###############################################
    ## fake robotiq 85
    # the open length of the gripper. 0 is closed, 0.085 is completely opened
    self.robotiq_open_length_limit = [0, 0.085]
    # the corresponding robotiq_85_left_knuckle_joint limit
    self.robotiq_joint_limit = [0.715 - math.asin((self.robotiq_open_length_limit[0] - 0.010) / 0.1143),
                                0.715 - math.asin((self.robotiq_open_length_limit[1] - 0.010) / 0.1143)]

    self.robotiq_controlJoints = ["gen3_robotiq_85_left_knuckle_joint",
                          "gen3_robotiq_85_right_knuckle_joint",
                          "gen3_robotiq_85_left_inner_knuckle_joint",
                          "gen3_robotiq_85_right_inner_knuckle_joint",
                          "gen3_robotiq_85_left_finger_tip_joint",
                          "gen3_robotiq_85_right_finger_tip_joint"]
    self.robotiq_main_control_joint_name = "gen3_robotiq_85_left_inner_knuckle_joint"
    self.robotiq_mimic_joint_name = [
      "gen3_robotiq_85_right_knuckle_joint",
      "gen3_robotiq_85_left_knuckle_joint",
      "gen3_robotiq_85_right_inner_knuckle_joint",
      "gen3_robotiq_85_left_finger_tip_joint",
      "gen3_robotiq_85_right_finger_tip_joint"
    ]
    self.robotiq_mimic_multiplier = [1, -1, 1, -1, -1, 1]
    self.robotiq_joints = AttrDict()

  def initialize(self,base_pos = [0,0,0], base_ori = [0,0,0,1]):
    ''''''
    # pb.connect(pb.GUI)
    # pb.setGravity(0, 0, -9.81)

    # ur5_urdf_filepath = os.path.join(constants.URDF_PATH, 'ur5/ur5_robotiq_85_gripper_fake.urdf')
    ur5_urdf_filepath = '/home/tianxu/Documents/Dynamic Skill Learning/Robot_Sim/kortex_description/robots/gen3_2f85.urdf'
    self.id = pb.loadURDF(ur5_urdf_filepath, base_pos, base_ori,useFixedBase=True)
    # self.is_holding = False
    self.gripper_closed = False
    self.holding_obj = None
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    

    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    self.gripper_joint_names = list()
    self.gripper_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      # print('kinova',i,joint_info)

      pb.enableJointForceTorqueSensor(self.id,i)
      if i in range(1, 8):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)
      elif i in range(9, 10):
        self.gripper_joint_names.append(str(joint_info[1]))
        self.gripper_joint_indices.append(i)

      elif i in range(10, self.num_joints):
        info = pb.getJointInfo(self.id, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                               jointMaxVelocity)
        self.robotiq_joints[singleInfo.name] = singleInfo
        # print("kinova joint", i, singleInfo.name, singleInfo)

  def reset(self):
    self.gripper_closed = False
    self.holding_obj = None
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

  def getGripperOpenRatio(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2)/2
    ratio = (mean - self.gripper_joint_limit[1]) / (self.gripper_joint_limit[0] - self.gripper_joint_limit[1])
    return ratio

  def controlGripper(self, open_ratio, max_it=100):
    p1, p2 = self._getGripperJointPosition()
    target = open_ratio * (self.gripper_joint_limit[0] - self.gripper_joint_limit[1]) + self.gripper_joint_limit[1]
    self._sendGripperCommand(target, target)
    it = 0
    while abs(target - p1) + abs(target - p2) > 0.001:
      self._setRobotiqPosition((p1 + p2) / 2)
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getGripperJointPosition()
      if it > max_it or (abs(p1 - p1_) < 0.0001 and abs(p2 - p2_) < 0.0001):
        return
      p1 = p1_
      p2 = p2_

  def closeGripper(self, max_it=100, primative=constants.PICK_PRIMATIVE):
    ''''''
    p1, p2 = self._getGripperJointPosition()
    limit = self.gripper_joint_limit[1]
    self._sendGripperCommand(limit, limit)
    # self._sendGripperCloseCommand()
    self.gripper_closed = True
    it = 0
    while (limit-p1) + (limit-p2) > 0.001:
      self._setRobotiqPosition((p1 + p2) / 2)
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getGripperJointPosition()
      if it > max_it or (abs(p1-p1_)<0.0001 and abs(p2-p2_)<0.0001):
        mean = (p1+p2)/2 + 0.005
        self._sendGripperCommand(mean, mean)
        return False
      p1 = p1_
      p2 = p2_
    return True

  def adjustGripperCommand(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2) / 2 + 0.005
    self._sendGripperCommand(mean, mean)

  def checkGripperClosed(self):
    limit = self.gripper_joint_limit[1]
    p1, p2 = self._getGripperJointPosition()
    if (limit - p1) + (limit - p2) > 0.001:
      return
    else:
      self.holding_obj = None

  def openGripper(self):
    ''''''
    p1, p2 = self._getGripperJointPosition()
    limit = self.gripper_joint_limit[0]
    self._sendGripperCommand(limit, limit)
    self.gripper_closed = False
    it = 0
    while p1 > 0.0:
      self._setRobotiqPosition((p1 + p2) / 2)
      pb.stepSimulation()
      it += 1
      if it > 100:
        return False
      p1, p2 = self._getGripperJointPosition()
    return True

  def _calculateIK(self, pos, rot):
    # return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:-8]

    # return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:7]

    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot,residualThreshold=1e-4)[:7]

  def _getGripperJointPosition(self):
    p1 = pb.getJointState(self.id, self.gripper_joint_indices[0])[0]
    p2 = pb.getJointState(self.id, self.gripper_joint_indices[1])[0]
    return p1, p2

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                 [0.]*num_motors, self.max_forces[:-2], [0.02]*num_motors, [1.0]*num_motors)

  def _sendGripperCommand(self, target_pos1, target_pos2):
    pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
                                 targetPositions=[target_pos1, target_pos2], forces=self.gripper_open_force,
                                 positionGains=[self.position_gain]*2, velocityGains=[1.0]*2)
    # pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
    #                              targetPositions=[target_pos1, target_pos2], forces=self.gripper_open_force)

  def _getJointState(self):
    indices = [0,1,2,3,4,5,6,7,8]
    st = pb.getJointStates(self.id,indices)
    return st

  def _getJointStateTorque(self):
    manipulator_torque_list=[]
    gripper_torque_list=[]
    indices = [1, 2, 3, 4, 5, 6, 7]
    for i in range(len(indices)):
      joint=indices[i]
      joint_info = pb.getJointInfo(self.id, joint)
      joint_name = joint_info[1].decode('utf-8')
      joint_state = pb.getJointState(self.id, joint)
      torque = joint_state[3]  # This is the torque at the joint
      manipulator_torque_list.append(torque)
      # print(f"Joint {joint_name}: Torque = {torque}")

    for i, jn in enumerate(self.robotiq_controlJoints):
      # print(self.robotiq_joints[jn])
      joint = self.robotiq_joints[jn].id
      joint_info = pb.getJointInfo(self.id, joint)
      joint_name = joint_info[1].decode('utf-8')
      joint_state = pb.getJointState(self.id, joint)
      torque = joint_state[3]  # This is the torque at the joint
      gripper_torque_list.append(torque)
      # print(f"Joint {joint_name}: Torque = {torque}")

    return manipulator_torque_list,gripper_torque_list

  def _getJointStatePower(self):
    manipulator_torque_list=[]
    gripper_torque_list=[]
    manipulator_power=0

    indices = [1, 2, 3, 4, 5, 6, 7]
    for i in range(len(indices)):
      joint=indices[i]
      joint_info = pb.getJointInfo(self.id, joint)
      joint_name = joint_info[1].decode('utf-8')
      joint_state = pb.getJointState(self.id, joint)
      velocity = joint_state[1] # This is the velocity at the joint
      torque = joint_state[3]  # This is the torque at the joint
      joint_power=abs(velocity*torque)
      manipulator_power+=joint_power
      manipulator_torque_list.append(torque)
      # print(f"Joint {joint_name}: Torque = {torque}")

    # for i, jn in enumerate(self.robotiq_controlJoints):
    #   # print(self.robotiq_joints[jn])
    #   joint = self.robotiq_joints[jn].id
    #   joint_info = pb.getJointInfo(self.id, joint)
    #   joint_name = joint_info[1].decode('utf-8')
    #   joint_state = pb.getJointState(self.id, joint)
    #   torque = joint_state[3]  # This is the torque at the joint
    #   gripper_torque_list.append(torque)
    #   # print(f"Joint {joint_name}: Torque = {torque}")

    return manipulator_power

  def _getJointStateAngle(self):
    manipulator_joint_list=[]
    # gripper_torque_list=[]
    # manipulator_power=0

    indices = [1, 2, 3, 4, 5, 6, 7]
    for i in range(len(indices)):
      joint=indices[i]
      joint_info = pb.getJointInfo(self.id, joint)
      # joint_name = joint_info[1].decode('utf-8')
      joint_state = pb.getJointState(self.id, joint)
      joint_angle = joint_state[0] # This is the velocity at the joint

      manipulator_joint_list.append(joint_angle)
      # print(f"Joint {joint_name}: Torque = {torque}")

    # for i, jn in enumerate(self.robotiq_controlJoints):
    #   # print(self.robotiq_joints[jn])
    #   joint = self.robotiq_joints[jn].id
    #   joint_info = pb.getJointInfo(self.id, joint)
    #   joint_name = joint_info[1].decode('utf-8')
    #   joint_state = pb.getJointState(self.id, joint)
    #   torque = joint_state[3]  # This is the torque at the joint
    #   gripper_torque_list.append(torque)
    #   # print(f"Joint {joint_name}: Torque = {torque}")
    manipulator_joint_list=np.array(manipulator_joint_list)

    return manipulator_joint_list

  
  def _getLinkState(self,link_idx):
    st = pb.getLinkState(self.id,link_idx, computeLinkVelocity=1)
    return st
  
  def _resetJointState(self, joint_value):
    indices = [1,2,3,4,5,6,7]
    # joint_value = [0] + list(joint_value)
    joint_value = tuple(joint_value)
    for i, jv in enumerate(joint_value):
      if indices[i]==7:
        # 0.528 / 180 * 3.14159
        pb.resetJointState(self.id, indices[i], targetValue=jv, targetVelocity=0)
      else:
        pb.resetJointState(self.id, indices[i], targetValue=jv, targetVelocity=0)
      # print("index",indices[i]," value:",jv )

  def _resetJointStateforce(self, joint_value):
    indices = [1, 2, 3, 4, 5, 6, 7]
    # joint_value = [0] + list(joint_value)
    joint_value = tuple(joint_value)
    for i, jv in enumerate(joint_value):
      if indices[i] == 7:
        # 0.528 / 180 * 3.14159
        # pb.resetJointState(self.id, indices[i], targetValue=jv, targetVelocity=0)
        # force=40
        pb.setJointMotorControl2(self.id,
                                 indices[i],
                                 pb.POSITION_CONTROL,
                                 targetPosition=jv,
                                 force=40,
                                 positionGain=0.3
                                 )
      else:
        # pb.resetJointState(self.id, indices[i], targetValue=jv, targetVelocity=0)
        pb.setJointMotorControl2(self.id,
                                 indices[i],
                                 pb.POSITION_CONTROL,
                                 targetPosition=jv,
                                 force=40,
                                 positionGain=0.3
                                 )
        # , positionGain = 0.1, velocityGain = 0.01
      # print("index", indices[i], " value:", jv)

    # for i in range(9,self.num_joints):
    #   pb.resetJointState(self.id, i, targetValue=0.0, targetVelocity=0)

      
      # pdb.set_trace()

  def _setRobotiqPosition(self, pos):
    percentage = pos/self.gripper_joint_limit[1]
    target = percentage * (self.robotiq_joint_limit[0]-self.robotiq_joint_limit[1]) + self.robotiq_joint_limit[1]
    for i, jn in enumerate(self.robotiq_controlJoints):
      # print(i,jn)
      # print(self.robotiq_joints[jn])
      motor = self.robotiq_joints[jn].id
      # pb.resetJointState(self.id, motor, target*self.robotiq_mimic_multiplier[i])
      # print("kinova motor", motor,"end")
      # print("kinova", self.robotiq_joints[jn], self.robotiq_mimic_multiplier[i])

      pb.setJointMotorControl2(self.id,
                               motor,
                               pb.POSITION_CONTROL,
                               targetPosition=target*self.robotiq_mimic_multiplier[i],
                               force=40)
