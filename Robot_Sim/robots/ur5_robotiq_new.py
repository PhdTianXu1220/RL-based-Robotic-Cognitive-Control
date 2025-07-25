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

class UR5_Robotiq(RobotBase):
  '''

  '''
  def __init__(self):
    super(UR5_Robotiq, self).__init__()
    # Setup arm and gripper variables
    self.max_forces = [150, 150, 150, 28, 28, 28, 30, 30]
    # self.max_forces = [50, 50, 30, 10, 10, 10, 10, 10]
    self.gripper_close_force = [30] * 2
    self.gripper_open_force = [30] * 2
    # self.end_effector_index = 12
    self.end_effector_index = 12

    # simulation experiment start position
    # self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # # physical experiment start position
    # self.home_positions = [0., 0.32, -90.65, 90.69, -90.69, -90.13, 0, 180.26, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #                        0., 0., 0., 0., 0., 0., 0., 0.]

    # # physical dual arm pickup experiment start position
    self.home_positions = [0., -0.73067221, -86.23227831,  75.47428562, -79.21878223, -89.99208295,
  -0.73334452, 180.26, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0.]

    self.home_positions=np.array(self.home_positions)/180*3.14159

    # physical experiment end position
    # self.home_positions = [0., -48.96, -74.94, 116.95, -130.41, 272.24, 129.94, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #                        0., 0., 0., 0., 0., 0., 0., 0.]

    # self.home_positions=np.array(self.home_positions)/180*3.14159

    self.home_positions_joint = self.home_positions[1:7]

    self.gripper_joint_limit = [0, 0.036]
    self.gripper_joint_names = list()
    self.gripper_joint_indices = list()

    ###############################################
    ## fake robotiq 85
    # the open length of the gripper. 0 is closed, 0.085 is completely opened
    self.robotiq_open_length_limit = [0, 0.085]
    # the corresponding robotiq_85_left_knuckle_joint limit
    self.robotiq_joint_limit = [0.715 - math.asin((self.robotiq_open_length_limit[0] - 0.010) / 0.1143),
                                0.715 - math.asin((self.robotiq_open_length_limit[1] - 0.010) / 0.1143)]

    self.robotiq_controlJoints = ["robotiq_85_left_knuckle_joint",
                          "robotiq_85_right_knuckle_joint",
                          "robotiq_85_left_inner_knuckle_joint",
                          "robotiq_85_right_inner_knuckle_joint",
                          "robotiq_85_left_finger_tip_joint",
                          "robotiq_85_right_finger_tip_joint"]
    self.robotiq_main_control_joint_name = "robotiq_85_left_inner_knuckle_joint"
    self.robotiq_mimic_joint_name = [
      "robotiq_85_right_knuckle_joint",
      "robotiq_85_left_knuckle_joint",
      "robotiq_85_right_inner_knuckle_joint",
      "robotiq_85_left_finger_tip_joint",
      "robotiq_85_right_finger_tip_joint"
    ]
    self.robotiq_mimic_multiplier = [1, 1, 1, 1, -1, -1]
    self.robotiq_joints = AttrDict()

  def initialize(self,base_pos = [0,0,0], base_ori = [0,0,0,1]):
    ''''''
    # pb.connect(pb.GUI)
    # pb.setGravity(0, 0, -9.81)

    # ur5_urdf_filepath = os.path.join(constants.URDF_PATH, 'ur5/ur5_robotiq_85_gripper_fake.urdf')
    ur5_urdf_filepath = '/home/tianxu/Documents/DMP-python/Dual_Arm_New/Robot_Sim/urdf/ur5/ur5_robotiq_85_gripper_true.urdf'
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
      pb.enableJointForceTorqueSensor(self.id,i)
      if i in range(1, 7):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)
      elif i in range(10, 12):
        self.gripper_joint_names.append(str(joint_info[1]))
        self.gripper_joint_indices.append(i)

      elif i in range(14, self.num_joints):
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
        print("ur5e joint",i,singleInfo.name,singleInfo)

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
    print("stop1")
    while (limit-p1) + (limit-p2) > 0.001:
      print("stop2")
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
    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot,residualThreshold=0.001,
                                              numSolverIterations=100)[:-8]

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
    # pb.resetJointState(self.id, self.gripper_joint_indices,
    #                              [target_pos1, target_pos2])
    # pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
    #                              targetPositions=[target_pos1, target_pos2], forces=self.gripper_open_force)

  def _getJointState(self):
    # indices = [0,1,2,3,4,5,6,7,8]
    indices = [1, 2, 3, 4, 5, 6, 7]
    st=[pb.getJointState(self.id, j)[0] for j in indices]
    # st = pb.getJointStates(self.id,indices)
    return st
  
  def _getLinkState(self,link_idx):
    st = pb.getLinkState(self.id,link_idx, computeLinkVelocity=1)
    return st
  
  def _resetJointState(self, joint_value):
    indices = [0,1,2,3,4,5,6,7]
    joint_value = [0] + list(joint_value)
    joint_value = tuple(joint_value)
    for i, jv in enumerate(joint_value):
      pb.resetJointState(self.id, i, targetValue=jv, targetVelocity=0)
      
      # pdb.set_trace()
  def _resetJointStateforce(self, joint_value):
    indices = [0, 1, 2, 3, 4, 5, 6, 7]
    joint_value = [0] + list(joint_value)
    joint_value = tuple(joint_value)
    for i, jv in enumerate(joint_value):
        # 0.528 / 180 * 3.14159
        # pb.resetJointState(self.id, indices[i], targetValue=jv, targetVelocity=0)
        pb.setJointMotorControl2(self.id,
                                 indices[i],
                                 pb.POSITION_CONTROL,
                                 targetPosition=jv,
                                 force=500)

  def _setRobotiqPosition(self, pos):
    percentage = pos/self.gripper_joint_limit[1]
    print("percentage",percentage)
    target = percentage * (self.robotiq_joint_limit[0]-self.robotiq_joint_limit[1]) + self.robotiq_joint_limit[1]
    for i, jn in enumerate(self.robotiq_controlJoints):
      motor = self.robotiq_joints[jn].id
      # pb.resetJointState(self.id, motor, target*self.robotiq_mimic_multiplier[i])
      print("ur5e motor",motor)
      print("ur5e",self.robotiq_joints[jn],self.robotiq_mimic_multiplier[i])

      pb.setJointMotorControl2(self.id,
                               motor,
                               pb.POSITION_CONTROL,
                               targetPosition=target*self.robotiq_mimic_multiplier[i],
                               force=500)
