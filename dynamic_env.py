import os
import time
import math
import random
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

import pybullet as pybullet
import pybullet_data

from pip._internal.operations import freeze

pkgs = freeze.freeze()
with open('requirements_homing.txt', 'w') as f:
    for pkg in pkgs:
        f.write(pkg)
        f.write('\n')


class TrajbotEnv(gym.Env):

    def __init__(self, natural=False, connect_type="GUI"):

        # Actions: dw1, dw2, dw3, dw4, dw5, dw6 - Delta omega - increments Or decrements of angular velocity
        # self.action_space = spaces.Box(np.array([-0.02, -0.02 ,-0.02, -0.02, -0.02 ,-0.02]), np.array([+0.02, +0.02 ,+0.02, +0.02, +0.02 ,+0.02]))

        self.action_space = spaces.Box(np.array([-0.1]),
                                       np.array([+1.0]), dtype=np.float64)

        # State space: Per link (6 links) : Pos - Rot - Linear Velocity - Angular Velocity = 78 parameters

        num_state_param = 15  # 15 + 25 rays = 40

        high = np.inf * np.ones(num_state_param, dtype="float64")

        self.observation_space = spaces.Box(-high, high, dtype=np.float64)
        # State Vector
        self._observation = []

        # Robot ID
        self.robot = -1

        # Robot Trajectory
        with open('dynamic_jointangle', 'r') as trajectory_file:
            dynamic_jointangle = [[float(num) for num in line.split(',')] for line in trajectory_file]
        self._traj = dynamic_jointangle
        print(self._traj[0])

        # Initial Completion Time
        self._ict = np.array([1.0])
        self._currentCT = self._ict

        # Torque Limit
        self._torquelimit = 50.0

        # Define the max torque of the trajectory
        self._maxtorque = 0.0

        # Joint Velocities
        self.CurrentJointVelocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Joint Limits
        self._jointsLL = [0., 0., 0., 0., 0., 0.]

        self._jointsUL = [0., 0., 0., 0., 0., 0.]

        self._jointsPosInitial = [0., 0.0, 0., 0., 0., 0.]

        self._jointIsOverLimit = False

        self._all_links = np.array([1., 2., 3., 4., 5., 6.])

        # Collision Flag
        self._isContact = False

        # Env Counter Flag
        self._envCounter = 0

        self._maxCounter = 1

        # This episode counter is introduced so that
        # the Agent at the early stages to be allowed
        # to do more - Allowing self collision
        # After 200 episodes that is changed so it will
        # not be permitted these moves.
        self._episode = 200

        self._episode_counter = 0

        self.SetMaxTimeSteps(200)

        if connect_type == "DIRECT":
            self._physicsClientId = pybullet.connect(pybullet.DIRECT)
        else:
            self._physicsClientId = pybullet.connect(pybullet.GUI)

            self.state = self._observation

        self.seed()

        self.set_initial()

        self.reset()

    def set_initial(self):
        self.CurrentJointVelocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        robot = self.ResetPybulletEnv()

        self.ResetRobot(robot)

        self.setState(robot)

    def SetMaxTimeSteps(self, maxTimeSteps):
        self._maxCounter = maxTimeSteps

    def GetJointLimits(self, robot):
        jointsLL = [0., 0., 0., 0., 0., 0.]

        jointsUL = [0., 0., 0., 0., 0., 0.]

        for i in range(6):
            jointsInfo = pybullet.getJointInfo(robot, i + 1, physicsClientId=self._physicsClientId)

            jointsLL[i] = jointsInfo[8]

            jointsUL[i] = jointsInfo[9]

        self._jointsLL = np.array(jointsLL)

        self._jointsUL = np.array(jointsUL)

    def CheckJointLimits(self, robot):
        self.GetJointLimits(robot)

        joints_pos = np.array([0., 0., 0., 0., 0., 0.])

        for i in range(6):
            joint_pos, _, _, _ = pybullet.getJointState(robot, i + 1)

            joints_pos[i] = joint_pos

        over_Limit = joints_pos > self._jointsUL

        under_Limit = joints_pos < self._jointsLL

        self._jointIsOverLimit = np.sum(over_Limit + under_Limit) > 0.

        return self._jointIsOverLimit

    def ContactFlag(self, robot):
        return pybullet.getContactPoints(robot, physicsClientId=self._physicsClientId)

    def ComputeMaxTorque(self, robot):
        time_interval = self._ict / len(self._traj)
        joint_angle1 = []
        joint_angle2 = []
        joint_angle3 = []
        joint_angle4 = []
        joint_angle5 = []
        joint_angle6 = []
        for i in range(len(self._traj)):
            joint_angle1.append(dynamic_jointangle[i][0])
            joint_angle2.append(dynamic_jointangle[i][1])
            joint_angle3.append(dynamic_jointangle[i][2])
            joint_angle4.append(dynamic_jointangle[i][3])
            joint_angle5.append(dynamic_jointangle[i][4])
            joint_angle6.append(dynamic_jointangle[i][5])

        joint_angles = [joint_angle1, joint_angle2, joint_angle3, joint_angle4, joint_angle5, joint_angle6]
        joint_velocities = []
        joint_accelerations = []
        for j in range(6):
            joint_angle = np.array(joint_angles[j])
            joint_velocity = np.zeros_like(joint_angle)
            joint_acceleration = np.zeros_like(joint_angle)
            # Calculate velocities (with zero initial and end velocities)
            # For intermediate points, use the average rate of change with adjacent points
            for i in range(1, len(joint_angle) - 1):
                joint_velocity[i] = (joint_angle[i + 1] - joint_angle[i - 1]) / (2 * time_interval)

            # Calculate accelerations
            for i in range(1, len(joint_velocity) - 1):
                joint_acceleration[i] = (joint_velocity[i + 1] - joint_velocity[i - 1]) / (2 * time_interval)

            joint_velocities.append(joint_velocity)
            joint_accelerations.append(joint_acceleration)
        dynamic_jointvelocity = []
        dynamic_jointacc = []
        for i in range(len(joint_angle1)):
            dynamic_jointvelocity.append([joint_velocities[0][i], joint_velocities[1][i], joint_velocities[2][i],
                                          joint_velocities[3][i], joint_velocities[4][i], joint_velocities[5][i]])
            dynamic_jointacc.append([joint_accelerations[0][i], joint_accelerations[1][i], joint_accelerations[2][i],
                                     joint_accelerations[3][i], joint_accelerations[4][i], joint_accelerations[5][i]])

        dynamic_torque = []
        for i in range(len(joint_angle1)):
            torque = pybullet.calculateInverseDynamics(robot, dynamic_jointangle[i], dynamic_jointvelocity[i],
                                                       dynamic_jointacc[i])
            dynamic_torque.append(torque)

        flattened_list = [item for sublist in dynamic_torque for item in sublist]
        self._maxtorque = max(flattened_list)

        return self._maxtorque

    def ResetJointPos(self):
        j1 = 0.0

        j2 = 0.0

        j3 = 0.0

        j4 = 0.0

        j5 = 0.0

        j6 = 0.0

        self.joints_pos_orig = np.array([j1, j2, j3, j4, j5, j6])

        self._jointsPosInitial = np.array([j1, j2, j3, j4, j5, j6])

        # for i in range(6):
        #     self.joints_pos_orig[i] = np.random.uniform(self._jointsLL[i], self._jointsUL[i])
        #
        #     self._jointsPosInitial[i] = self.joints_pos_orig[i]

        return self.joints_pos_orig

    def ResetJointPVel(self):
        j1 = 0.0

        j2 = 0.0

        j3 = 0.0

        j4 = 0.0

        j5 = 0.0

        j6 = 0.0

        self.joints_vel_orig = np.array([j1, j2, j3, j4, j5, j6])

        return self.joints_vel_orig

    def ResetRobot(self, robot):

        self.GetJointLimits(robot)

        joints_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        joints_pos = self.ResetJointPos()

        joints_vel = self.ResetJointPVel()

        for i in range(0, 6):
            pybullet.resetJointState(robot, i + 1, joints_pos[i], joints_vel[i],
                                     physicsClientId=self._physicsClientId)

        for _ in range(1):
            pybullet.stepSimulation(physicsClientId=self._physicsClientId)

        # is_done = True
        #
        # while is_done:
        #
        #     is_done = self.getDone(robot)

    def SetRobotPos(self, pos=np.array([0., 0., 0., 0., 0., 0.])):
        for i in range(6):
            pybullet.resetJointState(self.robot, i + 1, pos[i], 0., physicsClientId=self._physicsClientId)

            pybullet.stepSimulation(physicsClientId=self._physicsClientId)

    def ResetPybulletEnv(self):
        pybullet.resetSimulation(self._physicsClientId)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        pybullet.setGravity(0, 0, -9.80665, physicsClientId=self._physicsClientId)

        pybullet.setRealTimeSimulation(0, physicsClientId=self._physicsClientId)

        self._episode_counter = self._episode_counter + 1

        robot = pybullet.loadURDF("fanuc_lrmate200id_new/lrmate200id_brush.urdf", [0, 0, 0.0], useFixedBase=1,
                                  flags=pybullet.URDF_USE_SELF_COLLISION,
                                  physicsClientId=self._physicsClientId)  # use a fixed base!

        robot_num_joints = pybullet.getNumJoints(robot, physicsClientId=self._physicsClientId)

        self.robot = robot

        self._jointsLL = [0., 0., 0., 0., 0., 0.]

        self._jointsUL = [0., 0., 0., 0., 0., 0.]

        self.reset_flags()

        return robot

    def reset_flags(self):
        self._jointIsOverLimit = False

        # Current Displacement
        self._currentDisplacement = -1.0

        # Env Counter Flag
        self._envCounter = 0

        # Collision Flag
        self._isContact = False

    def setState(self, robot):
        joints_angle = []
        joints_vel = []

        for jointId in range(1, 7):
            joint_state = pybullet.getJointState(self.robot, jointId, physicsClientId=self._physicsClientId)
            joints_angle.append(joint_state[0])
            joints_vel.append(joint_state[1])

        last_link = 6

        state = pybullet.getLinkState(robot, last_link, computeLinkVelocity=0, physicsClientId=self._physicsClientId)

        # self._observation = np.concatenate((joints_angle, joints_vel, distance_to_goal), axis=None)

        robot_links = []
        for i in range(1, 7):
            linkId = i
            state_1 = pybullet.getLinkState(robot, linkId, computeLinkVelocity=1, physicsClientId=self._physicsClientId)
            linPos = np.array(state_1[0])
            linRot = np.array(state_1[1])
            linVel = np.array(state_1[6])
            angVel = np.array(state_1[7])
            link_state = np.concatenate((linPos, linRot, linVel, angVel), axis=None)
            robot_links.append(link_state)

        self._tool_position = robot_links[-1][:3]
        self._tool_rotation = robot_links[-1][3:7]

        self._observation = np.concatenate((joints_angle, joints_vel, robot_links[0], robot_links[1], robot_links[2],
                                            robot_links[3], robot_links[4], robot_links[5], self._traj), axis=None)

        # self._observation = np.concatenate((joints_angle, joints_vel, distance_to_goal, ray_distances), axis=None)

        self.state = self._observation

        return self.state

    def getState(self):
        self.state = np.array(self._observation)

        return np.array(self._observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def getRobot(self):
        return self.robot

    def getReward(self, robot):
        if self._maxtorque > self._torquelimit:
            return -999.9
        # self.changeWallColor(rgb = [1., 1., 1.])
        return -1*self._currentCT

    def getEnvCounter(self):
        return self._envCounter

    def getDone(self, robot):
        # Remove the Counter from limiting the Done.|
        # if self.CheckCollision(robot)==True or self._envCounter>self._maxCounter or self.CheckJointLimits(robot):
        if self._maxtorque < self._torquelimit:
            return True
        else:
            return False

    def step(self, action):

        self._currentCT = self._ict + action

        self._maxtorque = ComputeMaxTorque(self, robot)

        for _ in range(1):
            pybullet.stepSimulation(physicsClientId=self._physicsClientId)

        self.setState(self.robot)

        self._envCounter = self._envCounter + 1

        reward = self.getReward(self.robot)

        done = self.getDone(self.robot)

        return (self.getState()), reward, done, {}

    def updateVelocity(self, robot):
        actual_joints_vel = np.array([0., 0., 0., 0., 0., 0.])

        for i in range(6):
            _, actual_joint_vel, _, _ = pybullet.getJointState(robot, i + 1, physicsClientId=self._physicsClientId)

            actual_joints_vel[i] = actual_joint_vel

        self.CurrentJointVelocity = actual_joints_vel

    def getTrjectory(self, dynamic_jointangle):
        self._traj = dynamic_jointangle
        return self._traj

    def getCurrentVelocity(self):
        return self.CurrentJointVelocity

    def reset(self):
        self.CurrentJointVelocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.reset_flags()

        self.ResetRobot(self.robot)

        self.setState(self.robot)

        return self.state

    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: render to the current display or terminal and

          return nothing. Usually for human consumption.

        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),

          representing RGB values for an x-by-y pixel image, suitable

          for turning into a video.

        - ansi: Return a string (str) or StringIO.StringIO containing a

          terminal-style text representation. The text can include newlines

          and ANSI escape sequences (e.g. for colors).

        Note:

            Make sure that your class's metadata 'render.modes' key includes

              the list of supported modes. It's recommended to call super()

              in implementations to use the functionality of this method.

        Args:

            mode (str): the mode to render with

        """

        s = "position: {:2d}  reward: {:2d}  info: {}"

        print(s.format(self.state, self.reward, self.info))

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when

        garbage collected or when the program exits.

        """

        pass


def main():
    env = TrajbotEnv(gym.Env, connect_type="GUI")
    env.reset()
    while True:
        env.step(np.array([0.]))


if __name__ == "__main__":
    main()
