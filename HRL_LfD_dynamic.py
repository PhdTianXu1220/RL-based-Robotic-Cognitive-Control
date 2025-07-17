import stat
from threading import activeCount
import numpy as np
from utilities import *
from load_from_excel import *
import sys
import random
import matplotlib.pyplot as plt
import random
import gym
#from trajbotenv_goal_rays import TrajbotEnv
from dynamic_env import TrajbotEnv
import sac_lfd_dynamics
import agent_ori as V2_sac_agent
from collections import deque
import multiprocessing
import torch
import time
from mapping_v1 import mapping

import multiprocessing
from multiprocessing.queues import Queue as mp_queue
from sac_ori import __main__

# load demo waypoints and orientations
demo_waypoints, demo_orientations = load_demo_library('demo_library.xlsx')

print(demo_waypoints)
print(demo_orientations)

# load task waypoits and orientations
task_waypoints, task_orientations = load_task_configuration('task_configuration.xlsx')

print(task_waypoints)
print(task_orientations)

task_waypoints = np.array(task_waypoints)
task_orientations = np.array(task_orientations)

# exit()

# task_waypoints = np.array([[0, -0.3, 0.2], [0, -0.4, 0.2], [0, -0.5, 0.2], [0, -0.6, 0.2]])
# task_orientations = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])


dr1 = []

# Converting demo orientations from Euler to Quaternion Convertion

for i in range(demo_orientations.shape[0]):
    dr1_i = []
    for j in range(len(demo_orientations[i])):
        rotation = euler_to_quaternion(demo_orientations[i][j])
        dr1_i.append(rotation)
    dr1.append(dr1_i)

# Calculating Qproduct of demo orientations

deltadr1 = []
for i in range(len(dr1)):
    deltadr1_i = []
    for j in range(len(dr1[i])):
        deltadr1_i.append(qproduct(qconj(dr1[i][j]), dr1[-1][j]))
    deltadr1.append(deltadr1_i)

#  Run Q-learning algorithm

# Initialize cumulative reward for 20 trial and 1000 iterations
shape = (100, 1000)
cumul_reward = np.zeros(shape)

# Reinforcement learning starts


for trial in range(100):

    statecount = 0
    epsilon = 0.8
    alpha = 0.01
    gamma = 0.1
    tolerancea = 0.1
    st_index = 0

    state = []
    action_pair = []  # I include this to search a action index while searching the maximum Q-value or updating the Q-Table. action_parir list holds the all posible actions for a given demo library and set fo task constraints.

    Q_value = np.random.randint(-999, 0, size=(4, (task_waypoints.shape[0] - 1) * demo_waypoints.shape[0]),
                                dtype=np.int32)

    action_pair = np.empty(((task_orientations.shape[0] - 1) * demo_orientations.shape[0], 2), dtype=np.int32)

    a = 0
    for i in range(1, task_orientations.shape[0]):
        for j in range(0, demo_orientations.shape[0]):
            action_pair[a][0] = i
            action_pair[a][1] = j
            a += 1

    print(action_pair)
    matching_row_indices = np.where((action_pair == [3, 1]).all(axis=1))[0][0]

    for iteration in range(1000):

        action_count = 0
        currenttaskconf_index = 0
        cumul_reward[trial, iteration] = 0
        shape = (task_waypoints.shape[0], (task_waypoints.shape[0] - 1) * demo_waypoints.shape[0])
        reward = np.zeros(shape)

        # Completes one episode
        while currenttaskconf_index < task_orientations.shape[0] - 1:
            st_orientations = task_orientations[currenttaskconf_index:, :]
            st_waypoints = task_waypoints[currenttaskconf_index:, :]

            if statecount == 0:
                state.append(st_orientations)
                statecount += 1
            else:
                for i in range(len(state)):
                    if np.array_equal(state[i], st_orientations):
                        st_index = i
                        break
                    elif i == (len(state) - 1) and not np.array_equal(state[i], st_orientations):
                        # state[statecount] = st_orientations
                        state.append(st_orientations)
                        statecount += 1

            epsilon_greedy = epsilon * (1.0 / (iteration + 1))
            random1 = np.random.rand()

            # Constructing action array
            a1 = np.arange(currenttaskconf_index, task_orientations.shape[0])
            a2 = np.arange(demo_orientations.shape[0])

            actioncount = 0
            action_indexs = []
            A = np.empty((2, (len(a1) - 1) * len(a2)), dtype=np.int32)
            for i in range(a1[1], a1[-1] + 1):
                for j in range(len(a2)):
                    A[0, actioncount] = i
                    A[1, actioncount] = a2[j]
                    action_indexs.append(
                        np.where((action_pair == [A[0, actioncount], A[1, actioncount]]).all(axis=1))[0][0])
                    actioncount += 1
            print(action_indexs)

            # Selecting an action

            if random1 <= epsilon_greedy:
                at_index = random.randint(0, actioncount - 1)
                action_index = np.where((action_pair == [A[0, at_index], A[1, at_index]]).all(axis=1))[0][0]
                print('randomly selection at_index = {0}\n'.format(at_index))
                print(action_index)

            else:
                max = Q_value[st_index, action_indexs[0]]
                action_index = action_indexs[0]
                for i in range(action_indexs[0], action_indexs[-1] + 1):
                    if (Q_value[st_index, i] > max):
                        max = Q_value[st_index, i]
                        action_index = i

                print(action_index)
                at_index = action_indexs.index(action_index)
                print('Q_value-based selection at_index = {0}\n'.format(at_index))
                print(action_index)

            # Check the new task segment after the selected action
            seg_task_waypoints = task_waypoints[currenttaskconf_index:A[0, at_index] + 1, :]
            seg_task_orientations = task_orientations[currenttaskconf_index:A[0, at_index] + 1, :]

            if iteration > 990:
                # sequence[trial, iteration - 990] = A[:, at_index] #### Not clear
                action_count += 1

            # Check the demo  selected by the action
            select_demo_waypoints = demo_waypoints[A[1, at_index]]
            select_demo_orientations = demo_orientations[A[1, at_index]]

            dr_taskseg = []

            # Euler to Quaternion Convertion of the task segments selected by the action

            for i in range(len(seg_task_orientations)):
                dr_taskseg.append(euler_to_quaternion(seg_task_orientations[i]))

            # Calculating Qproduct of the task segments selected by the action

            deltadr_taskseg = []
            for i in range(len(dr_taskseg) - 1):
                deltadr_taskseg.append(qproduct(qconj(dr_taskseg[i]), dr_taskseg[-1]))

            # Qproduct of the selected demo (demo selected by the action)
            deltadr_demoselect = deltadr1[A[1, at_index]]

            # calculating similary and deviation for the selected task segments and demo
            similarity, mappingpoint, deviation = semantically_similar(deltadr_taskseg, deltadr_demoselect, tolerancea)

            # calculating reward based on the similarity
            if similarity == 0:
                reward[st_index, at_index] = -999
                cumul_reward[trial, iteration] -= 999
            else:
                sac_lfd_dynamics.main()
                the_dir1 = "/home/lambda1/LfD/src_v2/backup/Tian/results/model1/"
                search_dir = the_dir1
                list_df_ele = []
                lower_reward = []
                # remove anything from the list that is not a file (directories, symlinks)
                # thanks to J.F. Sebastion for pointing out that the requirement was a list
                # of files (presumably not including directories)
                files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
                files.sort(key=lambda x: os.path.getmtime(x))
                for filename in files:
                    trunc_string = re.findall("__.*?\.pth", filename)[0]
                    the_average_reward = re.findall("R_(\d+.\d+)", trunc_string)[0]
                    trunc_string_float = float(the_average_reward)
                    lower_reward.append(trunc_string_float)
                reward[st_index, at_index] = deviation + lower_reward[-1]
                cumul_reward[trial, iteration] += reward[st_index, at_index]
                mapping(select_demo_waypoints, select_demo_orientations, seg_task_waypoints[0], seg_task_waypoints[-1], seg_task_orientations[-1], 20)

            currenttaskconf_index = A[0, at_index]
            stplus1_orientations = task_orientations[currenttaskconf_index:, :]

            # Adding the resulting state to state array
            stplus1_index = -1
            for i in range(0, len(state)):
                if np.array_equal(state[i], stplus1_orientations):
                    stplus1_index = i
                    break
                elif i == (len(state) - 1) and not np.array_equal(state[i], stplus1_orientations):
                    state.append(stplus1_orientations)
                    stplus1_index = statecount
                    statecount += 1

            # Updating the Q_values based on the selected action

            Q_value[st_index, action_index] += alpha * (
                        reward[st_index, action_index] + gamma * np.max(Q_value[stplus1_index, :]) - Q_value[
                    st_index, action_index])
            currenttaskconf_index = A[0, at_index]

# Calculating average reward


average_reward = np.mean(cumul_reward, axis=0)

print(average_reward)

# Plotting
plt.figure(3)
plt.plot(average_reward)
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title('Iteration vs Reward')
plt.grid(True)
plt.show()
