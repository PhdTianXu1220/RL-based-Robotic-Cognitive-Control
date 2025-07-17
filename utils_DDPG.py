import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

def Action_adapter(a,max_action):
	#from [0,1] to [-max,max]
	# return 2 * (a - 0.5) * max_action
	# from [-1,1] to [2.5,50]
	return 0.5*(50-2.5)*(a+1)+2.5


def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)

def evaluate_policy_randomskill(env, agent, max_action):
	total_scores = 0
	pos_set=[]
	ori_set=[]
	pos_set2=[]
	ori_set2=[]
	skill_list=[1,2,3]

	for j in range(len(skill_list)):
		s, info = env.reset()
		# print(type(env.end_effector_pos1))
		# print(type(env.end_effector_ori1))
		# pos_set.append(env.end_effector_pos1)
		# ori_set.append(env.end_effector_ori1)
		#
		# judge_obs = np.concatenate((np.array([env.mass]), np.array([env.friction]),
		# 							np.array(env.block_start_position_random),
		# 							np.array(env.block_target_position_random)))
		#
		# skill_select, predicts = action_select(judge_obs, judge_model, judge_device)
		skill_select=skill_list[j]

		# skill_feasible = predicts[skill_select - 1]
		print("selected skill:", skill_select)

		done = False

		while not env.start_dmp_flag:
			# if skill_feasible < 0.6:
			# 	print("no feasible skill exist for current scene")
			# 	done = True
			# 	break

			act = [1.0]
			act_env = np.concatenate((np.array([skill_select]), np.array(act)))
			s_next, r, dw, info = env.step(act_env)
			s = s_next

			# pos_set.append(env.end_effector_pos1)
			# ori_set.append(env.end_effector_ori1)


			if env.env_end_flag:
				done = True
				print("initial grip process fail")
				break

		# pdb.set_trace()


		while not done:
			a = agent.select_action(s, deterministic=True)
			act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
			act_env = np.concatenate((np.array([skill_select]), np.array(act)))
			print("act",act)
			s_next, r, dw, info = env.step(act_env)
			done = (dw or env.env_end_flag)

			total_scores += r
			s = s_next

			# pos_set2.append(env.end_effector_pos1)
			# ori_set2.append(env.end_effector_ori1)

		# with open('Kinova_pos_pre.json', 'w') as json_file:
		# 	json.dump(pos_set, json_file)
		#
		# with open('Kinova_ori_pre.json', 'w') as json_file:
		# 	json.dump(ori_set, json_file)
		#
		# with open('Kinova_pos.json', 'w') as json_file:
		# 	json.dump(pos_set2, json_file)
		#
		# with open('Kinova_ori.json', 'w') as json_file:
		# 	json.dump(ori_set2, json_file)

	return total_scores/len(skill_list)


#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')