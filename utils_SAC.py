import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import pdb
import json

def build_net(layer_shape, hidden_activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hidden_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
		super(Actor, self).__init__()
		layers = [state_dim] + list(hid_shape)

		self.a_net = build_net(layers, hidden_activation, output_activation)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20

	def forward(self, state, deterministic, with_logprob):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
		# we learn log_std rather than std, so that exp(log_std) is always > 0
		std = torch.exp(log_std)
		dist = Normal(mu, std)
		if deterministic: u = mu
		else: u = dist.rsample()

		'''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
		a = torch.tanh(u) #range [-1,1]
		if with_logprob:
			# Get probability density of logp_pi_a from probability density of u:
			# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
		else:
			logp_pi_a = None

		return a, logp_pi_a

class Double_Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Double_Q_Critic, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]

		self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		q2 = self.Q_2(sa)
		return q1, q2

#reward engineering for better training
def Reward_adapter(r, EnvIdex):
	# For Pendulum-v0
	if EnvIdex == 0:
		r = (r + 8) / 8

	# For LunarLander
	elif EnvIdex == 1:
		if r <= -100: r = -10

	# For BipedalWalker
	elif EnvIdex == 4 or EnvIdex == 5:
		if r <= -100: r = -1
	return r


# def Action_adapter(a):
# 	#from [-1,1] to [1.0,10.0]
# 	# from [-1,1] to [2.5,50.0]
# 	# return  np.clip((a+1.0)/2.0,0.0,1.0)
# 	return  np.clip((a+1.0)*4.5+1,0.0,1.0)
def Action_adapter(a):
	#from [0,1] to [-max,max]
	# return 2 * (a - 0.5) * max_action
	# from [0,1] to [2.5,50]
	return (50-2.5)*a+2.5

def Action_adapter_reverse(act):
	#from [2.5,50.0] to [-1,1]
	return  np.clip((act-1)/4.5-1,-1.0,1.0)


def evaluate_policy(env, agent, turns = 3):
	total_scores = 0
	for j in range(turns):
		s,info= env.reset()
		done = False
		while not done:
			if env.terminate:
				break
			# Take deterministic actions at test time
			a = agent.select_action(s, deterministic=True)
			s_next, r, dw, info = env.step(a)
			done = dw

			total_scores += r
			s = s_next
	return int(total_scores/turns)


def evaluate_policy_randomskill(env, agent):
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
			a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
			act = Action_adapter(a)  # [0,1] to [-max,max]
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

def evaluate_policy_plot(env, agent, turns = 1):
	total_scores = 0

	for j in range(turns):
		s, info = env.reset()
		pdb.set_trace()
		done = False
		time_all = []
		agent1_action = []
		agent2_action = []

		kinova_tra = []
		ur5e_tra = []


		while not env.terminate:
			# Take deterministic actions at test time
			a = agent.select_action(s, deterministic=True)
			print('action:',a)

			a = Action_adapter(a)

			if env.flag_conv:
				a = np.array([1.0, 1.0])


			if env.flag_conv2:
				a = np.array([1.0, 1.0])



			s_next, r, dw, info = env.step(a)
			done = dw

			total_scores += r
			s = s_next

			time_all.append(env.total_time)

			# action=a
			agent1_action.append(a[0])
			agent2_action.append(a[1])

			kinova_tra.append(env.end_effector1.tolist())
			ur5e_tra.append(env.end_effector2.tolist())


			# if action == 0:
			# 	agent1_action.append(1)
			# 	agent2_action.append(0)
			# # return [0, 1]
			# elif action == 1:
			# 	agent1_action.append(0)
			# 	agent2_action.append(1)
			# # return [1, 0]
			# elif action== 2:
			# 	agent1_action.append(1)
			# 	agent2_action.append(1)

			print('goal1 achieve:', env.flag_conv)
			print('goal2 achieve:', env.flag_conv2)

			# if env.flag_conv:
			# 	agent2_action[-1]=1
			#
			#
			# if env.flag_conv2:
			# 	agent1_action[-1]=1

		plt.figure(figsize=(8, 6))  # 8 inches wide, 6 inches tall
		# Plot action1 with a dashed blue line and larger line width
		plt.plot(time_all, agent1_action, color='blue', linestyle='-', linewidth=2, label='robot1 go/no go')

		# Plot action2 with a solid red line and smaller line width
		plt.plot(time_all, agent2_action, color='red', linestyle='-', linewidth=2, label='robot2 go/no go')

		plt.xticks(fontsize=20)
		plt.yticks(fontsize=20)

		# Add legend to differentiate the two lines
		plt.legend(loc='best', fontsize=20)

		# Label the axes
		plt.xlabel('Time (s)', fontsize=20)
		plt.ylabel('Action Value', fontsize=20)

		# Add title to the plot
		plt.title('Action Values over Time', fontsize=20)

		# Save the figure as a PNG image
		plt.savefig('action_plot_sac_frp.png', format='png', dpi=300)  # Save with 300 DPI for better quality
		print('time',env.total_time)

		with open('kinova_tra_SAC.json', 'w') as json_file:
			json.dump(kinova_tra, json_file)

		with open('ur5e_tra_SAC.json', 'w') as json_file:
			json.dump(ur5e_tra, json_file)



	return total_scores/turns


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