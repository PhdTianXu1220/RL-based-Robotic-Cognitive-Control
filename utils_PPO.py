import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import numpy as np
from JudgeModule_load import RewardPredictor,action_select
import json
import pdb

class BetaActor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(BetaActor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.alpha_head = nn.Linear(net_width, action_dim)
		self.beta_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))

		alpha = F.softplus(self.alpha_head(a)) + 1.0
		beta = F.softplus(self.beta_head(a)) + 1.0

		return alpha,beta

	def get_dist(self,state):
		alpha,beta = self.forward(state)
		dist = Beta(alpha, beta)
		return dist

	def deterministic_act(self, state):
		alpha, beta = self.forward(state)
		mode = (alpha) / (alpha + beta)
		return mode

class GaussianActor_musigma(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(GaussianActor_musigma, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.sigma_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		sigma = F.softplus( self.sigma_head(a) )
		return mu,sigma

	def get_dist(self, state):
		mu,sigma = self.forward(state)
		dist = Normal(mu,sigma)
		return dist

	def deterministic_act(self, state):
		mu, sigma = self.forward(state)
		return mu


class GaussianActor_mu(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(GaussianActor_mu, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.mu_head.weight.data.mul_(0.1)
		self.mu_head.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		return mu

	def get_dist(self,state):
		mu = self.forward(state)
		action_log_std = self.action_log_std.expand_as(mu)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist

	def deterministic_act(self, state):
		return self.forward(state)


class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v

def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		print('Wrong Input.')
		raise


def Action_adapter(a,max_action):
	#from [0,1] to [-max,max]
	# return 2 * (a - 0.5) * max_action
	# from [0,1] to [2.5,50]
	return (50-2.5)*a+2.5


def Reward_adapter(r, EnvIdex):
	# For BipedalWalker
	if EnvIdex == 0 or EnvIdex == 1:
		if r <= -100: r = -1
	# For Pendulum-v0
	elif EnvIdex == 3:
		r = (r + 8) / 8
	return r

def evaluate_policy(judge_model,judge_device,env, agent, max_action, turns):
	total_scores = 0

	success_count=0

	for j in range(turns):
		pos_set = []
		ori_set = []
		pos_set2 = []
		ori_set2 = []


		s, info = env.reset()
		print(type(env.end_effector_pos1))
		print(type(env.end_effector_ori1))
		pos_set.append(env.end_effector_pos1)
		ori_set.append(env.end_effector_ori1)

		judge_obs = np.concatenate((np.array([env.mass_obs]), np.array([env.friction_obs]),
									np.array(env.block_start_position_random),
									np.array(env.block_target_position_random)))

		skill_select, predicts = action_select(judge_obs, judge_model, judge_device)
		# skill_select=1

		skill_feasible = predicts[skill_select - 1]
		print("selected skill:", skill_select)

		done = False

		while not env.start_dmp_flag:
			if skill_feasible < 0.6:
				print("no feasible skill exist for current scene")
				done = True
				# break

			act = [1.0]
			act_env = np.concatenate((np.array([skill_select]), np.array(act)))
			s_next, r, dw, info = env.step(act_env)
			s = s_next

			pos_set.append(env.end_effector_pos1)
			ori_set.append(env.end_effector_ori1)


			if env.env_end_flag:
				done = True
				print("initial grip process fail")
				break

		# pdb.set_trace()


		while not done:
			a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
			act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
			act_env = np.concatenate((np.array([skill_select]), np.array(act)))
			print("act",act)
			s_next, r, dw, info = env.step(act_env)
			done = (dw or env.env_end_flag)

			total_scores += r
			s = s_next

			pos_set2.append(env.end_effector_pos1)
			ori_set2.append(env.end_effector_ori1)

		success_count+=env.task_success

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

	return total_scores/turns

def collect_data(env, agent, max_action, turns):
	observations = []
	actions = []
	rewards = []
	for i in range(1,3+1):
		skill_select=i
		print("skill ID", skill_select)
		for j in range(turns):
			s, info = env.reset()
			print("mass obs",env.mass_obs,"friction obs",env.friction_obs)
			judge_obs = np.concatenate((np.array([env.mass_obs]), np.array([env.friction_obs]),
										np.array(env.block_start_position_random),
										np.array(env.block_target_position_random)))
			observations.append(judge_obs)
			actions.append(np.array([skill_select]))

			done = False

			while not env.start_dmp_flag:

				act = [1.0]
				act_env = np.concatenate((np.array([skill_select]), np.array(act)))
				s_next, r, dw, info = env.step(act_env)
				s = s_next


				if env.env_end_flag:
					done = True
					print("initial grip process fail")
					break

			# pdb.set_trace()


			while not done:
				a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
				act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
				act_env = np.concatenate((np.array([skill_select]), np.array(act)))
				print("act",act)
				s_next, r, dw, info = env.step(act_env)
				done = (dw or env.env_end_flag)

				s = s_next



			rewards.append(env.task_success)

	observations = np.array(observations)
	actions = np.array(actions).reshape(-1, 1)  # Ensure actions are 2D
	rewards = np.array(rewards)

	return observations, actions, rewards

def evaluate_policy_new(judge_model, judge_device, env, agent, max_action, turns):
	total_scores = 0

	success_count = 0
	energy_sum=0
	time_sum=0

	for j in range(turns):
		print("trial time:",j)
		pos_set = []
		ori_set = []
		pos_set2 = []
		ori_set2 = []

		s, info = env.reset()
		print(type(env.end_effector_pos1))
		print(type(env.end_effector_ori1))
		pos_set.append(env.end_effector_pos1)
		ori_set.append(env.end_effector_ori1)

		judge_obs = np.concatenate((np.array([env.mass_obs]), np.array([env.friction_obs]),
									np.array(env.block_start_position_random),
									np.array(env.block_target_position_random)))

		skill_select, predicts = action_select(judge_obs, judge_model, judge_device)
		# skill_select=1

		skill_feasible = predicts[skill_select - 1]
		print("selected skill:", skill_select)

		done = False

		while not env.start_dmp_flag:
			if skill_feasible < 0.6:
				print("no feasible skill exist for current scene")
				done = True
			# break

			act = [1.0]
			act_env = np.concatenate((np.array([skill_select]), np.array(act)))
			s_next, r, dw, info = env.step(act_env)
			s = s_next

			pos_set.append(env.end_effector_pos1)
			ori_set.append(env.end_effector_ori1)

			if env.env_end_flag:
				done = True
				print("initial grip process fail")
				break

		# pdb.set_trace()

		while not done:
			a, logprob_a = agent.select_action(s, deterministic=True)  # Take deterministic actions when evaluation
			act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
			# act=act*2
			act_env = np.concatenate((np.array([skill_select]), np.array(act)))
			print("act", act)
			print("act: execution ratio:", 10 / act)
			s_next, r, dw, info = env.step(act_env)
			done = (dw or env.env_end_flag)

			total_scores += r
			s = s_next

			pos_set2.append(env.end_effector_pos1)
			ori_set2.append(env.end_effector_ori1)

		success_count += env.task_success
		if env.task_success:
			energy_sum +=env.energy
			time_sum+=env.execution_time



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

	return total_scores / turns, success_count/ turns, time_sum/success_count, energy_sum/success_count

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
			a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
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