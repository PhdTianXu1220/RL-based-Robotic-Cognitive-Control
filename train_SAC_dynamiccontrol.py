from utils_SAC import str2bool, evaluate_policy, evaluate_policy_plot, Action_adapter, Action_adapter_reverse, Reward_adapter,evaluate_policy_randomskill
from datetime import datetime
from SAC import SAC_countinuous
import gymnasium as gym
import os, shutil
import argparse
import torch
# from ur5e_env3 import  UR5eEnv
from DynamicControlEnv_ML_EN_normact import ControlModuleEnv
# from normalization_mappo import Normalization
import numpy as np
from JudgeModule_load import RewardPredictor,action_select


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda:1', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=2, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=10000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=10, help='Training Fraquency, in stpes')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=5e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=5e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


'''load judgement module'''
obs_dim = 8
act_dim = 1
# Setup device
judge_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Test on {judge_device}")

# Initialize the model
judge_model = RewardPredictor(obs_dim, act_dim).to(judge_device)
judge_model.load_model()
judge_model.eval()

# reward_norm = Normalization(1)


def main():
    # EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    # BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']
    max_score=-1000


    env = ControlModuleEnv(GUI_flag=False)
    # eval_env = DualArmEnv(GUI_flag=True)
    opt.state_dim = 22+2
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    print('max_action:',opt.max_action)
    opt.max_e_steps = 2e3
    print('Algorithm: SAC', '  Env:', 'DynamicControl', '  state_dim:', opt.state_dim,
          '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'sac_writer/SAC' + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('sac_model'): os.mkdir('sac_model')
    agent = SAC_countinuous(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel:
        print('load model')
        agent.load(opt.ModelIdex)

    if opt.render:
        while True:
            print('start rollout')
            score = evaluate_policy_randomskill(env, agent)
            print('EnvName:', 'Move', 'score:', score)
    else:
        total_steps = 0
        i_episode = 0



        while total_steps < opt.Max_train_steps:
            # print('total_step:', total_steps)
            # print('episode', i_episode)
            s,info = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
            # judge_obs = np.concatenate((np.array([env.mass]), np.array([env.friction]),
            #                       np.array(env.block_start_position_random),
            #                       np.array(env.block_target_position_random)))
            #
            # skill_select,predicts=action_select(judge_obs,judge_model,judge_device)
            # skill_feasible=predicts[skill_select-1]

            skills = [1, 2, 3]
            skill_select = np.random.choice(skills)

            print("mass", env.mass, "friction param", env.friction, "skill_select", skill_select)

            # print("obs",judge_obs,"skill_select",skill_select,"skill_feasible",skill_feasible)

            # env_seed += 1
            done = False
            reward_sum=0

            while not env.start_dmp_flag:
                # if skill_feasible<0.6:
                #     print("no feasible skill exist for current scene")
                #     done=True
                #     break


                act=[1.0]
                act_env = np.concatenate((np.array([skill_select]), np.array(act)))
                s_next, r, dw, info = env.step(act_env)
                s = s_next
                if env.env_end_flag:
                    done = True
                    print("initial grip process fail")
                    break



            print("start record actions")



            '''Interact & trian'''
            while not done:
                if env.env_end_flag:
                    break


                if total_steps < (opt.max_e_steps):
                    a = env.action_space.sample()  # act∈[-max,max]
                    act = Action_adapter(a)  # a∈[-1,1]
                    print('random action')
                else:
                    a = agent.select_action(s, deterministic=False)  # a∈[-1,1]
                    # act=a
                    act = Action_adapter(a)  # act∈[-max,max]
                    print('learned action')

                print('action:', a,type(a))
                act_env=np.concatenate((np.array([skill_select]),np.array(act)))

                print("act: execution ratio:", 10 / act)

                s_next, r, dw,  info = env.step(act_env)  # dw: dead&win; tr: truncated
                # r = Reward_adapter(r, opt.EnvIdex)
                tr = env.env_end_flag

                if env.start_dmp_flag:
                    print("save experience")
                    reward_sum+=r
                    done = (dw or tr)
                    dw = torch.tensor(dw, dtype=torch.bool)


                    agent.replay_buffer.add(s, a, r, s_next, dw)
                    s = s_next
                    total_steps += 1





                    '''train if it's time'''
                    # train 50 times every 50 steps rather than 1 training per step. Better!
                    if (total_steps >= opt.max_e_steps) and (total_steps % opt.update_every == 0):
                        # print('start training')
                        for j in range(opt.update_every):
                            agent.train()

                    '''record & log'''


                    if opt.write and (total_steps % opt.eval_interval== 0):
                        score = evaluate_policy_randomskill(env, agent)
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('alpha', agent.alpha, global_step=total_steps)
                        writer.add_scalar('actor loss', agent.a_loss, global_step=total_steps)
                        writer.add_scalar('critic loss', agent.q_loss, global_step=total_steps)
                    # print(f'EnvName:Move, Steps: {int(total_steps/10000)}k, Episode Reward:{reward_sum}')

                    '''save model'''
                    if total_steps % opt.save_interval == 0:
                        agent.save(timestep=int(total_steps/10000))


            if reward_sum >= max_score:
                agent.save(int(1e4))
                max_score = reward_sum


            print('episode:',i_episode,'reward_sum:',reward_sum)
            writer.add_scalar('reward_sum_episode', reward_sum, global_step=i_episode)
            writer.add_scalar('reward_sum_step', reward_sum, global_step=total_steps)
            i_episode+=1
        # env.close()
        # eval_env.close()


if __name__ == '__main__':
    main()