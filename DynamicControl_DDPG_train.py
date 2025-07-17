from utils_DDPG import str2bool,evaluate_policy,evaluate_policy_randomskill,Action_adapter
from datetime import datetime
from DDPG import DDPG_agent
import gymnasium as gym
import os, shutil
import argparse
import torch
from DynamicControlEnv_ML_EN_normact import ControlModuleEnv
import numpy as np


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda:1', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=1e7, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=5e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=1e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--random_steps', type=int, default=5e4, help='random steps before trianing')
parser.add_argument('--noise', type=float, default=0.1, help='exploring noise')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def main():
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']

    # Build Env
    # env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    # eval_env = gym.make(EnvName[opt.EnvIdex])
    env = ControlModuleEnv(GUI_flag=False)
    eval_env= ControlModuleEnv(GUI_flag=False)

    opt.state_dim = 22+2
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    # print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
    #       f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{env._max_episode_steps}')

    max_score = -1000

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
        writepath = 'ddpg_writer/DDPG'+ timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = DDPG_agent(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(opt.ModelIdex)

    if opt.render:
        while True:
            score = evaluate_policy_randomskill(env, agent, opt.max_action)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    else:
        total_steps = 0
        i_episode=0

        while total_steps < opt.Max_train_steps:
            s, info = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
            # env_seed += 1
            skills = [1, 2, 3]
            skill_select = np.random.choice(skills)

            print("mass", env.mass, "friction param", env.friction, "skill_select", skill_select)

            done = False
            reward_sum = 0

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

                if total_steps < opt.random_steps: a = env.action_space.sample()
                else: a = agent.select_action(s, deterministic=False)

                act = Action_adapter(a, opt.max_action)  # [-1,1] to [-max,max]
                print("act: execution ratio:", 10 / act)

                act_env = np.concatenate((np.array([skill_select]), np.array(act)))

                s_next, r, dw, info = env.step(act_env) # dw: dead&win; tr: truncated
                tr=env.env_end_flag
                done = (dw or tr)

                reward_sum += r

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1

                '''train'''
                if total_steps >= opt.random_steps:
                    agent.train()

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy_randomskill(env, agent, opt.max_action)
                    if opt.write: writer.add_scalar('ep_r', score, global_step=total_steps)
                    print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, Episode Reward:{score}')

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(int(total_steps/1000))


            if env.start_dmp_flag:
                if reward_sum >= max_score:
                    agent.save(int(1e4))
                    max_score = reward_sum

                print('episode:', i_episode, "mass", env.mass, "friction param", env.friction,"skill", skill_select,"start",
                      env.block_start_position_random, "target", env.block_target_position_random, 'reward_sum:',
                      reward_sum)
                writer.add_scalar('reward_sum_episode', reward_sum, global_step=i_episode)
                writer.add_scalar('reward_sum_step', reward_sum, global_step=total_steps)
                i_episode+=1

        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()




