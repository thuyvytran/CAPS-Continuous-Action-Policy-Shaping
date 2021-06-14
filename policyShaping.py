import gym
import random
import pickle
import os.path
import math
import glob
import time
import stopwatch
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pdb

#from atari_wrappers import make_atari, wrap_deepmind,LazyFrames
#from IPython.display import clear_output
from tensorboardX import SummaryWriter

#from Oracle import TorchOracle as TO
#from PolicyShaping import get_shaped_action, get_argmax_action

from PS_utils import sample_normal
from sac_torch import Agent
from buffer import ReplayBuffer

import threading


"""
The following class was adopted from stackoverflow  user timgeb 
from url: https://stackoverflow.com/questions/24000455/python-how-to-get-input-from-console-while-an-infinite-loop-is-running
"""


class FeedbackThread(threading.Thread):
    def __init__(self):
        super(FeedbackThread, self).__init__()
        self.daemon = True
        self.last_user_input = None

    def run(self):
        while True:
            self.last_user_input = input(
                'Feedback: press g for good and l for bad: ')

    def reset(self):
        self.last_user_input = None


"""
ActionQueue is a class for credit assignment. It keeps track of the last "size" actions
and their time stamps. Then when feedback is received, it calculates if those actions 
fall within the credit assignment interval and adds them to the replay buffer if so. 
After adding to the replay buffer, it learns from the actions it just addded. 
"""


class ActionQueue:
    def __init__(self, size=5):
        self.size = size
        self.queue = []

    def enqeue(self, action_memory):
        if len(self.queue) > self.size:
            self.queue.pop()
            self.queue.insert(0, action_memory)
        else:
            self.queue.insert(0, action_memory)

    def push_to_buffer_and_learn(self, agent, actor, tf, feedback_value, interval_min=0, interval_max=.8):
        if len(self.queue) == 0:
            return None
        else:
            avg_loss = []
            i = 0
            for action_memory in self.queue:
                b = tf-action_memory[2]
                a = tf - action_memory[3]
                # feedback must occur within 0.2-4 seconds after feedback to have non-zero importance weight
                #print(a, b)
                pushed = (b >= interval_min and a <= interval_max)
                # push: state, action, ts, te, tf, feedback
                # [observation, action, ts, te, tf, feedback_value, mu, sigma, old_observation, done]
                if pushed:
                    agent.remember(
                        action_memory[8], action_memory[1], feedback_value, action_memory[0], action_memory[9])
                    #loss = agent.learn_from_feedback(action_memory[6], action_memory[7], action_memory[1], action_memory[5])
                    # print("banana")
                    #print(f"ts: {action_memory[2]}. te: {action_memory[3]}, tf: {tf}")
                    # avg_loss.append(loss)
                    # print(feedback_value)
                i += 1
            return np.mean(np.array(avg_loss))


for j in range(10,30):
    #env = gym.make('BipedalWalker-v3')
    env = gym.make('BipedalWalker-v3')

    # Note: these credit assignment intervals impact how the agent behaves a lot.
    # Because of this sensitivity the model is overall very sensitive.
    # There is a frame time delay of .1 so teaching is not boring. Could make the
    # the agent much better if played at around 10 frames per second (not sure of  current fps)
    interval_min = .1
    interval_max = .8

    episodes = 300
    USE_CUDA = torch.cuda.is_available()
    learning_rate = .001
    replay_buffer_size = 100000
    learn_buffer_interval = 200  # interval to learn from replay memory
    batch_size = 200
    print_interval = 1000
    log_interval = 1000
    learning_start = 100
    # win_reward = 21     # Pong-v4
    win_break = True
    queue_size = 1000



    #actor = Agent(alpha=0.000314854, beta=0.000314854, input_dims=env.observation_space.shape, env=env, batch_size=128,
                #tau=0.02, max_size=500000, layer1_size=400, layer2_size=300,
                #n_actions=env.action_space.shape[0], reward_scale=1, auto_entropy=True)


    actor = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape, env=env, batch_size=256,
                tau=0.005, n_actions=env.action_space.shape[0], 
                reward_scale=2, auto_entropy=False)

    # actor = Agent(input_dims=env.observation_space.shape, env=env,
    # n_actions=env.action_space.shape[0], reward_scale=2, auto_entropy=False)

    agent = Agent(alpha=.001, beta=.001, max_size=100000, input_dims=env.observation_space.shape, batch_size=256, env=env,
                n_actions=env.action_space.shape[0], reward_scale=10, auto_entropy=False)

    #agent.actor = copy.deepcopy(actor.actor)
    #agent.reward_scale = 10
    #agent.memory = ReplayBuffer(100000, env.observation_space.shape, env.action_space.shape[0])


    """
    def __init__(self, alpha=0.001, beta=0.001, input_dims=[8],
                env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
    """
    """
    hp = {'batch_size': 128, 'buffer_size': 50000, 'gamma': 0.99, 
        'learning_starts': 1000, 'log_std_init': 0.409723, 
        'lr': 0.000314854, 'net_arch': 'medium', 
        'tau': 0.02, 'ent_coef': 'auto', 'target_entropy': 'auto',
        'train_freq': 128}
    """
    """ 
    def sac1(args, env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
            steps_per_epoch=5000, epochs=100, replay_size=int(2e6), gamma=0.99, reward_scale=1.0,
            polyak=0.995, lr=5e-4, alpha=0.2, batch_size=200, start_steps=10000,
            max_ep_len_train=1000, max_ep_len_test=1000, logger_kwargs=dict(), save_freq=1)
    """
    frame = env.reset()


    episode_rewards = []
    all_rewards = []
    sum_rewards = []
    kl_sums = []
    losses = []
    episode_num = 0
    is_win = False

    #stopwatch = stopwatch.Stopwatch()

    #feedback_thread = FeedbackThread()
    #feedback_thread.start()

    cnt = 0
    start_f = 0
    end_f = 0

    action_queue = ActionQueue(queue_size)

    rewards = []
    env_interacts_arr = []
    e = 0.05
    render = False
    env_interacts = 0
    env_only = False
    best = -500
    agent_learn = True
    kappa = 1
    time_since = 1

    for i in range(episodes):
        start_f = end_f
        #stopwatch.restart()
        loss = 0
        # if render:
        # env.render()
        # time.sleep(.1)
        observation = env.reset()
        ep_rewards = 0
        feedback_value = 0
        while(True):
            # print(feedback_value)
            #stopwatch.start()
            # if episode_num > 100:
            #if render:
            #env.render()
            #print(observation[0])
            #time.sleep(.2)
            end_f += 1
            #ts = stopwatch.duration
            ts=0
            #if episode_num < 150:
            # if env_only == False:
            # if(env_interacts%10 == 0 or env_interacts == 1):
            if agent_learn:
                action, dist, mu, sigma, kl_total = sample_normal(
                    agent, actor, observation, with_noise=False, max_action=env.action_space.high, env_only=False, kappa=.9)
            else:
            #env_only = True
                action, dist, mu, sigma, kl_total = sample_normal(
                    agent, actor, observation, with_noise=False, max_action=env.action_space.high, env_only=False, with_grad_agent=False, kappa=.05)
                #action, dist, mu, sigma = agent.sample_action(observation)
            # print(action)
            old_observation = observation
            observation, reward, done, _ = env.step(action)
            actor.remember(old_observation, action, reward, observation, done)
            episode_rewards.append(reward)
            env_interacts+=1
            ep_rewards += reward
            kl_sums.append(kl_total)
            #te = stopwatch.duration
            te=0
            tf = 0
            # time.sleep(e) # Delay to make the game seeable
            """feedback = ""
            feedback = feedback_thread.last_user_input
            #feedback_value = 0
            if feedback == "/" or feedback == "'":
                if feedback == "/":
                    feedback_value = 1
                else:
                    feedback_value = -1
                tf = stopwatch.duration
                # feedback_thread.reset()
            if feedback == "y":
                feedback_value = 0"""

            #action_queue.enqeue([observation, action, ts, te, tf,
                                #feedback_value, mu, sigma, old_observation, done])

            # Oracle
            if np.random.random() > -1:
                if (observation[2] > .1 and observation[3] > -.1 and (observation[13] == 0 or observation[8] == 0) and reward>-90):
                    feedback_value = 1
                    agent.remember(old_observation, action,
                                feedback_value, observation, done)
                    # print(observation[3])
                else:
                    feedback_value = -1
                    agent.remember(old_observation, action,
                                feedback_value, observation, done)
            # if episode_num > 150:
                #env_only = True
            else:
                if observation[2] > .1 and (observation[13] == 0 or observation[8] == 0) and reward > -95:
                    feedback_value = -1
                    agent.remember(old_observation, action,
                                feedback_value, observation, done)
                else:
                    feedback_value = 1
                    agent.remember(old_observation, action,
                                feedback_value, observation, done)

            """if feedback_value != 0:
                tf = stopwatch.duration
                # [observation, action, ts, te, tf, feedback_value, mu, sigma, old_observation, done]
                loss = action_queue.push_to_buffer_and_learn(agent, actor, tf, feedback_value)
                #print(feedback_value)
                agent.remember(old_observation, action, feedback_value, observation, done)
                #print(loss, "feedback loss")"""

            # if end_f % learn_buffer_interval == 0:
            #    loss = agent.learn_from_replay(batch_size)
            #    print(loss, "replay loss")
            """
            if feedback == "r":
                feedback_thread.reset()
                done = True

            if feedback == "e":
                e = 0

            if feedback == "p":
                e = 1

            if feedback == "q":
                e = 0.05

            if feedback == "z":
                e = 0.035

            if feedback == "m":
                env.close
                render = not render

            if feedback == "x":
                agent_learn = not agent_learn
                print(f"agent is learning {agent_learn}")

            if feedback == "t":
                env_only = not env_only
                print(f"enviornment only is {env_only}")"""
            # if env_interacts % 128 == 0:
            actor.learn()
            # else:
            # actor.learn(update_target=False)
            #if episode_num < 150:
            if agent_learn:
                agent.learn()

            
            #feedback_thread.reset()

            if done:
                """print("frames: %5d, reward: %5f, loss: %4f, episode: %4d" % (end_f-start_f, np.sum(episode_rewards),
                                                                                        loss, episode_num))"""
    
                #if episode_num >= 150:
                    #time_since+=1
                    #print(.9*((1-.01)**time_since))
                print(ep_rewards)
                print("Episode:", str(i), "Run: ", str(j))
                rewards.append(ep_rewards)
                #if episode_num > 150:
                    #render = True
                #if episode_num >= 100:
                    #agent_learn = False
                #if episode_num == 100:
                    #actor.auto_entropy = True
                env_interacts_arr.append(env_interacts)
                if ep_rewards > best:
                    best = ep_rewards
                    print(f"BEST: {best}")
                ep_rewards = 0
                all_rewards.append(episode_rewards)
                sum_rewards.append(np.sum(episode_rewards))
                episode_rewards = []
                episode_num += 1
                avg_reward = float(np.mean(episode_rewards[-10:]))
                feedback_value = 0
                frame = env.reset()
                #feedback_thread.reset()
                # evaluate env:

                """with torch.no_grad():
                    eval_rewards = []
                    num_episodes = 2
                    for j in range(num_episodes):
                        ep_rew = []
                        env.reset()
                        while(True):
                            action, dist, mu, sigma = sample_normal(
                                agent, actor, observation, with_noise=False, max_action=env.action_space.high, env_only=True, with_grad=False)
                            observation, reward, done, _ = env.step(action)
                            ep_rew.append(reward)
                            if done:
                                eval_rewards.append(np.sum(ep_rew))
                                break
                    print(f"Eval Score: {np.mean(eval_rewards)}")"""

                break

    #np.save(f"BP_KL_forward_final{j}", rewards)
    #np.save(f"BP_kl_sums_data_{j}", kl_sums)
    #np.save(f"KL_envInteracts_max_{j}", env_interacts_arr)
