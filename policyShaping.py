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
import matplotlib.pyplot as plt
# %matplotlib inline

import torch

from tensorboardX import SummaryWriter

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
                if pushed:
                    agent.remember(
                        action_memory[8], action_memory[1], feedback_value, action_memory[0], action_memory[9])
                i += 1
            return np.mean(np.array(avg_loss))


for j in range(10,30):
    
    env = gym.make('BipedalWalker-v3')

    # Define window for credit assignment
    interval_min = .1
    interval_max = .8

    episodes = 300
    USE_CUDA = torch.cuda.is_available()
    learning_rate = .001
    replay_buffer_size = 100000
    learn_buffer_interval = 200  # interval to learn from replay memory
    queue_size = 1000

    actor = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape, env=env, batch_size=256,
                tau=0.005, n_actions=env.action_space.shape[0], 
                reward_scale=2, auto_entropy=False)

    agent = Agent(alpha=.001, beta=.001, max_size=100000, input_dims=env.observation_space.shape, batch_size=256, env=env,
                n_actions=env.action_space.shape[0], reward_scale=10, auto_entropy=False)

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
            action, dist, mu, sigma, kl_total = sample_normal(
                agent, actor, observation, with_noise=False, max_action=env.action_space.high, env_only=False, kappa=.9)
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
            if (observation[2] > .1 and observation[3] > -.1 and (observation[13] == 0 or observation[8] == 0) and reward>-90):
                feedback_value = 1
                agent.remember(old_observation, action,
                            feedback_value, observation, done)
                # print(observation[3])
            else:
                feedback_value = -1
                agent.remember(old_observation, action,
                            feedback_value, observation, done)

            actor.learn()
            agent.learn()

            
            #feedback_thread.reset()

            if done:
                print(ep_rewards)
                print("Episode:", str(i), "Run: ", str(j))
                rewards.append(ep_rewards)
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
                break
