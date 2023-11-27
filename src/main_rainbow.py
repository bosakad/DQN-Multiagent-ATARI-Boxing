import random
import gymnasium as gym
import numpy as np
import torch
from Atari_Agents import Atari_Agents
from pettingzoo.atari import boxing_v2
import EnvPreprocess

SEED = 42

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def train_boxing():

    # environment 
    env = boxing_v2.parallel_env()

    # preprocess the environment
    env = EnvPreprocess.preprocess_boxing(env)

    # set seed 
    np.random.seed(SEED)
    random.seed(SEED)
    seed_torch(SEED)
    
    # parameters
    num_frames = 1000
    memory_size = 10000
    batch_size = 4
    target_update = 100
    
    agents = Atari_Agents(env, memory_size, batch_size, target_update, SEED)
    agents.train(num_frames)


def train_cartPole():
    # environment
    env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")

    seed = 777

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    # parameters
    num_frames = 1000
    memory_size = 10000
    batch_size = 128
    target_update = 100

    # train
    agents = Atari_Agents(env, memory_size, batch_size, target_update, seed)

    agents.train(num_frames)

    video_folder="../results/videos/rainbow"
    agents.test(video_folder=video_folder)


if __name__ == "__main__":

    # train_cartPole()
    train_boxing()