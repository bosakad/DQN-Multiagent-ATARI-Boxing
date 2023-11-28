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
    num_frames = 500
    memory_size = 10000
    batch_size = 32
    target_update = 100
    
    agents = Atari_Agents(env, memory_size, batch_size, target_update, SEED)
    agents.train(num_frames)

                                        # test the agent
    # create a new env with rendering                                    
    env.close()
    env = boxing_v2.parallel_env(render_mode="human")
    env = EnvPreprocess.preprocess_boxing(env)
    
    video_folder="../results/videos/rainbow"
    agents.test(video_folder, env)

    env.close()


if __name__ == "__main__":

    # train_cartPole()
    train_boxing()