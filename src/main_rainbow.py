import random
import gymnasium as gym
import numpy as np
import torch
from Atari_Agents import Atari_Agents
from pettingzoo.atari import boxing_v2
import EnvPreprocess
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for debugging CUDA code

# SEED = 0
SEED = 12

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def train_boxing():

    # environment 
    # env = boxing_v2.parallel_env()
    env = boxing_v2.parallel_env(render_mode="human")

    # preprocess the environment
    env = EnvPreprocess.preprocess_boxing(env)

    # set seed 
    np.random.seed(SEED)
    random.seed(SEED)
    seed_torch(SEED)
    
    # parameters
    num_frames = 30000
    memory_size = 5000
    batch_size = 16
    target_update = 100
    init_buffer_fill = 1500
    gamma = 0.93

    # define a suppport - might have to increase number of atoms
    v_min = -50
    v_max = 50
    atom_size = 51

    # define the architecture type
    architectureType = "xtra-small"

    # define path to save models
    PATH = "../results/models/1_VS_MOVE/" + architectureType
    
    agents = Atari_Agents(env, memory_size, batch_size, target_update, SEED, v_min=v_min, v_max=v_max,
                          atom_size=atom_size, archType=architectureType, gamma=gamma, PATH=PATH)
    agents.train(num_frames, init_buffer_fill=init_buffer_fill)

                                        # test the agent
    # create a new env with rendering                                    
    env.close()
    env = boxing_v2.parallel_env(render_mode="human")
    env = EnvPreprocess.preprocess_boxing(env, training=False)
    
    agents.test(env)

    env.close()


def test_boxing(PATH): # test boxing using saved models

    env = boxing_v2.parallel_env(render_mode="human")
    env = EnvPreprocess.preprocess_boxing(env, training=False)

    # parameters - dont matter
    memory_size = 300
    batch_size = 32
    target_update = 20

    # define a suppport - might have to increase number of atoms
    # IMPORTANT: make sure to use the same v_min and v_max as the one used in training
    v_min = -50
    v_max = 50
    atom_size = 51

        # set seed 
    np.random.seed(SEED)
    random.seed(SEED)
    seed_torch(SEED)

    # define the architecture type
    architectureType = "xtra-small"
    
    agents = Atari_Agents(env, memory_size, batch_size, target_update, SEED, v_min=v_min, v_max=v_max,
                          atom_size=atom_size, archType=architectureType)
    
    agents.load(PATH) # load the models
    agents.test(env)


    env.close()

if __name__ == "__main__":

    # train_boxing()
    test_boxing("../results/models/1_VS_NOOP/xtra-small.pt")


