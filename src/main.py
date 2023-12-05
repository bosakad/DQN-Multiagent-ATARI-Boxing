import random
import gymnasium as gym
import numpy as np
import torch
from Atari_Agents import Atari_Agents
from pettingzoo.atari import boxing_v2
import EnvPreprocess
import os
import time
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for debugging CUDA code

# SEED = 0
SEED = 12

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def train_boxing(init_buffer_fill = {"first_0": 0, "second_0": 0}, 
                 architectureTypes = {"first_0": "xtra-small", "second_0": "small"}, 
                 randomization = {"first_0": "noisy", "second_0": "eps"}):

    # environment 
    # env = boxing_v2.parallel_env()
    env = boxing_v2.parallel_env(auto_rom_install_path="../ROMS", render_mode="human")

    # preprocess the environment
    env = EnvPreprocess.preprocess_boxing(env)

    # set seed 
    np.random.seed(SEED)
    random.seed(SEED)
    seed_torch(SEED)
    
    ############################################### parameters ###############################################
    
    num_frames = 300_000
    memory_size = 6_000
    batch_size = 64
    target_update = 100
    gamma = 0.99

    # define a suppport - might have to increase number of atoms
    v_min = -50
    v_max = 50
    atom_size = 51

    # define the architecture types


    # define path to save models
    PATH = "../results/models/1_VS_1/" + architectureTypes["first_0"] + "_"\
                                    + architectureTypes["second_0"] + "_BF1-" +\
                                    str(init_buffer_fill["first_0"]) + "_BF2-" + str(init_buffer_fill["second_0"])
    FIG_PATH = f"../results/figures/1v1_" + architectureTypes["first_0"] + "_"\
                                            + architectureTypes["second_0"] + "_BF1-" +\
                            str(init_buffer_fill["first_0"]) + "BF2-" + str(init_buffer_fill["second_0"]) + ".pdf"
    

    ############################################### training ###############################################
    
    agents = Atari_Agents(env, memory_size, batch_size, target_update, SEED, v_min=v_min, v_max=v_max,
                          atom_size=atom_size, archTypes=architectureTypes, gamma=gamma, 
                          PATH=PATH, fig_path=FIG_PATH, randomization=randomization, num_frames_train=num_frames)
    
    # uncomment next line to load pretrained models
    # agents.load_params("../results/models/1_VS_RANDOM/" + architectureType + "_finetuned2"+ ".pt")

    # train the agent
    agents.train(num_frames, init_buffer_fill_val=init_buffer_fill)


    ############################################### testing ###############################################

    # test the agent
    # create a new env with rendering                                    
    env.close()
    env = boxing_v2.parallel_env(auto_rom_install_path="../ROMS", render_mode="human")
    env = EnvPreprocess.preprocess_boxing(env, training=False)
    
    agents.test(env)

    env.close()


def test_boxing(PATH, architectureTypes, randomization): # test boxing using saved models

    env = boxing_v2.parallel_env(render_mode="human")
    env = EnvPreprocess.preprocess_boxing(env, training=False)

    # define a suppport - might have to increase number of atoms
    # IMPORTANT: make sure to use the same v_min and v_max as the one used in training
    v_min = -50 # <-50, 50>; 51 atoms with vs NOOP <-30, 30> with 51 for random
    v_max = 50
    atom_size = 51

        # set seed 
    np.random.seed(SEED)
    random.seed(SEED)
    seed_torch(SEED)
    
    agents = Atari_Agents(env, 300, 32, 20, SEED, v_min=v_min, v_max=v_max,
                          atom_size=atom_size, archTypes=architectureTypes, randomization=randomization)
    
    agents.load(PATH) # load the models
    agents.test(env)

    # close the env
    env.close()

if __name__ == "__main__":

    # Comparison 1: replay buffer prefilling
    # train_boxing(init_buffer_fill = {"first_0": 5000, "second_0": 0},  # initial buffer fill for each agent
    #              architectureTypes = {"first_0": "xtra-small", "second_0": "xtra-small"}, # different architectures for different agents
    #              randomization = {"first_0": "noisy", "second_0": "noisy"}) # select the type of randomization for each agent
    
    # Comparison 2: Feature Extraction Enhancement
    # train_boxing(init_buffer_fill = {"first_0": 1600, "second_0": 1600},  # initial buffer fill for each agent
    #              architectureTypes = {"first_0": "xtra-small", "second_0": "small"}, # different architectures for different agents
    #              randomization = {"first_0": "noisy", "second_0": "noisy"}) # select the type of randomization for each agent
    
    
    # Comparison 3: Stochastic Elements
    # train_boxing(init_buffer_fill = {"first_0": 1600, "second_0": 1600},  # initial buffer fill for each agent
    #              architectureTypes = {"first_0": "small", "second_0": "small"}, # different architectures for different agents
    #              randomization = {"first_0": "noisy", "second_0": "eps"}) # select the type of randomization for each agent


    # NOTE: the models are not saved the git since they are too big
    #      the models can be given upon request

    # test the 3. comparison (Stochastic Elements)
    # test_boxing("../results/models/1_VS_1/2-xtra-small_1600-init_noisy-eps.pt", 
    #             architectureTypes = {"first_0": "xtra-small", "second_0": "xtra-small"},
    #             randomization = {"first_0": "noisy", "second_0": "eps"})

    # test the 2. comparison (Feature Extraction Enhancement)
    # test_boxing("../results/models/1_VS_1/xtra-small-small_1600-init_2x-noisy.pt", 
    #             architectureTypes = {"first_0": "xtra-small", "second_0": "small"},
    #             randomization = {"first_0": "noisy", "second_0": "noisy"})

    # test the 1. comparison (replay buffer prefilling)
    test_boxing("../results/models/1_VS_1/2-xtra-small_5000A1-0A2_2x-noisy.pt", 
                 architectureTypes = {"first_0": "xtra-small", "second_0": "xtra-small"},
                 randomization = {"first_0": "noisy", "second_0": "noisy"})


