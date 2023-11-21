import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.optim as optim

from pettingzoo.atari import boxing_v2

# custom modules
from DQN import DQN
from ReplayBuffer import ReplayBuffer   
from Optimize import optimize_model
from Plotter import plot_durations
from EpsilonScheduler import EpsilonScheduler
from EnvPreprocess import preprocess_boxing

def train_cartPole():

    # set up the environment
    env = gym.make("CartPole-v1")

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", torch.cuda.get_device_name(device))


    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayBuffer(10000)

    # initilize the epsilon scheduler
    epsScheduler = EpsilonScheduler(EPS_START, EPS_END, EPS_DECAY)

    steps_done = 0
    episode_durations = []


    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for _ in range(num_episodes):
        
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # for each episode, run the simulation until it's done
        for t in count():
            action = select_action(state, device, env, policy_net, epsScheduler.value(steps_done))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            steps_done += 1

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network) once in a while
            
            optimize_model(memory, BATCH_SIZE, GAMMA, policy_net, target_net, optimizer, device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)


            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations=episode_durations)
                break

    print('Complete')
    plot_durations(show_result=True, episode_durations=episode_durations)
    plt.ioff()
    plt.show()


def train_Boxing_MRL():
    """
    Train the 2 boxers on the Boxing-MRL environment
    """

    # set up the environment
    env = boxing_v2.parallel_env() 

    # preprocess the environment
    env = preprocess_boxing(env)

    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", torch.cuda.get_device_name(device))

    # Hyperparameters
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    # define agents
    A1 = "first_0"
    A2 = "second_0"

    # Get number of actions from gym action space
    n_actions = env.action_space(A1).n

    # Get the number of state observations
    state, _ = env.reset()
    initState = state[A1]
    
    # specify image dimensions (C, H, W)
    imageDims = (initState.shape[2], initState.shape[0], initState.shape[1])

    policy_net = DQN(imageDims[0] * imageDims[1] * imageDims[2], n_actions).to(device)
    target_net = DQN(imageDims[0] * imageDims[1] * imageDims[2], n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayBuffer(10000)

    # initilize the epsilon scheduler
    epsScheduler = EpsilonScheduler(EPS_START, EPS_END, EPS_DECAY)

    steps_done = 0
    episode_durations = []

    # specify the number of episodes
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50


    # train each episode
    for _ in range(num_episodes):
        
        # Initialize the environment and get it's state
        state, _ = env_reset(env)
        
        state = torch.tensor(state[A1], dtype=torch.float32, device=device)
        state = state.permute(2, 0, 1).unsqueeze(0)

        plt.imshow(state[0, 1, :, :].cpu().numpy(), cmap="gray")
        # plt.plot()
        plt.show()

        print(state.shape)
        exit()

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # for each episode, run the simulation until it's done
        for t in count():
            action = select_action(state, device, env, policy_net, epsScheduler.value(steps_done))
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            steps_done += 1

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network) once in a while
            
            optimize_model(memory, BATCH_SIZE, GAMMA, policy_net, target_net, optimizer, device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)


            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations=episode_durations)
                break

    env.close()

    print('Complete')
    plot_durations(show_result=True, episode_durations=episode_durations)
    plt.ioff()
    plt.show()


def env_reset(env):
    """
    Reset the boxing environment
    """
    
    _, _ = env.reset()
    _, _, _, _, _ = env.step({"first_0": 0, "second_0": 0}) # need to do no-action to initialize the environment
    state, _, _, _, _ = env.step({"first_0": 0, "second_0": 0}) # need to do no-action to initialize the environment

    return state, None



def select_action(state, device, env, policy_net, eps_threshold=None):
    """
    Epsilon greedy policy
    """

    sample = random.random()
    
    # greedy action
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    
    # random action
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


if __name__ == "__main__":
    # train_cartPole()
    train_Boxing_MRL()