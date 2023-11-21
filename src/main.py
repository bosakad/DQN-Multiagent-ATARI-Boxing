import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.optim as optim

# custom modules
from DQN import DQN
from ReplayBuffer import ReplayBuffer   
from Optimize import optimize_model
from Plotter import plot_durations
from EpsilonScheduler import EpsilonScheduler

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
        num_episodes = 200
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        
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
            if t + 1 % 4 == 0:
                optimize_model(memory, BATCH_SIZE, GAMMA, policy_net, target_net, optimizer, device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            if t + 1 % 100 == 0:
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
    train_cartPole()