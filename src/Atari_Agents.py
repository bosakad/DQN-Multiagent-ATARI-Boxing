import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import utils
import copy

from ReplayBuffer import ReplayBuffer
from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from DQN_rainbox import Network

sns.set()


class Atari_Agents:
    """DQN Agent interacting with environment.
    
        defines the train and test of agents in the boxing scenario
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
        # add number of agents 
        n_agents = 2,
        TAU = 0.05, # convex combination of copying
        archTypes = {"first_0": "xtra-small", "second_0": "small"}, # small or big type of architecture
        PATH="../results/models/dqn", # path with filename, will save as path1.pt and path2.pt
        fig_path = "../results/figures",
        n_saves = 5
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
            archTypes ({str}): architecture type of the network
            PATH (str): path to save the models
            fig_path (str): path to save the figures
            n_saves (int): number of saves
        """
        obsShapeOrig = env.observation_space("first_0").shape
        obs_dim = (obsShapeOrig[2], obsShapeOrig[0], obsShapeOrig[1]) # pytorch expects (C,H,W)
        action_dim = env.action_space("first_0").n
        
        self.tau = TAU
        self.PATH = PATH
        self.fig_path = fig_path
        self.n_saves = n_saves
        self.saved_models = {}

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        self.agents = n_agents 
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # define agent names
        self.A1 = "first_0"
        self.A2 = "second_0"

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps

        # each agent has its own memory
        self.memory = [PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma), 
                       PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma)]
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = [ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma),
                             ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)]

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = [torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device), 
                        torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)]

        # networks: dqn, dqn_target for each agent
        self.dqn = [Network(obs_dim, action_dim, self.atom_size, self.support[0], architectureType=archTypes[self.A1]).to(self.device),
                    Network(obs_dim, action_dim, self.atom_size, self.support[1], architectureType=archTypes[self.A2]).to(self.device)]
        self.dqn_target = [Network(obs_dim, action_dim, self.atom_size, self.support[0], architectureType=archTypes[self.A1]).to(self.device),
                           Network(obs_dim, action_dim, self.atom_size, self.support[1], architectureType=archTypes[self.A2]).to(self.device)]
        
        for i, net in enumerate(self.dqn_target):
            net.load_state_dict(self.dqn[i].state_dict())
            net.eval()
        
        # optimizer
        self.optimizer = [optim.Adam(self.dqn[0].parameters(), lr=0.0005),
                          optim.Adam(self.dqn[1].parameters(), lr=0.0005)]

        # transition to store in memory
        self.transition = [list(), list()]
        
        # mode: train / test
        self.is_test = False

        # information needed for the init buffer fill
        self.frames_cur_episode = 0
        self.episode_num = 0
    
    def select_action(self, state: np.ndarray, random=False) -> np.ndarray:
        """Select an action from the input state."""
        
        # NoisyNet: no epsilon greedy action selection

        # alloc array of selected actions
        selected_action = [np.array(0)]*self.agents

        # add batch dimension - to be accepted by DQN
        state = state.unsqueeze(0) 

        # select action based on the Q - TODO: eps greedy
        if not random:
            for i in range(self.agents):  
                selected_action[i] = self.dqn[i](state).argmax()
                selected_action[i] = selected_action[i].detach().cpu().numpy()
        
        
        if not self.is_test:
            
            # select random action for both agents
            if random: 
                for i, agent in enumerate(self.env.agents):
                    selected_action[i] = np.array(self.env.action_space(agent).sample())    

            # save the transition
            for i in range(self.agents):
                self.transition[i] = [state.detach().cpu().numpy()[0], selected_action[i]]

        return {agent: selected_action[i] for i,agent in enumerate(self.env.agents)}
        

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        
        observations, rewards, terminations, truncations, _ = self.env.step(actions)
        next_state = utils.getState(observations, self.device) # get state from the observations
        
        # done = terminated or truncated
        done = terminations["first_0"] or truncations["first_0"]

        if not self.is_test:
            
            for i,agent in enumerate(self.env.agents):
                self.transition[i] += [rewards[agent], next_state.detach().cpu().numpy(), done]

                # N-step transition
                if self.use_n_step:
                    one_step_transition = self.memory_n[i].store(*self.transition[i])

                # one step transition
                else:
                    one_step_transition = self.transition[i]
                if one_step_transition:
                    self.memory[i].store(*one_step_transition)
                
        return next_state, rewards, done

    def update_model(self, agent) -> torch.Tensor:
        """Update the model by gradient descent."""
        
        # PER needs beta to calculate weights
        samples = self.memory[agent].sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma, agent)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n[agent].sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma, agent)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer[agent].zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn[agent].parameters(), 10.0)
        self.optimizer[agent].step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory[agent].update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn[agent].reset_noise()
        self.dqn_target[agent].reset_noise()

        return loss.item()
        
    def train(self, num_frames: int, plotting_interval: int = 200, init_buffer_fill={}):
        """Train the agent."""

        t_saves = np.linspace(0, num_frames, self.n_saves - 1, endpoint=False)

        self.is_test = False

        # fill the replay buffer with some experiences - TODO: change to two
        if init_buffer_fill[self.A1] > 0:
            self.fill_replay_buffer(init_buffer_fill[self.A1])

        # reset the env and get the initial state
        observations, _ = self.env.reset(seed=self.seed)
        state = utils.getState(observations, self.device) # get state from the observations
        

        # alloc the variables
        update_cnt = [0]*self.agents
        losses = [[] for _ in range(self.agents)]
        scores = [[] for _ in range(self.agents)]
        score = [0]*self.agents

        for frame_idx in range(1, num_frames + 1):
            
            actions = self.select_action(state)
            
            next_state, rewards, done = self.step(actions)

            state = next_state
            for i,agent in enumerate(self.env.agents):
                score[i] += rewards[agent]

            # NoisyNet: removed decrease of epsilon
            
            # PER: increase beta - importance sampling parameter for off-policy
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends - record the scores and restart the env
            if done:
                observations, _ = self.env.reset(seed=self.seed)
                state = utils.getState(observations, self.device) # get state from the observations

                self.frames_cur_episode = 0
                self.episode_num += 1

                for i in range(self.agents):
                    scores[i].append(score[i])
                    score[i] = 0

            # if training is ready - update the models
            for i in range(self.agents):
                
                # if i == 1: # dont train the second agent, TODO: remove this for both agents to learn    
                #     continue

                if len(self.memory[i]) >= self.batch_size: # enough experience
                    loss = self.update_model(agent=i)
                    losses[i].append(loss)
                    update_cnt[i] += 1
                    
                    # update each iteration - TODO: experiment
                    # target_net_state_dict = self.dqn[i].state_dict()
                    # policy_net_state_dict = self.dqn_target[i].state_dict()
                    # for key in policy_net_state_dict:
                    #     target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                    # self.dqn_target[i].load_state_dict(target_net_state_dict)

                    # if hard update is needed - update the target network
                    if update_cnt[i] % self.target_update == 0:
                        self._target_hard_update(i) # TODO: experiment with soft update

                        # print out the frame progress from time to time
                        if i == 0:
                            print("frame: ", frame_idx)
 
            self.frames_cur_episode += 1

            # save the model certain times
            if frame_idx - 1 in t_saves:
                model_name = f'{100 * (frame_idx - 1) / num_frames:04.1f}'.replace('.', '_')
                self.saved_models["dqn1_" + model_name] = copy.deepcopy(self.dqn[0].state_dict())
                self.saved_models["dqn2_" + model_name] = copy.deepcopy(self.dqn[1].state_dict())

        # plotting the result
        self._plot(frame_idx, np.array(scores), np.array(losses))
                
        self.env.close()

        # save the models
        self.saved_models["dqn1_" + "100_0"] = copy.deepcopy(self.dqn[0].state_dict())
        self.saved_models["dqn2_" + "100_0"] = copy.deepcopy(self.dqn[0].state_dict())

        torch.save(self.saved_models, self.PATH + ".pt")

    def load_params(self, PATH):
        """Load pretrained parameters."""
        self.dqn[0].load_state_dict(torch.load(PATH)["dqn1_100_0"]) # last params saved
        self.dqn[1].load_state_dict(torch.load(PATH)["dqn2_100_0"])

    def load(self, PATH):
        """Load the models."""
        self.saved_models = torch.load(PATH)

    def test(self, env: gym.Env) -> None:
        """Test the agent."""
        
        self.is_test = True
        

        for key, params in self.saved_models.items():
            if (key.startswith("dqn2")):
                continue

            print("Testing: " + key)
            
            self.dqn[0].load_state_dict(params)
            self.dqn[1].load_state_dict(self.saved_models[key.replace("dqn1", "dqn2")])
            
            self.dqn[0].eval() # set the evaluation mode of the model - removes the noisy layer
            self.dqn[1].eval()

            # create a recorder
            self.env = env

            # reset the env and get the initial state        
            state, _ = self.env.reset()
            state = utils.getState(state, self.device) # get state from the observations

            done = False
            score = [0, 0]
            
            while not done:
                actions = self.select_action(state)
                next_state, reward, done = self.step(actions)

                state = next_state
                score[0] += reward[self.A1]
                score[1] += reward[self.A2]
        
            print("score: ", score)
        
        self.env.close()
        
        # reset
        self.env = env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float, agent: int) -> torch.Tensor:
        """Return categorical dqn loss."""

        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)    
        
        with torch.no_grad():
            # Double DQN
            next_action = self.dqn[agent](next_state).argmax(1)
            next_dist = self.dqn_target[agent].dist(next_state)
            indices = torch.arange(self.batch_size).to(self.device)
            next_dist = next_dist[indices, next_action].to(self.device)

            t_z = reward + (1 - done) * gamma * self.support[agent]
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b.float())).view(-1)
            )
            
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b.float() - l.float())).view(-1)
            )

        dist = self.dqn[agent].dist(state)

        indices = torch.arange(self.batch_size).to(self.device)
        log_p = torch.log(dist[indices, action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss


    def fill_replay_buffer(self, num_frames: int):
        """Fill replay buffer with experiences."""

        # move the players in the corners F - these are places where they easily meet and get a lot rewards
        def moveToCorners_random(env, device):
            action = np.random.choice([6, 7, 8, 9]) # random corner
            actions = {"first_0": action, "second_0": action}
            for _ in range(10):
                observations, _, _, _, _ = env.step(actions)
            return utils.getState(observations, device) # return the last state
        
        def switchSides(env, side):
            pass # TODO

        # reset the env and get the initial state
        observations, _ = self.env.reset(seed=self.seed)
        state = utils.getState(observations, self.device)

        # fill the buffer with random actions
        for frame in range(0, num_frames + 1):

            # select a random actions and step the env (saves the transition in the memory)
            actions = self.select_action(state, random=True)
            next_state, rewards, done = self.step(actions)
            state = next_state

            # if episode ends - restart the env
            if done:
                observations, _ = self.env.reset(seed=self.seed)
                state = utils.getState(observations, self.device) # get state from the observations
            
            # move the players in the corners from time to time
            if frame % 50 == 0:
                state = moveToCorners_random(self.env, self.device)
                

        # print(self.memory[0].size)
        # exit()

    def _target_hard_update(self, agent): # TODO: try concex combination of target and local instead?
        """Hard update: target <- local."""
        self.dqn_target[agent].load_state_dict(self.dqn[agent].state_dict()) 
           
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        
        smooth_y = lambda K, y : np.convolve(y, np.ones(K)/K, mode='same')
        
        fig, axs = plt.subplots(1,2,figsize=(16,5))
        # score plot
        axs[0].set_title('Frame %s. Score A1: %s. Score A2: %s' % (frame_idx, np.mean(scores[0][-10:]), np.mean(scores[1][-10:])))
        axs[0].plot(scores[0], "b", label="A1")
        axs[0].plot(scores[1], "r", label="A2")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Score")
        # loss plot 
        axs[1].set_title("Training Loss (w. Running Average)")
        axs[1].plot(losses[0], "b", alpha=0.25)
        axs[1].plot(smooth_y(20,losses[0]), "b", label="A1")
        axs[1].plot(losses[1], "r", alpha=0.25)
        axs[1].plot(smooth_y(20,losses[1]), "r", label="A2")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("Loss")
        # set common things 
        for ax in axs:
            ax.legend()
        
        plt.savefig(self.fig_path)