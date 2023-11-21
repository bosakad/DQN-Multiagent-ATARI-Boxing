import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ReplayBuffer import *


def optimize_model(memory, BATCH_SIZE, GAMMA, policy_net, target_net, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return
    
    # get a batch of transitions
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states - leave zero for final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # compute the target
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # TODO: change this into ddqn
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values 
    
    # Compute the expected Q values
    target = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, target.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # TODO: might have to delete this 
    optimizer.step()
