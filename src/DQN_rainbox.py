import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NoisyLinear import NoisyLinear

# TODO: redo the network such that the agents share the convolutional layers, but have seperate FF layers:) 

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: (int, int, int), 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set feature layer - TODO: experiment with adding the last layer? If it learns better
        self.feature_layer = nn.Sequential(nn.Conv2d(in_dim, 32, 8, stride=4, padding=0), nn.ReLU(), nn.BatchNorm2d(32),
                                           nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(), nn.BatchNorm2d(64),
                                           nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(), nn.BatchNorm2d(64),
                                        #    nn.Conv2d(64, 128, 1, stride=1, padding=0), nn.ReLU(), nn.BatchNorm2d(128),
                                           )
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(64, 64) # todo: change number of inputs to fit the image after convs
        self.advantage_layer = NoisyLinear(64, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(64, 64)
        self.value_layer = NoisyLinear(64, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""

        feature = self.feature_layer(x) # TODO: put convs instead of FC - no need, feature_layer is the convolutions:-)
        
        
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()