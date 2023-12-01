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
        support: torch.Tensor,
        architectureType = "small",
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set feature layer
        historyLen = in_dim[0]

        if architectureType == "xtra-small":

            self.feature_layer = nn.Sequential(nn.Conv2d(historyLen, 16, 5, stride=5, padding=0), nn.ReLU(), nn.BatchNorm2d(16),
                                nn.Conv2d(16, 32, 5, stride=5, padding=0), nn.ReLU(), nn.BatchNorm2d(32))
        
            self.convOutputSize = 640 # change this if you change the convs above

        elif architectureType == "small":

            self.feature_layer = nn.Sequential(nn.Conv2d(historyLen, 32, 5, stride=5, padding=0), nn.ReLU(), nn.BatchNorm2d(32),
                                nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU(), nn.BatchNorm2d(64))
        
            self.convOutputSize = 1280 # change this if you change the convs above

        elif architectureType == "big": # architecture is large

             # set feature layer - TODO: experiment with adding the last layer? If it learns better
            self.feature_layer = nn.Sequential(nn.Conv2d(historyLen, 32, 8, stride=4, padding=0), nn.ReLU(), nn.BatchNorm2d(32),
                                nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(), nn.BatchNorm2d(64),
                                nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(), nn.BatchNorm2d(64),
                            #    nn.Conv2d(64, 128, 1, stride=1, padding=0), nn.ReLU(), nn.BatchNorm2d(128),
                                )
            
            self.convOutputSize = 8960 # change this if you change the convs above
        
        elif architectureType == "alexnet": # the AlexNet (ish) architecture 
            self.feature_layer = nn.Sequential(nn.Conv2d(historyLen, 32, kernel_size=9, stride=2, padding=0), nn.ReLU(),
                                               nn.MaxPool2d(kernel_size=3, stride=2), 
                                               nn.BatchNorm2d(32),
                                               nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                                               nn.MaxPool2d(kernel_size=3, stride=2),
                                               nn.BatchNorm2d(64),
                                               nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                               nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                               nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                                               nn.MaxPool2d(kernel_size=3, stride=2),
                                               )
            self.convOutputSize = 8960 # remember to change if changes to the CNN is made 

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(self.convOutputSize, self.convOutputSize) 
        self.advantage_layer = NoisyLinear(self.convOutputSize, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(self.convOutputSize, self.convOutputSize)
        self.value_layer = NoisyLinear(self.convOutputSize, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""

        feature = self.feature_layer(x)     
        
        # flatten the feature layer
        feature = feature.view(-1, self.convOutputSize)
        
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