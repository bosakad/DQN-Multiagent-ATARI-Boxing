import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NoisyLinear import NoisyLinear

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: (int, int, int), 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor,
        architectureType = "small",
        randomization = "noisy"
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
        
            self.convOutputSize = 288 # change this if you change the convs above

        elif architectureType == "small":

            self.feature_layer = nn.Sequential(nn.Conv2d(historyLen, 32, 5, stride=5, padding=0), nn.ReLU(), # nn.BatchNorm2d(32),
                                nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())                         #, nn.BatchNorm2d(64))
        
            self.convOutputSize = 576 # change this if you change the convs above

        elif architectureType == "big": # architecture is large
            
            self.feature_layer = nn.Sequential(nn.Conv2d(historyLen, 32, 9, stride=4, padding=0), nn.ReLU(),
                                               nn.Conv2d(32, 64, 5, stride=3, padding=0), nn.ReLU(),
                                               nn.Conv2d(64, 128, 3, stride=1, padding=0), nn.ReLU()
                                    )
            
            self.convOutputSize = 1920  
        
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
            self.convOutputSize = 3072 # remember to change if changes to the CNN is made 

        if randomization == "noisy":

            # set advantage layer
            self.advantage_hidden_layer = NoisyLinear(self.convOutputSize, self.convOutputSize) 
            self.advantage_layer = NoisyLinear(self.convOutputSize, out_dim * atom_size)

            # set value layer
            self.value_hidden_layer = NoisyLinear(self.convOutputSize, self.convOutputSize)
            self.value_layer = NoisyLinear(self.convOutputSize, atom_size)

        elif randomization == "eps": # based on epsilon scheduler and duelling networks

            # set advantage layer
            self.advantage_hidden_layer = nn.Linear(self.convOutputSize, self.convOutputSize) 
            self.advantage_layer = nn.Linear(self.convOutputSize, out_dim * atom_size)

            # set value layer
            self.value_hidden_layer = nn.Linear(self.convOutputSize, self.convOutputSize)
            self.value_layer = nn.Linear(self.convOutputSize, atom_size)

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