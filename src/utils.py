"""
    This file contains utility functions for the project. Such as image normalization
"""

import torch

def getObservation(states: dict, device) -> torch.Tensor:
    """
    Get the observation from the states dictionary and normalize it
    """

    # permute the dimension
    state = torch.tensor(states["first_0"], dtype=torch.float32, device=device)
    state = state.permute(2, 0, 1) 

    # normalize the image
    state = Normalize(state)

    return state

def Normalize(image: torch.Tensor) -> torch.Tensor :
    """
    Normalize the image by channels
    """
    
    means = image.mean(dim=(1, 2), keepdim=True)
    stds = image.std(dim=(1, 2), keepdim=True)

    eps = 1e-6
    image = (image - means) / (stds + eps)

    return image

