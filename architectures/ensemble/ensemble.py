import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VotingSegmenter(nn.Module):
    """Majority-based voting ensamble.

    This model ensamble decides on the final outcome by averaging the 
    results of 5 different models.
    """

    def __init__(self, ms, cs):
        """Initializes the voting ensemble.

        Args:
            ms (list(nn.Module)): The list of segmentation models
            cs (list(int)): The list of input channels for each model
        """
        super().__init__()

        # Initialize models and channels
        self.ms = nn.Sequential(*ms)
        self.cs = cs
    
    
    def forward(self, x):
        """Performs forward pass of the model ensemble.

        Args:
            x (torch.tensor): The raw input of shape (N, C, H, W)
        
        Returns:
            output (ndarray): The averaged segmentation output
        """
        # Get the correct device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize empty activations list
        xs = []

        for m, c in zip(self.ms, self.cs):
            # Get the activations of an individual model `m`
            a = m(x.clone().repeat(1, c, 1, 1))
            a = F.softmax(a, dim=1).detach().cpu().numpy()
            xs.append(a)

        # Concat all outputs and average over
        xs = np.array(xs)
        output = torch.tensor(np.mean(xs, axis=0)).to(device)
        
        return output