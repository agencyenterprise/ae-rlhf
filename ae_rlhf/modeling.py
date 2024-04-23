import gymnasium as gym
import numpy as np
import torch
from torch import nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Use orthogonal initialization for the weights and 0 for the biases."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RandomAgent:
    """An agent that takes random actions."""

    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, observation):
        return torch.tensor(self.env.action_space.sample())


class RewardModel(nn.Module):
    """The reward model used to predict reward from observations.

    Note: we normalize the output of the reward model to have mean 0 and standard
    deviation 0.05 on the atari environment during inference.  This follows the
    same practice as in the paper.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        super().__init__()
        self.register_buffer("a", torch.tensor(a))
        self.register_buffer("b", torch.tensor(b))

    def init_reward_normalization(
        self,
        loader: torch.utils.data.DataLoader,
        target_mu: float = 0.0,
        target_sigma: float = 0.05,
    ):
        """Initialize the reward normalization parameters.

        In the paper they normalize the output of the reward model to have mean 0 and
        standard deviation 0.05 on the atari environment.  This function computes the
        normalization parameters by running the model over a dataset of pairs and
        computing the main and standar deviation of the output.

        This should be called *after* training the reward model.

        We assume that the data loader yeilds batches of pairs of observations of the
        form (obs_left, obs_right, label).  And that the observations are of shape
        (batch, sequence, channels, height, width).  Since the model is expecting
        a batch of 2D images we reshape the observations to be
        (batch * sequence, channels, height, width).  And since we only care about
        each frame in the sequence independently for this function we don't need to
        reshape it back before calculating the mean and standard deviation.

        Note: The target statistics are at the *frame* level.

        Args:
            loader: A data loader that yields batches of pairs of observations.
            target_mu: The target mean of the output of the reward model.
            target_sigma: The target standard deviation of the output of the reward
                model.
        """
        device = next(self.parameters()).device
        preds = []
        with torch.no_grad():
            for obs_left, obs_right, _ in loader:
                obs_left = obs_left.reshape(-1, *obs_left.shape[2:]).to(device)
                obs_right = obs_right.reshape(-1, *obs_right.shape[2:]).to(device)
                preds.append(self(obs_left))
                preds.append(self(obs_right))
            pred = torch.cat(preds)
            self.a = target_sigma / pred.std()
            self.b = target_mu - self.a * pred.mean()
