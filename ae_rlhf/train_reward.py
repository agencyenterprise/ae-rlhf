import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def train_reward_model(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 200,
    verbose: bool = True,
):
    """Train a reward model on a list of pairs

    The batches loaded by the data loaders should be a 3-tuple of the form
    (obs_left, obs_right, label) where obs_left and obs_right are the observations
    and label is the label for each pair of shape (batch, 2) where [1,0] means
    we prefer the left obs and [0,1] means we prefer the right obs. The labels can
    be fractional, e.g. [0.5, 0.5] means we have no preference between the two.




    Args:
        model: The reward model to train
        optimizer: The optimizer to use for training
        train_loader: A dataloader for the training data.
        val_loader: A dataloader for the validation data.
        epochs: The number of epochs to train for
        verbose: Whether to print progress to stdout
    """
    for epoch in range(epochs):
        train_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=epoch,
            verbose=verbose,
        )


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epoch: int,
    verbose=True,
):
    """Train the model for a single epoch.

    This implements a minimal training loop for training the reward model.

    Args:
        model: The reward model to train
        optimizer: The optimizer to use for training
        train_loader: A dataloader for the training data.
        val_loader: A dataloader for the validation data.
        epoch: The current epoch number
        verbose: Whether to print progress to stdout
    """
    model.train()
    device = next(model.parameters()).device
    total_batches = 0
    total_loss = 0
    for obs_left, obs_right, labels in train_loader:
        obs_left = obs_left.to(device)
        obs_right = obs_right.to(device)
        labels = labels.to(device)
        loss = train_step(model, optimizer, obs_left, obs_right, labels)
        total_loss += loss
        total_batches += 1
        avg_loss = total_loss / total_batches
        if verbose:
            print(f"\repoch {epoch} train loss {avg_loss:.3f}", end="")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for obs_left, obs_right, labels in val_loader:
            obs_left = obs_left.to(device)
            obs_right = obs_right.to(device)
            labels = labels.to(device)
            loss = batch_loss(
                model=model, obs_left=obs_left, obs_right=obs_right, labels=labels
            )
            val_loss += loss.item()
        val_loss /= len(val_loader)

        if verbose:
            print(f"\repoch {epoch} train loss {avg_loss:.3f} val loss {val_loss:.3f}")


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_left: torch.Tensor,
    obs_right: torch.Tensor,
    labels: torch.Tensor,
):
    """Train the model on a single batch of data"""
    loss = batch_loss(
        model=model, obs_left=obs_left, obs_right=obs_right, labels=labels
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def batch_loss(
    *,
    model: nn.Module,
    obs_left: torch.Tensor,
    obs_right: torch.Tensor,
    labels: torch.Tensor,
):
    """Computes the loss for a single batch.

    The batches of data are rgb observations of shape
       (batch, sequence, features).
    where the features are the cartpole state.

    The observations are sampled as short segments of a longer trajectory and the number
    of frames samples is the `sequence` dimension.

    Additionally, to compute the loss for a given pair of observations, we sum the
    predictions over the sequence dimension as is done in the paper.

    Args:
        obs_left: The observations that were on the left side of the screen during
            data collection.  Shape (batch, sequence, channels, height, width)
        obs_right: The observations that were on the right side of the screen during
            data collection.  Shape (batch, sequence, channels, height, width)
        label: The label for each pair of shape (batch, 2) where [1,0] means
            the we prefer the left obs and [0,1] means we prefer the right obs. The
            labels can also be fractional, e.g. [0.5, 0.5] means we have no preference
            between the two or even smoothed e.g. [0.9, 0.1].
    """
    if obs_left.shape != obs_right.shape:
        raise ValueError("obs_left and obs_right must have the same shape")

    batch, sequence, features = obs_left.shape[0], obs_left.shape[1], obs_left.shape[2:]

    # fold the sequence dimension into the batch dimension
    # since the model is expected (batch, features) input
    obs_left = obs_left.reshape(-1, *features)
    obs_right = obs_right.reshape(-1, *features)

    # push the data through the model
    pred_left = model(obs_left)
    pred_right = model(obs_right)

    # unfold the sequence dimension and sum the prediction over the sequence
    pred_left = pred_left.reshape(batch, sequence, 1).sum(dim=1)
    pred_right = pred_right.reshape(batch, sequence, 1).sum(dim=1)
    pred = torch.cat([pred_left, pred_right], dim=1)  # now shape (batch, 2))

    # compute the loss
    loss = F.binary_cross_entropy_with_logits(pred, labels)
    return loss
