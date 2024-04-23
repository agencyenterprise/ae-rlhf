import copy
import logging
from pathlib import Path
from typing import Protocol

import cv2
import gymnasium as gym
import numpy as np
import requests
import torch
from sqlmodel import Session, select
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    StickyActionEnv,
)
from torch.utils.data import Dataset

from ae_rlhf.app import crud
from ae_rlhf.app.models import Model, Pair, Run, Segment
from ae_rlhf.config import CONSTANTS, settings

logger = logging.getLogger(__name__)


##############################
# Sampling
##############################
class Actor(Protocol):
    def act(self, obs: np.ndarray) -> int:
        ...


def sample_segments(
    agent: Actor, env: gym.Env, n_pairs: int, segment_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample segments of trajectories.

    Observation Trajectory shaape: (n_pairs * 2, segment_length, *obs_shape)
    Image Trajectory shape: (n_pairs * 2, segment_length, *img_shape)

    The observations are used for training while the images are used to create
    videos to be displayed to the user for labeling.

    Args:
        agent (Actor): The agent to sample with.
        env (gym.Env): The environment to sample from.
        n_pairs (int): The number of pairs to sample.
        segment_length (int): The length of each segment.

    Returns:
        tuple[np.ndarray, np.ndarray]: The observation and image trajectories.

    """
    if hasattr(agent, "parameters"):
        device = next(agent.parameters()).device
    else:
        device = None

    n_samples = n_pairs * 2
    n_timesteps_needed = n_samples * segment_length
    timestep = 0

    # clone a single env since this is easier to work with
    if isinstance(env, gym.vector.SyncVectorEnv):
        env = copy.deepcopy(env.envs[0])
    else:
        env = copy.deepcopy(env)

    obs, _ = env.reset()
    img: np.ndarray = np.array(env.render())

    # buffers for observations and images
    obs_buffer_shape: tuple[int, ...] = (n_timesteps_needed, *obs.shape)
    img_buffer_shape: tuple[int, ...] = (n_timesteps_needed, *img.shape)

    obs_buffer = np.zeros(obs_buffer_shape, dtype=obs.dtype)
    img_buffer = np.zeros(img_buffer_shape, dtype=img.dtype)

    while timestep < n_timesteps_needed:
        # put the current frame in the buffer
        obs_buffer[timestep] = obs
        img_buffer[timestep] = img

        # act in the environment
        if device:
            if not isinstance(obs, torch.Tensor) and not isinstance(obs, np.ndarray):
                obs = torch.tensor(np.array(obs), device=device)
            elif not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, device=device)
            act = agent.act(obs)
        else:
            act = agent.act(obs)
        obs, rew, terminated, truncated, info = env.step(act.cpu().item())
        img = np.array(env.render())

        timestep += 1
        if terminated:
            env.reset()

    # reshape to (n_samples=2*n_pairs, segment_length, *shape)
    obs_buffer = obs_buffer.reshape((n_samples, segment_length, *obs.shape))
    img_buffer = img_buffer.reshape((n_samples, segment_length, *img.shape))

    # shuffle so obs are paired up randomly
    idx = np.random.permutation(n_samples)
    return obs_buffer[idx], img_buffer[idx]


##############################
# Saving and loading
##############################
def save_pairs(
    db: Session,
    *,
    obs: np.ndarray,
    images: np.ndarray,
    run_name: str,
    env_id: str,
    iteration: int,
    fps: int,
) -> None:
    """Save observation and image pairs to disk and to the database.

    Args:
        db (Session): The database session.
        obs (np.ndarray): The observations.
        images (np.ndarray): The images.
        run_name (str): The name of the run.
        iteration (int): The iteration of the run. By convention pretrain is
            iteration 0.
        fps (int): The frames per second of the video.

    """
    if len(obs) != len(images):
        raise ValueError("Observations and images must be the same length.")
    if len(obs) % 2 != 0:
        raise ValueError("Observations and images must be even length.")

    for i in range(0, len(obs), 2):
        try:
            with db.begin_nested():
                save_pair(
                    db,
                    obs=(obs[i], obs[i + 1]),
                    imgs=(images[i], images[i + 1]),
                    run_name=run_name,
                    env_id=env_id,
                    iteration=iteration,
                    fps=fps,
                )
        except Exception:
            logger.error("Error saving pair.", exc_info=True)


def save_pair(
    db: Session,
    *,
    obs: tuple[np.ndarray, np.ndarray],
    imgs: tuple[np.ndarray, np.ndarray],
    run_name: str,
    env_id: str,
    iteration: int,
    fps: int,
) -> None:
    """Save an observation and image pair to disk and to the database.


    This function first creates new segments and associates them with a new pair
    in the database.  Then it uses the id of the new segments to fill a filename
    template and then 1. updates the database with the new filename and 2. saves
    the observation and video to disk with the created filenames.

    Args:
        db (Session): The database session.
        obs (list[np.ndarray]): The observations.
        imgs (list[np.ndarray]): The images.
        run_name (str): The name of the run.
        iteration (int): The iteration of the run. By convention pretrain is
            iteration 0.
        fps (int): The frames per second of the video.

    """
    if len(obs) != 2:
        raise ValueError("Observations must be a list of length 2.")
    if len(imgs) != 2:
        raise ValueError("Images must be a list of length 2.")

    run = crud.read_run(db, name=run_name)
    if run is None:
        run = Run(name=run_name, env_id=env_id)
        run = crud.create_run(db, run)

    # first add and flush so segment ids are available
    segments = [Segment() for _ in range(2)]
    pair = Pair(segments=segments, run=run, iteration=iteration)
    db.add(pair)
    db.flush()

    # now add the filenames filled in with the segment ids
    for seg in segments:
        if seg.id is None:
            raise RuntimeError("Segment id is None.")
        seg.obs_uri = obs_uri(run_name=run_name, iteration=iteration, id=seg.id)
        seg.video_uri = video_uri(run_name=run_name, iteration=iteration, id=seg.id)

    db.add(pair)
    db.flush()

    # save the data with the filenames
    for i, seg in enumerate(segments):
        if seg.obs_uri is None or seg.video_uri is None:
            raise RuntimeError("Segment uri is not set.")

        obs_path = Path(seg.obs_uri)
        video_path = Path(seg.video_uri)

        if obs_path.exists():
            raise RuntimeError(f"Segment obs uri already exists: {seg.obs_uri}")
        if video_path.exists():
            raise RuntimeError(f"Segment video uri already exists: {seg.video_uri}")

        if not obs_path.parent.exists():
            obs_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.parent.exists():
            video_path.parent.mkdir(parents=True, exist_ok=True)

        save_obs(obs[i], filename=seg.obs_uri)
        save_video(
            imgs[i], filename=seg.video_uri, fps=fps, video_4cc=settings.VIDEO_4CC
        )


def save_obs(obs: np.ndarray, *, filename: str):
    """Save an observation to disk.

    Args:
        obs (np.ndarray): The observation to save.
        filename (str): The filename to save to.
    """
    np.save(filename, obs)


def load_obs(filename: str) -> np.ndarray:
    """Load an observation from disk.

    Args:
        filename (str): The filename to load from.

    Returns:
        np.ndarray: The loaded observation.
    """
    return np.load(filename)


def save_video(images: np.ndarray, *, filename: str, fps: int, video_4cc: str):
    """Save a video from a sequence of images.

    Args:
        images (np.ndarray): sequence of images.
        save_dir (str): directory to save the movie.
    """
    width, height, channels = images.shape[1:]

    fourcc = cv2.VideoWriter_fourcc(*video_4cc)  # type: ignore
    writer = cv2.VideoWriter(str(filename), fourcc, fps, (height, width))
    for img in images:
        writer.write(img)
    writer.release()


def save_model(
    db: Session,
    *,
    model: torch.nn.Module,
    run_name: str,
    iteration: int,
    model_name: str,
    optimizer: torch.optim.Optimizer | None = None,
    **kwargs,
):
    """Save a model to disk and to the database."""
    run = crud.read_run(db, name=run_name)
    if run is None:
        raise ValueError(f"Run {run_name} does not exist. Cannot save model.")

    if model_exists(db, run_name=run_name, iteration=iteration, model_name=model_name):
        raise ValueError(
            f"Model {model_name} with iteration {iteration}"
            f" already exists for run {run_name}."
        )

    db_model = Model(name=model_name, iteration=iteration)
    if run.models is None:
        run.models = [db_model]
    else:
        run.models.append(db_model)
    db.add(run)
    db.flush()

    if db_model.id is None:
        raise RuntimeError("Model id is None.")

    uri = model_uri(
        run_name=run_name, iteration=iteration, model_name=model_name, id=db_model.id
    )
    db_model.uri = str(uri)
    db.add(run)

    opt_state_dict = {} if optimizer is None else optimizer.state_dict()
    data = {
        "model": model.state_dict(),
        "optimizer": opt_state_dict,
        "iteration": iteration,
        **kwargs,
    }

    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, uri)


def load_most_recent_state_dict(db: Session, *, run_name: str, model_name: str):
    """Finds the most recent model based on iteration and loads the state dict."""
    stmt = (
        select(Model)
        .join(Run)
        .where(
            Run.name == run_name,
            Model.name == model_name,
        )
        .order_by(Model.iteration.desc())  # type: ignore
    )
    db_model = db.exec(stmt).first()
    if db_model is None:
        raise ValueError(f"Model {model_name} does not exist for run {run_name}.")
    if db_model.uri is None:
        raise ValueError(f"Model {model_name} does not have a uri for run {run_name}.")
    return torch.load(db_model.uri)


def load_state_dict(
    db: Session,
    *,
    run_name: str,
    iteration: str,
    model_name: str,
):
    """Load a state dict from disk and return it."""
    stmt = (
        select(Model)
        .join(Run)
        .where(
            Run.name == run_name, Model.name == model_name, Model.iteration == iteration
        )
    )
    db_model = db.exec(stmt).all()
    if len(db_model) == 0:
        raise ValueError(
            f"Model {model_name} with iteration {iteration}"
            f" does not exist for run {run_name}."
        )
    if len(db_model) > 1:
        raise ValueError(
            f"Multiple models {model_name} with iteration {iteration}"
            " exist for run {run_name}."
        )
    if db_model[0].uri is None:
        raise ValueError(
            f"Model {model_name} with iteration {iteration}"
            f" does not have a uri for run {run_name}."
        )
    return torch.load(db_model[0].uri)


def obs_uri(*, run_name: str, iteration: int, id: int) -> str:
    return settings.OBS_TEMPLATE.format(run_name=run_name, iteration=iteration, id=id)


def video_uri(*, run_name: str, iteration: int, id: int) -> str:
    return settings.VIDEO_TEMPLATE.format(run_name=run_name, iteration=iteration, id=id)


def model_uri(*, run_name: str, iteration: int, model_name: str, id: int) -> Path:
    return settings.MODEL_TEMPLATE.format(
        run_name=run_name, iteration=iteration, model_name=model_name, id=id
    )


##############################
# Helpers for checking state
##############################
def model_exists(
    db, *, run_name: str, model_name: str, iteration: int | None = None
) -> bool:
    """Checks if the model exists in the database."""
    stmt = (
        select(Model)
        .join(Run)
        .where(
            Run.name == run_name,
            Model.name == model_name,
        )
    )
    if iteration is not None:
        stmt = stmt.where(Model.iteration == iteration)

    return len(db.exec(stmt).all()) > 0


def all_labeled(db, **kwargs) -> bool:
    pairs = crud.read_pairs(db, **kwargs)
    if pairs is None:
        raise ValueError("Pairs not initialized")
    return all(pair.label is not None for pair in pairs)


def more_recent_pairs_exist(db, *, run_name: str, iteration: int) -> bool:
    """Checks if there are more recent pairs than the given iteration."""
    stmt = (
        select(Pair)
        .join(Run)
        .where(
            Run.name == run_name,
            Pair.iteration > iteration,  # type: ignore
        )
    )
    return len(db.exec(stmt).all()) > 0


##############################
# API Helpers
##############################
def check_labeling_ui_is_running():
    """Check that the labeling ui is running."""
    try:
        r = requests.get(settings.API_URL + "/api/health")
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "The labeling ui is not running.  Please start it with "
            "`python asgi.py`"
            f" and then visit {settings.API_URL} in your browser."
        )


##############################
# Env Helper
##############################
def wrap_atari_env(env: gym.Env):
    """Applies standard wrappers for Atari games."""
    env = NoopResetEnv(env, noop_max=30)
    env = StickyActionEnv(env, action_repeat_probability=0.25)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


def make_env(env_id: str, fps: int, atari=True):
    def thunk():
        if atari:
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                frameskip=1,
                repeat_action_probability=0.00,
            )
        else:
            env = gym.make(env_id, render_mode="rgb_array")

        env.metadata["render_fps"] = fps
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if atari:
            env = wrap_atari_env(env)
        return env

    return thunk


##############################
# Train Data Helpers
##############################
class PairsDataset(Dataset):
    """A dataset of pairs of observations"""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        obs_left = load_obs(self.pairs[idx].segments[0].obs_uri)
        obs_right = load_obs(self.pairs[idx].segments[1].obs_uri)
        label = label_to_tensor(self.pairs[idx].label)
        return obs_left, obs_right, label


def label_to_tensor(label: int) -> torch.Tensor:
    """Convert a label to a tensor.

    Args:
        label (int): The label.

    Returns:
        torch.Tensor: The label as a tensor.
    """
    match label:
        case CONSTANTS.left.id:
            return torch.tensor([1.0, 0.0])
        case CONSTANTS.right.id:
            return torch.tensor([0.0, 1.0])
        case CONSTANTS.tie.id:
            return torch.tensor([0.5, 0.5])
        case _:
            raise ValueError("Invalid label, please filter labels before calling.")


###################################
# Helpers for handing iterations
###################################
def should_train_this_iteration_maker(start_iteration: int):
    """Make a function that determines whether to train the model at an iteration.

    Assumes the start_iteration is the iteration at which we should start training.
    So if we load in a model from that was saved at iteration 5, then we should pass
    start_iteration=6 to this function since we already trained at iteration 5.

    Args:
        start_iteration (int): the iteration at which to start training
    """

    def should_train_this_iter(iteration):
        if iteration < start_iteration:
            return False
        return True

    return should_train_this_iter


def should_save_this_iteration_maker(*, sample_every: int, steps_per_iter: int):
    """Make a function that determines whether to save the model at an iteration.

    We pause training to save the model, sample more data, and then wait for
    the reward model to train after a fixed number of iterations.  This function
    helps us determine when we need to pause.

    This is based on the number of steps we take within a given iteration and the
    number of steps we want to take between pausing for saving/sampling.

    We pause on the iteration, i, such that

    (i-1)* steps_per_iter < sample_every * n < i * steps_per_iter

    for some integer n.  In other words, we pause when we cross over the sample_every
    threshold.


    Args:
        sample_every (int): the number of steps between samples
        steps_per_iter (int): the number of steps per iteration
    """

    def should_save_this_iteration(iteration):
        total_steps = (iteration + 1) * steps_per_iter
        remainder = total_steps % sample_every

        # if the remainder is less then steps_per_iter then we just
        # crossed over the sample_every threshold and should save
        # e.g. if sample_every=1000 and steps_per_iter=150 and we are at iteration
        # 7 then total_steps=7*150=1050 and remainder=1050%1000=50 so we should save
        # but at iteration 8 total_steps=8*150=1200 and remainder=1200%1000=200 so we
        # should not save
        if remainder < steps_per_iter:
            return True
        return False

    return should_save_this_iteration


def n_pairs_this_iter_maker(
    *, timesteps: int, total_pairs: int, sample_every: int, steps_per_iter: int
):
    """Make a function that calculates the number of pairs to sample at an iteration.

    This function is based on the schedule described in the paper.  We sample so that
    a sample at timestep T is proportional to 5e6/(T + 5e6).   That is at time T we
    collect N*(5e6/(T + 5e6)) pairs where N is a fixed multipler.  Given the total
    number of timsteps and the sampling, the sampling interval, and the total number
    of pairs we want to collect we can calculate N.  From there we can calculate
    the number of pairs to collect at each iteration.

    Args:
        timesteps (int): the total number of timesteps
        total_pairs (int): the total number of pairs to collect
        sample_every (int): the number of steps between samples


    """

    # the total number of sampling steps
    num_sample_steps = timesteps // sample_every

    # the proportion of data to collect at each step
    proportion = [5e6 / (sample_every * (i + 1) + 5e6) for i in range(num_sample_steps)]

    # base number of pairs to collect (total pairs will be this time proportion)
    # n_pair_multipler = N in the docstring
    n_pair_multipler = total_pairs // sum(proportion)

    def n_pairs_this_iter(iteration):
        idx = int((iteration * steps_per_iter) // sample_every)
        idx = min(idx, len(proportion) - 1)
        n_pairs = int(n_pair_multipler * proportion[idx])
        return max(n_pairs, 1)

    return n_pairs_this_iter
