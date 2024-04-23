# type: ignore
"""
Adapted from CleanRl `ppo_atari.py`
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py

Updated to use the reward model as the reward signal instead of the environment reward.
"""
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ae_rlhf import utils
from ae_rlhf.app import crud
from ae_rlhf.app.database import Session
from ae_rlhf.config import VALID_LABELS, settings
from ae_rlhf.train_reward import train_reward_model
from ae_rlhf.utils import (
    n_pairs_this_iter_maker,
    should_save_this_iteration_maker,
    should_train_this_iteration_maker,
)

# local imports from same directory
from make_env import make_env  # isort:skip
from modeling import Agent, RewardModel  # isort:skip


@dataclass
class Args:
    name: str
    """the unique name of the run associated with the data and reward model"""
    env_id: str
    """the environment id"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rlhf"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    lr: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # added for rlhf
    segment_length: int = 25
    """the length of the segments to sample for pair wise comparisons"""
    reward_lr: float = 2.5e-4
    """the learning rate of the reward model"""
    sample_every: int = int(1e5)
    """the number of steps between samples"""
    n_pairs: int = 5000
    """The number of pairs to sample in total through training"""
    epochs: int = 200
    """The number of epochs to train the reward model for"""
    reward_batch_size: int = 32
    """The batch size to use for training the reward model"""
    weight_decay: float = 1e-5
    """The weight decay to use for training the reward model"""
    reuse_optimizer: bool = False
    """Whether to reuse the optimizer from the most recent training run"""
    limit: int = 3000
    """The maximum number of most recent pairs to use for reward model training"""
    fps: int = 15
    """The frame rate to render the environment at"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    STEPS_PER_ITER = args.batch_size

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"{settings.SAVE_DIR}/{args.name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id=args.env_id) for _ in range(args.num_envs)],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    ##############################################
    # NEW
    # setup agent and reward model, possibly restoring from previous iter
    ##############################################
    with Session() as db:
        if utils.model_exists(
            db, run_name=args.name, model_name=settings.PPO_MODEL_NAME
        ):
            agent_state_dict = utils.load_most_recent_state_dict(
                db, run_name=args.name, model_name=settings.PPO_MODEL_NAME
            )
            agent = Agent(envs).to(device)
            agent.load_state_dict(agent_state_dict["model"])
            optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
            optimizer.load_state_dict(agent_state_dict["optimizer"])
            start_iteration = agent_state_dict["iteration"]
            global_step = agent_state_dict["global_step"]
        else:
            agent = Agent(envs).to(device)
            optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
            start_iteration = 0
            global_step = 0

        if not utils.model_exists(
            db, run_name=args.name, model_name=settings.REWARD_MODEL_NAME
        ):
            raise ValueError(
                f"No reward model found for run_name={args.name}"
                f" Please run `python pretrain_reward_model.py --name={args.name}`"
                f" to initialize a reward model."
            )

        reward_model = RewardModel().to(device)
        reward_optimizer = optim.AdamW(
            reward_model.parameters(), lr=args.reward_lr, weight_decay=args.weight_decay
        )
        reward_state_dict = utils.load_most_recent_state_dict(
            db, run_name=args.name, model_name=settings.REWARD_MODEL_NAME
        )
        reward_model.load_state_dict(reward_state_dict["model"])
        if args.reuse_optimizer:
            reward_optimizer.load_state_dict(reward_state_dict["optimizer"])

    # setup function for calculating whether to train, save, or sample this iteration
    should_train_this_iteration = should_train_this_iteration_maker(start_iteration)
    should_save_this_iteration = should_save_this_iteration_maker(
        sample_every=args.sample_every,
        steps_per_iter=STEPS_PER_ITER,
    )
    n_pairs_this_iter = n_pairs_this_iter_maker(
        timesteps=args.total_timesteps,
        total_pairs=args.n_pairs,
        sample_every=args.sample_every,
        steps_per_iter=STEPS_PER_ITER,
    )

    ##############################################
    # END NEW
    ##############################################

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if should_train_this_iteration(iteration):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.lr
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )
                next_done = np.logical_or(terminations, truncations)

                next_obs, next_done = (
                    torch.Tensor(next_obs).to(device),
                    torch.Tensor(next_done).to(device),
                )

                ##############################################
                # NEW: use reward model to get the reward
                ##############################################
                with torch.no_grad():
                    rewards[step] = reward_model(next_obs).view(-1)
                ##############################################
                # END NEW
                ##############################################

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(
                                f"global_step={global_step},"
                                f" episodic_return={info['episode']['r']}"
                            )
                            writer.add_scalar(
                                "charts/episodic_return",
                                info["episode"]["r"],
                                global_step,
                            )
                            writer.add_scalar(
                                "charts/episodic_length",
                                info["episode"]["l"],
                                global_step,
                            )

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

        ##############################################
        # NEW
        ##############################################
        # save the agent
        if should_save_this_iteration(iteration):
            with Session() as db:
                if not utils.model_exists(
                    db,
                    run_name=args.name,
                    iteration=iteration,
                    model_name=settings.PPO_MODEL_NAME,
                ):
                    utils.save_model(
                        db,
                        model=agent,
                        optimizer=optimizer,
                        run_name=args.name,
                        iteration=iteration,
                        model_name=settings.PPO_MODEL_NAME,
                        global_step=global_step,
                    )
                else:
                    print(f"Model exists for iteration {iteration}. Skipping save.")

            # sample segment using the agent
            with Session() as db:
                n_pairs_needed = n_pairs_this_iter(iteration)
                n_pairs_collected = len(
                    crud.read_pairs(db, run_name=args.name, iteration=iteration)
                )
                n_pairs_to_sample = max(n_pairs_needed - n_pairs_collected, 0)
                if n_pairs_to_sample > 0:
                    obs_, imgs_ = utils.sample_segments(
                        agent=agent,
                        env=envs,
                        n_pairs=n_pairs_this_iter(iteration),
                        segment_length=args.segment_length,
                    )
                    utils.save_pairs(
                        db,
                        obs=obs_,
                        images=imgs_,
                        run_name=args.name,
                        iteration=iteration,
                        env_id=args.env_id,
                        fps=args.fps,
                    )
                else:
                    print(
                        f"Already collected {n_pairs_collected} pairs for"
                        f" iteration {iteration}. Skipping sampling."
                    )

            # wait for the newly collected pairs to be labeled
            while True:
                with Session() as db:
                    if utils.all_labeled(db, run_name=args.name):
                        break
                    else:
                        print("Waiting for all pairs to be labeled...")
                        print(
                            f"Visit {settings.API_URL}/feedback/{args.name}"
                            " to label them."
                        )
                        time.sleep(10)

            # Now that we know they are labeled make sure the labels are valid
            with Session() as db:
                raw_pairs = crud.read_pairs(db, run_name=args.name)
                pairs = [p for p in raw_pairs if p.label in VALID_LABELS]

                # it must be that all the pairs have been labeled "unknown" so we can't
                # train on them.
                if len(pairs) == 0:
                    raise ValueError(
                        "No pairs have valid labels. Did you label every"
                        " pair as 'unknown'?"
                    )

            # train the reward model
            with Session() as db:
                raw_pairs = crud.read_pairs(
                    db, run_name=args.name, limit=args.limit, order_by="id", sort="desc"
                )
                pairs = [p for p in raw_pairs if p.label in VALID_LABELS]

                n_train = int(0.8 * len(pairs))
                train_pairs = pairs[:n_train]
                val_pairs = pairs[n_train:]
                train_dataset = utils.PairsDataset(train_pairs)
                val_dataset = utils.PairsDataset(val_pairs)
                train_loader = DataLoader(
                    train_dataset, batch_size=args.reward_batch_size, shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=args.reward_batch_size, shuffle=True
                )

                train_reward_model(
                    model=reward_model,
                    optimizer=reward_optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=args.epochs,
                )

                reward_model.init_reward_normalization(train_loader)

                utils.save_model(
                    db,
                    model=reward_model,
                    optimizer=reward_optimizer,
                    run_name=args.name,
                    iteration=iteration,
                    model_name=settings.REWARD_MODEL_NAME,
                )

        ##############################################
        # END NEW
        ##############################################

    envs.close()
    writer.close()
