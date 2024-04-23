from dataclasses import dataclass

import tyro

from ae_rlhf import utils
from ae_rlhf.app.database import Session, init_db
from ae_rlhf.modeling import RandomAgent

from make_env import make_env  # isort:skip


@dataclass
class Args:
    name: str
    """The unique name of the run associated with the data collection.
    It can be any string."""
    n_pairs: int = 500
    """Number of pretraining pairs to collect."""
    segment_length: int = 25
    """The length of a training segment (in frames*4). Note frameskip=4."""
    env_id: str = "Cartpole-v1"
    """The environment id to collect data from."""
    fps: int = 15
    """The frame rate to render the environment at."""


def main():
    args = tyro.cli(Args)
    env = make_env(env_id=args.env_id)()
    agent = RandomAgent(env)
    obs, imgs = utils.sample_segments(
        agent=agent, env=env, n_pairs=args.n_pairs, segment_length=args.segment_length
    )
    init_db()
    with Session() as db:
        utils.save_pairs(
            db,
            obs=obs,
            images=imgs,
            run_name=args.name,
            env_id=args.env_id,
            fps=args.fps,
            iteration=0,
        )


if __name__ == "__main__":
    main()
