import time
from dataclasses import dataclass

import torch
import tyro
from torch.utils.data import DataLoader

from ae_rlhf import utils
from ae_rlhf.app import crud
from ae_rlhf.app.database import Session
from ae_rlhf.config import CONSTANTS, VALID_LABELS, settings
from ae_rlhf.train_reward import train_reward_model

from modeling import RewardModel  # isort:skip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Args:
    """Arguments for training the reward model"""

    name: str
    """The unique name of the run associated with the pretraining data"""
    lr: float = 2.5e-4
    """The learning rate"""
    batch_size: int = 32
    """The batch size"""
    epochs: int = 200
    """The number of epochs to train for"""


def main():
    """Train the reward model"""
    args = tyro.cli(Args)

    # check if there is a reason we shouldn't train the model and fail if there is.
    with Session() as db:
        if utils.model_exists(
            db, run_name=args.name, iteration=0, model_name=settings.REWARD_MODEL_NAME
        ):
            raise ValueError(
                f"Reward model already exists for run {args.name} with iteration 0"
            )

        raw_pairs = crud.read_pairs(db, run_name=args.name, iteration=0)
        pairs = [p for p in raw_pairs if p.label in VALID_LABELS]
        all_unknown = all(p.label == CONSTANTS.unknown.id for p in raw_pairs)

        # if no pairs at all, suggest running data collection
        if len(raw_pairs) == 0:
            raise ValueError(
                f"No pairs found for run {args.name} with iteration 0 (pretrain)."
                f" Please run `python collect_pretrain_data.py --name {args.name}`"
            )

        # if all pairs have "unknown" label then the labeler did a bad thing
        # this shouldn't happen, but if it does provide some help
        elif all_unknown:
            raise ValueError(
                f"All pairs for run {args.name} with iteration 0 (pretrain) are"
                " unknown. Did you click 'unknown' for every pair in the pretrain set?"
                " The easiest way to recover from this is to start a new run"
                " otherwise delete the pairs or update the labels."
                "\n\n"
                f"labels = {[p.label for p in raw_pairs]}"
            )

    # poll for labels until all the collected pairs are labeled
    while True:
        with Session() as db:
            if utils.all_labeled(db, run_name=args.name, iteration=0):
                break
            else:
                print("Waiting for all pairs to be labeled...")
                print(f"Visit {settings.API_URL}/feedback/{args.name} to label them.")
                time.sleep(5)

    # finally train the model
    with Session() as db:
        raw_pairs = crud.read_pairs(db, run_name=args.name, iteration=0)
        pairs = [p for p in raw_pairs if p.label in VALID_LABELS]

        n_train = int(0.8 * len(pairs))

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:]
        train_dataset = utils.PairsDataset(train_pairs)
        val_dataset = utils.PairsDataset(val_pairs)
        if len(val_dataset) == 0:
            raise ValueError(
                "No validation pairs."
                f" len(pairs)={len(pairs)}, n_train={n_train},"
                f" len(val_dataset)={len(val_dataset)}"
            )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        assert pairs[0].run is not None
        model = RewardModel().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        train_reward_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
        )

        model.init_reward_normalization(train_loader)

        utils.save_model(
            db,
            model=model,
            optimizer=optimizer,
            run_name=args.name,
            iteration=0,
            model_name=settings.REWARD_MODEL_NAME,
        )


if __name__ == "__main__":
    main()
