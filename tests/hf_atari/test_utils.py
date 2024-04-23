import numpy as np
from ae_rlhf import utils
from ae_rlhf.app.models import Pair
from sqlmodel import select


def test_database_is_rolled_back_on_save_error(session, monkeypatch):
    """Verifies that database entries are not written if there is an error on save."""

    class SaveError(Exception):
        pass

    def error_on_save(*args, **kwargs):
        raise SaveError("Error on save.")

    monkeypatch.setattr(utils, "save_obs", error_on_save)
    monkeypatch.setattr(utils, "save_video", error_on_save)

    with session as db:
        assert len(db.exec(select(Pair)).all()) == 0

    try:
        with session as db:
            utils.save_pair(
                db,
                obs=(np.zeros((4, 84, 84)), np.zeros((4, 84, 84))),
                imgs=(np.zeros((4, 84, 84, 3)), np.zeros((4, 84, 84, 3))),
                run_name="test",
                env_id="fake-env",
                iteration="0",
                fps=4,
            )
    except SaveError:
        pass

    with session as db:
        assert len(db.exec(select(Pair)).all()) == 0
