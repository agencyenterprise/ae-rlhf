import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings
from sqlalchemy.engine.url import URL


@dataclass
class Preference:
    id: int
    name: str


Left: Preference = Preference(id=0, name="left")
Right: Preference = Preference(id=1, name="right")
Tie: Preference = Preference(id=2, name="tie")
Unknown: Preference = Preference(id=3, name="unknown")


class CONSTANTS:
    left: Preference = Left
    right: Preference = Right
    tie: Preference = Tie
    unknown: Preference = Unknown


VALID_LABELS = set([CONSTANTS.left.id, CONSTANTS.right.id, CONSTANTS.tie.id])


class Settings(BaseSettings):
    ENV: str
    DEBUG: bool = False
    RELOAD: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_URL: str = "http://localhost:8000"
    DATABASE_DRIVER: str = "sqlite"
    DATABASE_NAME: str = "rlhf.db"
    DATABASE_USER: str | None = None
    DATABASE_PASSWORD: str | None = None
    DATABASE_HOST: str | None = None
    DATABASE_PORT: int | None = None
    SAVE_LOCATION: str | None = None
    VIDEO_FORMAT: str = "webm"
    VIDEO_4CC: str = "VP80"
    REWARD_MODEL_NAME: str = "reward"
    PPO_MODEL_NAME: str = "ppo_agent"
    DPO_MODEL_NAME: str = "dpo_agent"

    @property
    def SAVE_DIR(self) -> str:
        if self.SAVE_LOCATION is None:
            path = Path.home() / ".ae_rlhf" / f"{self.ENV}" / "data"
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
            # return str(Path(__file__).parents[1] / "data" / f"{self.ENV}")
        return self.SAVE_LOCATION

    @property
    def OBS_TEMPLATE(self) -> str:
        save_path = Path(self.SAVE_DIR)
        save_path = save_path / "{run_name}" / "iter_{iteration:0>6}" / "obs_{id}.npy"
        return str(save_path)

    @property
    def VIDEO_TEMPLATE(self) -> str:
        fmt = self.VIDEO_FORMAT
        sp = Path(self.SAVE_DIR)
        sp = sp / "{run_name}" / "iter_{iteration:0>6}" / f"video_{{id}}.{fmt}"
        return str(sp)

    @property
    def MODEL_TEMPLATE(self) -> str:
        sv_path = Path(self.SAVE_DIR)
        filename = "{model_name}_{id}.pt"
        sv_path = sv_path / "{run_name}" / "iter_{iteration:0>6}" / filename
        return str(sv_path)

    @property
    def DATABASE_PATH(self) -> str:
        save_path = Path(self.SAVE_DIR)
        save_path = save_path / f"{self.DATABASE_NAME}"
        return str(save_path)

    @property
    def DATABASE_URL(self) -> str:
        return str(
            URL.create(
                drivername=self.DATABASE_DRIVER,
                username=self.DATABASE_USER,
                password=self.DATABASE_PASSWORD,
                host=self.DATABASE_HOST,
                port=self.DATABASE_PORT,
                database=self.DATABASE_PATH,
            )
        )


class ProdSettings(Settings):
    ENV: str = "prod"
    # DATABASE_NAME: str = str(Path(__file__).parent / "app" / "prod.db")


class DevSettings(Settings):
    ENV: str = "dev"
    # DATABASE_NAME: str = str(Path(__file__).parent / "app" / "dev.db")
    RELOAD: bool = True
    DEBUG: bool = True


class TestSettings(Settings):
    ENV: str = "test"
    # DATABASE_NAME: str = ""


def get_settings(env: str | None = None):
    env = env or os.getenv("ENV", "prod")
    assert isinstance(env, str)  # for mypy
    options = {
        "prod": ProdSettings,
        "dev": DevSettings,
        "test": TestSettings,
    }
    if env.casefold() not in options:
        raise ValueError(f"Invalid environment {env} expected one of {options.keys()}")
    return options[env.casefold()]()


# default settings from env
settings = get_settings()
