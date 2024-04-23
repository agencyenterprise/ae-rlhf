import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    StickyActionEnv,
)


def make_env(env_id: str, fps: int = 15):
    def thunk():
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            frameskip=1,
            repeat_action_probability=0.00,
        )

        env.metadata["render_fps"] = fps
        env = wrap_atari_env(env)
        return env

    return thunk


def wrap_atari_env(env: gym.Env):
    """Applies standard wrappers for Atari games."""
    env = NoopResetEnv(env, noop_max=30)
    env = StickyActionEnv(env, action_repeat_probability=0.25)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    return env
