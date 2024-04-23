import gymnasium as gym


def make_env(env_id="CartPole-v1", fps: int = 15):
    env_id = "CartPole-v1"

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env.metadata["render_fps"] = fps
        return env

    return thunk
