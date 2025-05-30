from gymnasium.envs.registration import register

register(
    id="gymnasium_env/KnightWorldEnv-v0",
    entry_point="gymnasium_env.envs:KnightWorldEnv",
)

register(
    id="gymnasium_env/GridWorldEnv-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

