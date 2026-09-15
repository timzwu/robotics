import gymnasium as gym

gym.register(
    id="So101-Blocks-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.blocks_env_cfg:BlocksEnvCfg"},
)

gym.register(
    id="So101-Blocks-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.blocks_env_cfg:BlocksDREnvCfg"},
)
