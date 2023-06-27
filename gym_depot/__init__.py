from gymnasium.envs.registration import register

register(
    id='BusDepot-v0',
    entry_point='gym_depot.envs:DepotEnv',
    max_episode_steps=2000,
)
