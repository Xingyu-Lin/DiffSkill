import gym
from .env import PlasticineEnv
from gym import register

ENVS = []
for env_name in ['Move', 'Torus', 'Rope', 'Writer', "Pinch", "Rollingpin", "Chopsticks", "Table", 'TripleMove', 'Assembly']:
    for id in range(5):
        register(
            id=f'{env_name}-v{id + 1}',
            entry_point=f"plb.envs.env:PlasticineEnv",
            kwargs={'cfg_path': f"{env_name.lower()}.yml", "version": id + 1},
            max_episode_steps=50
        )

register(id='MultiMerge-v1',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "multi_merge.yml", "version": 1},
         max_episode_steps=50)

register(id='PushSpread-v1',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "push_spread.yml", "version": 1},
         max_episode_steps=50)

register(id='GatherMove-v1',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "gather_move.yml", "version": 1},
         max_episode_steps=50)

register(id='LiftSpread-v1',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "lift_spread.yml", "version": 1},
         max_episode_steps=50)

register(id='CutRearrange-v1',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "../cut/cut_rearrange.yml", "version": 1},
         max_episode_steps=50)

register(id='CutRearrange-v2',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "../cut/cut_rearrange_0528.yml", "version": 1},
         max_episode_steps=50)

register(id='Multicut-v1',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "multicut.yml", "version": 1},
         max_episode_steps=100)

register(id='CutRearrangeSpread-v1',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "cut_rearrange_spread.yml", "version": 1},
         max_episode_steps=100)

# Single-stage task
register(id='Lift-v1', entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "single_stage/lift.yml", "version": 1},
         max_episode_steps=50)
register(id='Spread-v1', entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "single_stage/spread.yml", "version": 1},
         max_episode_steps=50)


def make(env_name, nn=False, return_dist=False, **kwargs):
    env: PlasticineEnv = gym.make(env_name, nn=nn, return_dist=return_dist, **kwargs)
    return env
