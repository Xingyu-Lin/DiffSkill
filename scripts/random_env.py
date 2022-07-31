import argparse
from tqdm import tqdm
import numpy as np

from plb.envs import make
from plb.engine.taichi_env import TaichiEnv
from plb.utils.visualization_utils import save_numpy_as_gif
from core.diffskill.env_spec import set_render_mode

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='LiftSpread-v1')
args = parser.parse_args()

env = make(args.env_name)
env.seed(100)
taichi_env: TaichiEnv = env.unwrapped.taichi_env
set_render_mode(env, args.env_name, 'mesh')

env.reset()
frames = []
for i in tqdm(range(50)):
    action = env.action_space.sample()
    env.step(action)
    frames.append(env.render(mode='rgb'))
save_numpy_as_gif(np.array(frames), f'{args.env_name}_random.gif')
print('Random actions done!')
