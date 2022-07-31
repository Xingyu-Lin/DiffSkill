# go --devices 0 --go -m test2.py -o --dir -1,1,0
# go --devices 0 --go -m test2.py -o --lr 0.01,0.05,0.1 --dir -1,1,0 --no_gap_loss 1
import os

import cv2

from plb.envs.multitask_env import MultitaskPlasticineEnv
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


import argparse

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dir', default=0, choices=[-1, 1, 0], type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--no_gap_loss', default=0, type=int)
    args=parser.parse_args()
    return args


args = get_args()
filepath = f"{args.dir}_{args.lr}_{args.seed}".replace('-', '_')
if args.no_gap_loss:
    filepath = filepath + '_no_gap'

os.makedirs(filepath, exist_ok=True)


def render(images, mode, fps=1, name='xx.webm'):
    print(name)
    if mode == 'plt':
        plt.imshow(images[-1])
        plt.show()
    elif mode == 'human':
        return animate(images, name, _return=True, fps=fps)
    else:
        raise NotImplementedError


def render_pcd(taichi_env):
    p = taichi_env.simulator.x.to_numpy()[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    o3d.visualization.draw_geometries([pcd])


def animate(imgs, filename='animation.webm', _return=True, fps=1):
    print(f'animating {filename}')
    from moviepy.editor import ImageSequenceClip
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=True)


def env_step_range(taichi_env, direction, n_times, cam_positions, cam_rotations):
    images = []
    for _ in range(n_times):
        taichi_env.step(np.array([direction]))
        images.append(render_rgb_multiview(taichi_env, cam_positions, cam_rotations))
    # padding
    for _ in range(5):
        taichi_env.step(np.array([0.] * len(direction)))
        images.append(render_rgb_multiview(taichi_env, cam_positions, cam_rotations))

    return images


def render_rgb(taichi_env):
    frame = taichi_env.render(mode='rgb', target=False,
                              shape=1, primitive=1, spp=2,
                              target_opacity=0.5)   # show target
    frame = np.uint8(frame[:, :, :3])
    return frame


def render_rgb_multiview(taichi_env, cam_positions, cam_rotations):
    images = []
    for (cam_pos, cam_rot) in zip(cam_positions, cam_rotations):
        taichi_env.renderer.set_camera_pose(camera_pos=np.array(cam_pos), camera_rot=np.array(cam_rot))
        images.append(render_rgb(taichi_env))
    img = np.concatenate(images, axis=1)
    return img


def calc_lame_params(E, nu):
    _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    return _mu, _lam

#%%

import os
from plb.envs.env import PATH
from plb.config.utils import load

import os
from plb.envs.env import PATH
from plb.config.utils import load
x = os.path.join(PATH, 'env_configs/cut_rearrange.yml')
cfg = load(x)
print(cfg)
from plb.engine.taichi_env import TaichiEnv
env = TaichiEnv(cfg, nn=False, loss=False)

env.initialize()
initial_state = env.get_state()
#env.render('plt')

#%%

env.set_state(**initial_state)
cam_positions = [(0.5, 2.5, 0.5), (0.5, 0.8, 2.5)]
cam_rotations = [(1.55, 0.), (0.2, 0.)]

def show():
    img = render_rgb_multiview(env, cam_positions, cam_rotations)
    print(img.min(), img.max())
    plt.imshow(img)
    plt.show()

#show()

#%%

images = []
env.set_state(**initial_state)
#images.extend(env_step_range(env,
#                             [-1, 0, 0] + [0, 0, 0] + [0, 0, 0],
#                             5, cam_positions, cam_rotations))
images.extend(env_step_range(env,
                             [0, 0, 0] + [0, 0, 0] + [0, 0, 0, 0],
                             30, cam_positions, cam_rotations))
state = env.get_state()
#render(images, mode='plt', fps=20, name='xx.webm')


#%%


cam_positions = [(0.5, 2.5, 0.5), (0.5, 0.8, 2.5)][1:]
cam_rotations = [(1.55, 0.), (0.2, 0.)][1:]

images = []
env.set_state(**state)

env.primitives[0].position.from_numpy(np.array([[0.5, 0.3, 0.5]]))
images.extend(env_step_range(env,
                             [0.1 * args.dir, 0, 0] + [0, 0, 0] + [0, 0, 0, 0],
                             35, cam_positions, cam_rotations))
images.extend(env_step_range(env,
                             [0, -0.5, 0] + [0, 0, 0] + [0, 0, 0, 0],
                             35, cam_positions, cam_rotations))
images.extend(env_step_range(env,
                             [0, 0.5, 0] + [0, 0, 0] + [0, 0, 0, 0],
                             35, cam_positions, cam_rotations))
images.extend(env_step_range(env,
                             [0, 0, 0] + [0, 0, 0] + [0, 0, 0, 0],
                             25, cam_positions, cam_rotations))
render(images, mode='human', fps=10, name=os.path.join(filepath, 'gt.webm'))


#%%

# build target

goal = env.get_state()['state'][0]
print(goal.shape)

#%%

from plb.lang.solver import Solver
solver = Solver(env, return_deform_grad=False)


#%%

import torch
g = torch.tensor(goal, device='cuda:0')

#%%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(g.detach().cpu().numpy())
label = torch.tensor(np.array(kmeans.labels_)>0, dtype=np.bool).to('cuda:0')

a = g[label]
b = g[~label]
print(label.float().mean(), (~label).float().mean())

#%%
env.set_state(**state)
env.simulator.set_x(0, g.detach().cpu().numpy())
img = env.render('rgb_array')
cv2.imwrite(os.path.join(filepath, 'goal.png'), img)
#%%

def compute_loss(idx, shape, *arguments, **kwargs):
    contact = shape[:, -2]
    x = shape[:, :3]
    a = x[label]
    b = x[~label]
    gap = ((a.mean(dim=0) - b.mean(dim=0))**2).sum(dim=-1)**0.5
    a_close = (((a - a.mean(dim=0)[None, :]) ** 2).sum(dim=-1)**0.5).mean()
    b_close = (((b - b.mean(dim=0)[None, :]) ** 2).sum(dim=-1)**0.5).mean()

    dist = (((x - g)**2).sum(dim=1) ** 0.5).mean() + torch.relu(contact.min())
    if not args.no_gap_loss:
        dist = dist + a_close + b_close - gap
    return dist

initial_action = np.zeros((40, 10))
env.set_state(**state)

#env.render('plt')
#%%
start_stress = 20.
yield_stress = float(env.simulator.yield_stress[0])
def callback(self, iter):
    to = (yield_stress-start_stress) * (iter+1)/200 + start_stress
    if iter < 250:
        to = 50.
    else:
        to = 50.
    env.simulator.yield_stress.fill(to)
out = solver.solve(initial_action, compute_loss, lr=args.lr, max_iter=300, callbacks=(callback,))

#from plb.lang.utils import send_email
#send_email(f"test_cut finished for {filepath}", 'cut')

#out = solver.solve(None, compute_loss)

#%%

#import torch
#torch.save(solver, 'solver')

#%%

import torch
import numpy as np
#solver = torch.load('solver')
T = np.argmin([solver._buffer[i]['loss'] for i in range(len(solver._buffer))])
actions = solver._buffer[T]['action']
print(solver._buffer[T]['loss'])
solver.initial_state['is_copy'] = True
env.set_state(**solver.initial_state)

images = [render_rgb(env)]
for i in actions:
    env.step(i)
    images.append(render_rgb(env))

render(images, 'human', 10, os.path.join(filepath, 'xx.webm'))

#%%
print(1,2,3)
