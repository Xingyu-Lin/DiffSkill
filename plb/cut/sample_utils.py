import os
import copy
import tqdm
import numpy as np
from plb.envs.env import PATH
from plb.config.utils import load
import matplotlib.pyplot as plt

def animate(imgs, filename='animation.webm', _return=True, fps=10):
    if isinstance(imgs, dict):
        imgs = imgs['image']
    print(f'animating {filename}')
    from moviepy.editor import ImageSequenceClip
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps)
    if _return:
        from IPython.display import Video
        return Video(filename, embed=True)


def calc_lame_params(E, nu):
    _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    return _mu, _lam

def init_env():
    x = os.path.join(PATH, '../cut/cut_rearrange.yml')
    cfg = load(x)
    from plb.engine.taichi_env import TaichiEnv
    env = TaichiEnv(cfg, nn=False, loss=False)
    env.initialize(cfg)
    initial_state = env.get_state()
    env.set_state(**initial_state)
    for i in range(50):
        env.step(np.array([0, 0, 0] + [0, 0, 0] + [0, 0, 0, 0]))
    state = env.get_state()
    return env, state


def env_step_range2(direction, n_times):
    return [direction] * n_times + [np.array([0.] * len(direction))] * 5

def execute(env, initial_state, actions, filename='xx.webm', spp=2, verbose=False, render_freq=1, img_size=512, **kwargs):
    env.set_state(initial_state)
    if filename is not None:
        def render_it():
            img = env.render('rgb', spp=spp, img_size=img_size)
            return np.uint8((img * 255).clip(0, 255))
        images = [render_it()]

    ran = enumerate(actions)
    if verbose:
        ran = tqdm.tqdm(ran, total=len(actions))

    for idx, act in ran:
        env.step(act)
        if filename is not None:
            if idx % render_freq == 0:
                img = render_it()
                images.append(img)

    if filename is not None:
        if filename.endswith('.webm'):
            return animate(images, filename, **kwargs)
        else:
            return np.array(images)
    else:
        return env.get_state()


def cut(env, state, dir, *args, slide=0, **kwargs):
    actions = env_step_range2([0.06 * dir, 0, 0] + [0, 0, 0] + [0, 0, 0, 0], 35) + \
              env_step_range2([0, -0.5, 0] + [0, 0, 0] + [0, 0, 0, 0], 35)

    if slide:
        actions += env_step_range2([0.3 * slide, 0, 0] + [0, 0, 0] + [0, 0, 0, 0], 20)

    actions += env_step_range2([0, 1., 0] + [0, 0, 0] + [0, 0, 0, 0] , 50)

    return execute(env, state, actions, *args, **kwargs)


def find_cluster(state):
    s = state['state'][0]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(s)
    return np.array(kmeans.labels_)>0

def move_cluster(env, state, flag, dir_y, dir_z=0):
    # need to move forward..
    new_state = copy.deepcopy(state)
    print(flag.sum())
    #print(new_state['state'][0][flag].mean(axis=0))
    new_state['state'][0][flag, 2] += dir_y * 0.25
    new_state['state'][0][flag, 1] += dir_z
    #print(new_state['state'][0][flag].mean(axis=0))
    return new_state


def generate(env, initial_state, dir=0, slide=1., lift=0., sample_box=None):
    # return several stages ..


    trajs = []
    state1 = cut(env, initial_state, dir, slide=slide, filename=None)
    trajs.append([initial_state, state1, np.ones(len(state1['state'][0])), 0])
    flag = find_cluster(state1)

    state2 = move_cluster(env, state1, np.logical_not(flag), -1, dir_z=0.)
    state3 = move_cluster(env, state2, flag, 1, dir_z=lift)

    trajs.append([state1, state3, flag, 1])
    trajs.append([state2, state3, flag, 1])
    return trajs
