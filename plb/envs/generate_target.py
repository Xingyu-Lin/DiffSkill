import taichi as ti
import numpy as np
import cv2
import os
from plb.utils.visualization_utils import save_rgb
target_path = './plb/envs/targets/'


def generate_target(env_name):
    ti.reset()
    from plb.envs.multitask_env import MultitaskPlasticineEnv
    env = MultitaskPlasticineEnv(cfg_path=f'{env_name}_target.yml')
    taichi_env = env.taichi_env

    if env_name == 'push_spread':
        N = 10
        xs = np.linspace(0.4, 0.55, N)

        def case1(cfg):
            print('setting x pos to ', xs[i], ' i:', i)
            pos = eval(cfg.SHAPES[0]['init_pos'])
            new_pos = (xs[i], pos[1], pos[2])
            cfg.SHAPES[0]['init_pos'] = new_pos

        target_imgs = []
        for i in range(10):
            env.reset(target_cfg_modifier=case1)
            img = taichi_env.render(mode='rgb', target=False, shape=1, primitive=1)
            taichi_env.simulator.clear_and_compute_grid_m(0)
            np.save(os.path.join(target_path, f'target_{i}.npy'), taichi_env.simulator.get_m())
            save_rgb(os.path.join(target_path, 'target_{}.png'.format(i)), img)
            target_imgs.append(img)

        def case2(cfg):
            pos = eval(cfg.SHAPES[0]['init_pos'])
            cfg.SHAPES[0]['shape'] = 'sphere'
            del cfg.SHAPES[0]['width']
            cfg.SHAPES[0]['radius'] = 0.07
            new_pos = (xs[i], 0.08, pos[2])
            cfg.SHAPES[0]['init_pos'] = new_pos

        for i in range(10):
            env.reset(target_cfg_modifier=case2)
            img = taichi_env.render(mode='rgb', target=False, shape=1, primitive=1)
            taichi_env.simulator.clear_and_compute_grid_m(0)
            np.save(os.path.join(target_path, f'target_{i + 10}.npy'), taichi_env.simulator.get_m())
            save_rgb(os.path.join(target_path, 'target_{}.png'.format(i + 10)), img)
            target_imgs.append(img)

        np.save(os.path.join(target_path, 'target_imgs.npy'), np.array(target_imgs))
    else:
        raise NotImplementedError


if __name__ == '__main__':
    generate_target('push_spread')
