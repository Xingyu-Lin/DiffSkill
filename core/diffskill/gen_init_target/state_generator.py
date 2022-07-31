import numpy as np
import lzma
import os
import pickle
from core.diffskill.env_spec import set_render_mode
from plb.envs import make
from plb.utils.visualization_utils import save_rgb, save_numpy_as_gif


def clean_dir(folder):
    if os.path.isdir(folder):
        # ans = query_yes_no(f'Cached state path already exists. Deleting {folder}?', default='no')
        ans = True
        if ans:
            os.system(f'rm -rf {folder}')
    os.makedirs(folder, exist_ok=True)


class StateGenerator(object):
    def __init__(self, env_name, save_dir, mode, img_size=64, do_clean_dir=True):
        np.random.seed(0)
        self.env = make(env_name, nn=False, generating_cached_state=True)
        self.taichi_env = self.env.taichi_env
        set_render_mode(self.env, env_name, 'mesh')
        self.init_dir, self.target_dir = os.path.join(save_dir, 'init'), os.path.join(save_dir, 'target')
        if do_clean_dir:
            clean_dir(self.init_dir)
            clean_dir(self.target_dir)
        self.init_imgs = []
        self.target_imgs = []
        self.mode = mode
        self.img_size = img_size

    def save_init(self, idx, state=None):
        if state is not None:
            self.taichi_env.set_state(**state)
        else:
            state = self.taichi_env.get_state()
        img = self.taichi_env.render(mode='rgb', img_size=self.img_size)
        self.init_imgs.append(img)

        state_name = os.path.join(self.init_dir, f'state_{idx}.xz')
        with lzma.open(state_name, 'wb') as f:
            pickle.dump(state, f, protocol=4)
        save_rgb(os.path.join(self.init_dir, f'state_{idx}.png'), np.array(img[:, :, :3]).astype(np.float32))

    def save_init_gif(self):
        gifpath = os.path.join(self.init_dir, 'vis_all.gif')
        save_numpy_as_gif(np.array(self.init_imgs)[:, :, :, :3], gifpath)

    def save_target(self, idx, state=None):
        if state is not None:
            self.taichi_env.set_state(**state)
        img = self.taichi_env.render(mode='rgb', img_size=self.img_size)
        self.target_imgs.append(img)
        np.save(os.path.join(self.target_dir, f'target_{idx}.npy'), self.taichi_env.simulator.get_x(0))
        save_rgb(os.path.join(self.target_dir, f'target_{idx}.png'), np.array(img[:, :, :3]).astype(np.float32))

    def save_target_gif(self):
        gifpath = os.path.join(self.target_dir, 'vis_all.gif')
        save_numpy_as_gif(np.array(self.target_imgs)[:, :, :, :3], gifpath)
        np.save(os.path.join(self.target_dir, 'target_imgs.npy'), np.array(self.target_imgs))

    def generate(self):
        self._generate()
        self.save_init_gif()
        self.save_target_gif()

    def _generate(self):
        raise NotImplementedError


if __name__ == '__main__':
    import argparse
    from core.diffskill.env_spec import get_generator_class

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()

    generator = get_generator_class(args.env_name)(args.env_name, args.save_dir, args.mode, args.img_size)
    generator.generate()
