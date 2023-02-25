import gym
from gym.spaces import Box
import os
import glob
import yaml
import numpy as np

np.set_printoptions(precision=4, suppress=True)
from ..config import load
from yacs.config import CfgNode
from .utils import merge_lists
from copy import deepcopy
import gzip
import lzma
import pickle
import os.path as osp

PATH = os.path.dirname(os.path.abspath(__file__))
import torch

device = 'cuda'
from plb.engine.primitive.primitives import RollingPinExt, Gripper

def load_target_pcs(cached_state_path):
    target_mass_grids = []
    import glob
    from natsort import natsorted
    target_paths = natsorted(glob.glob(os.path.join(cached_state_path, 'target/target_[0-9]*.npy')))
    for path in target_paths:
        target_mass_grid = np.load(path)
        target_mass_grids.append(target_mass_grid)
    return np.array(target_mass_grids)

class MultitaskPlasticineEnv(gym.Env):
    def __init__(self, cfg_path, version=None, nn=False, loss=True, return_dist=False, generating_cached_state=False):
        self.variants = None
        self.target_cfg_modifier = None  # target_cfg_modifier: Change cfg after it is loaded
        from ..engine.taichi_env import TaichiEnv
        self.cfg_path = cfg_path
        cfg = self.load_varaints(cfg_path)
        self.taichi_env = TaichiEnv(cfg, nn, loss=loss, return_dist=return_dist)  # build taichi environment
        self.taichi_env.initialize(cfg, target_path=None)
        self.generating_cached_state = generating_cached_state
        self.cfg = cfg.ENV
        if not self.generating_cached_state:
            self.num_inits = len(glob.glob(osp.join(self.cfg.cached_state_path, 'init/state_*.xz')))
            self.num_targets = len(glob.glob(osp.join(self.cfg.cached_state_path, 'target/target_[0-9]*.npy')))
            self._load_target_imgs()
            self.target_pcs = load_target_pcs(self.cfg.cached_state_path)

        self.taichi_env.set_copy(True)
        self._init_state = self.taichi_env.get_state()
        self._n_observed_particles = self.cfg.n_observed_particles

        _ = self.reset()
        self.observation_space = Box(-np.inf, np.inf, (0,))  # Observation space should not be used
        self.action_space = Box(-1, 1, (self.taichi_env.primitives.action_dim,))

    def _load_target_imgs(self):
        np_target_imgs = np.load(os.path.join(self.cfg.cached_state_path, 'target/target_imgs.npy'))
        self.target_imgs = np_target_imgs

    def reset(self, init_v=None, target_v=None, target_cfg_modifier=None, contact_loss_mask=None):  # Init version and target version
        # Reload cfg each time
        cfg = self.load_varaints(self.cfg_path)
        self.cfg = cfg.ENV
        if target_cfg_modifier is not None:
            target_cfg_modifier(cfg)

        if not self.generating_cached_state:
            if init_v is None:
                assert target_v is None
                init_v = np.random.randint(0, self.num_inits)
                target_v = np.random.randint(0, self.num_targets)
            self.init_v, self.target_v = init_v, target_v
            self.target_img, self.target_pc = self.target_imgs[target_v], self.target_pcs[target_v]
            target_path = osp.join(self.cfg.cached_state_path, 'target', f'target_{target_v}.npy')
            print('Env reseting to: {}, init v: {}, target v: {}'.format(target_path, init_v, target_v))
        else:
            target_path = None

        self.taichi_env.initialize(cfg, target_path=target_path)
        self.taichi_env.set_copy(True)

        if not self.generating_cached_state:
            state_path = os.path.join(osp.join(self.cfg.cached_state_path, 'init', f'state_{init_v}.xz'))
            with lzma.open(state_path, 'rb') as f:
                self._init_state = pickle.load(f)
            self.taichi_env.set_state(**self._init_state)
        else:
            print('Env reset: No initial state during cache generation')

        self._n_observed_particles = self.cfg.n_observed_particles

        # self.taichi_env.set_state(**self._init_state)
        self._recorded_actions = []
        self.taichi_env.set_init_emd()
        print('emd after reset:', self.taichi_env.init_emd)

        # Set contact loss mask
        self.taichi_env.contact_loss_mask = torch.zeros(len(self.taichi_env.primitives), device=device)

        if contact_loss_mask is not None:
            if isinstance(contact_loss_mask, int) or isinstance(contact_loss_mask, float):
                self.taichi_env.contact_loss_mask[contact_loss_mask] = 1.
            elif isinstance(contact_loss_mask, list):
                for i in range(len(self.taichi_env.primitives)):
                    self.taichi_env.contact_loss_mask[i] = contact_loss_mask[i]
        else:
            print('======================WARNING: contact loss mask not set================')
        return self._get_obs()

    def reset_primitive(self):
        self.taichi_env.set_primitive_state(**self._init_state)

    @property
    def action_dims(self):
        return self.taichi_env.primitives.action_dims

    def state_to_vec(self, d):
        vec = np.concatenate([d['particles'].flatten(), d['tool_state'].flatten(), d['tool_particles'].flatten()]).flatten()
        return vec

    def _get_obs(self, t=0):
        particles, tool_state = self.taichi_env.get_obs(t, device='cpu')
        tool_particles = self.taichi_env.get_tool_particles(0)
        # TODO hardcoded sampling of 2000 particles. Add as a hyperparameter of the environment
        # idx = np.random.choice(range(particles.shape[0]), 1000, replace=False)
        idx = np.arange(0, 1000)
        return self.state_to_vec({'particles': particles[idx],
                                  'tool_state': tool_state,
                                  'tool_particles': tool_particles})

    def step(self, action):
        action = np.clip(action, -1., 1.)
        self.taichi_env.step(action)
        r, info = self.taichi_env.get_reward_and_info()
        self._recorded_actions.append(action)
        obs = self._get_obs()
        if np.isnan(obs).any() or np.isnan(r):
            if np.isnan(r):
                print('nan in r')
            import pickle, datetime
            with open(f'{self.cfg_path}_nan_action_{str(datetime.datetime.now())}', 'wb') as f:
                pickle.dump(self._recorded_actions, f)
            raise Exception("NaN..")
        return obs, r, False, info

    def render(self, mode='human', *args, **kwargs):
        return self.taichi_env.render(mode, *args, **kwargs)

    @classmethod
    def load_varaints(self, cfg_path):
        cfg_path = os.path.join(PATH, cfg_path)
        cfg = load(cfg_path)
        return cfg

    def get_state(self):
        state = self.taichi_env.get_state()
        # state['loss_state'] = self.taichi_env.loss.get_state()
        return state

    def set_state(self, state):
        self.taichi_env.set_state(**state)
        # self.taichi_env.loss.set_state(**state['loss_state'])

    def get_primitive_state(self):
        return [i.get_state(0) for i in self.taichi_env.primitives]

    def primitive_reset_to(self, idx, reset_states, thr=1e-2, img_size=64, reset_gripper2=False):
        reset_state = reset_states[idx]
        states, actions, obses, rs, infos = [], [], [], [], []
        step, max_step = 0, 200
        while step < max_step:
            curr_state = self.taichi_env.primitives[idx].get_state(0)
            action = np.zeros(self.taichi_env.primitives.action_dim)
            if (idx == 1 and reset_gripper2) or (idx == 1 and 'cut_rearrange' in self.cfg_path and 'cut_rearrange_spread' not in self.cfg_path):
                t_action = np.zeros(7)
                if abs(reset_state[-1] - curr_state[-1]) > 0.01:
                    t_action[-1] = (curr_state[-1] - reset_state[-1]) * 10
                else:
                    dist = np.linalg.norm(reset_state[[0, 2]] - curr_state[[0, 2]])
                    if dist < thr:
                        if abs(reset_state[1] - curr_state[1]) < thr:
                            t_action = None
                        else:
                            t_action[1] = (reset_state[1] - curr_state[1]) * 40
                    else:
                        if abs(curr_state[1] - 0.3) > thr:
                            t_action = np.zeros(7)
                            t_action[1] = (0.3 - curr_state[1]) * 40
                        else:
                            t_action[[0, 2]] = (reset_state[[0, 2]] - curr_state[[0, 2]]) * 15
                if t_action is not None:
                    t_action = t_action.clip(-1, 1)
            # elif (idx == 0 and 'cut_rearrange' in self.cfg_path):
            #     print(f'Reset state: {reset_state}, curr_state {curr_state}')
            #     if abs(reset_state[1] - curr_state[1]) > 0.01:  # First lift the knife
            #         t_action = np.zeros(3)
            #         t_action[1] = (curr_state[1] - reset_state[1]) * 10
            #     else:
            #         t_action = self.taichi_env.primitives[idx].inv_action(curr_state, reset_state, thr=5e-3)
            else:
                t_action = self.taichi_env.primitives[idx].inv_action(curr_state, reset_state, thr=5e-3)
            if t_action is None:
                break
            l, r = self.taichi_env.primitives.action_dims[idx], self.taichi_env.primitives.action_dims[idx + 1]
            action[l:r] = t_action
            state, r, _, info = self.step(action)
            obs = self.taichi_env.render(mode='rgb', img_size=img_size)
            states.append(state)
            actions.append(action)
            obses.append(obs)
            rs.append(r)
            infos.append(info)
            step += 1
        print('reset primitive in {} steps'.format(step))
        return np.array(states), np.array(actions), np.array(obses), np.array(rs), infos

    def set_particles_to_goal(self, target_v):
        if not hasattr(self, 'full_target_pcs'):
            self.full_target_pcs = load_target_pcs(self.taichi_env.cfg.ENV.cached_state_path)
        state = self.get_state()
        state['state'][0] = self.full_target_pcs[target_v]
        self.set_state(state)
