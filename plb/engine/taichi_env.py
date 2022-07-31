from functools import partial
import numpy as np
import cv2
import taichi as ti
import torch
import os
import pykeops
from plb.engine.primitive.primitives import Gripper

myhost = os.uname()[1]
if 'vision' in myhost:
    dir = os.path.join(os.getenv('XINGYUHOME'), 'pykeops_cache/' + torch.cuda.get_device_name(0).replace(' ', '_'))
else:
    dir = os.path.join(os.path.expanduser("~"), '.pykeops_cache/' + torch.cuda.get_device_name(0).replace(' ', '_'))
os.makedirs(dir, exist_ok=True)
print('Setting pykeops dir to ', dir)
pykeops.set_bin_folder(dir)

# TODO: run on GPU, fast_math will cause error on float64's sqrt; removing it cuases compile error..
ti.init(arch=ti.gpu, debug=False, fast_math=True)


def create_emd_loss():
    from geomloss import SamplesLoss
    loss = SamplesLoss(loss='sinkhorn', p=1, blur=0.001)
    return loss

def create_chamfer_loss(bidirectional):
    from chamferdist import ChamferDistance
    print("Using chamfer loss")
    loss = partial(ChamferDistance(), bidirectional=bidirectional)
    return loss

def visualize_pc(pc):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.1, marker='^', alpha=0.2)
    plt.show()


@ti.data_oriented
class TaichiEnv:
    def __init__(self, cfg, nn=False, loss=True, return_dist=False):
        """
        A taichi env builds scene according the configuration and the set of manipulators
        """
        # primitives are environment specific parameters ..
        # move it inside can improve speed; don't know why..
        from .mpm_simulator import MPMSimulator
        from .primitive import Primitives
        from .renderer import Renderer
        from plb.env_modeling.tina_renderer import TinaRenderer
        from .shapes import Shapes
        from .losses import Loss
        from .nn.mlp import MLP

        self.has_loss = loss
        self.cfg = cfg.ENV
        self.env_name = self.cfg.env_name if 'env_name' in self.cfg else None
        self.primitives = Primitives(cfg.PRIMITIVES)
        self.shapes = Shapes(cfg.SHAPES)
        self.init_particles, self.particle_colors = self.shapes.get()

        cfg.SIMULATOR.defrost()
        self.n_particles = len(self.init_particles)

        self.simulator = MPMSimulator(cfg.SIMULATOR, self.primitives)
        self.dim = self.simulator.dim
        if 'name' not in cfg.RENDERER.__dict__ or cfg.RENDERER.name == 'tina':
            self.renderer = TinaRenderer(cfg.RENDERER, self.primitives)
            self.renderer_name = 'tina'
        if cfg.RENDERER.name == 'plb':
            raise NotImplementedError
            self.renderer = Renderer(cfg.RENDERER, self.primitives)
            self.renderer_name = 'plb'

        if nn:
            self.nn = MLP(self.simulator, self.primitives, (256, 256))

        self._is_copy = True

        self.target_x, self.tensor_target_x = None, None
        self.device = 'cuda'
        self.contact_loss_mask = None
        self.return_dist = return_dist
        if self.return_dist:
            self.dists = []
            self.dists_start_idx = []
            for i in self.primitives:
                self.dists_start_idx.append(len(self.dists))
                if isinstance(i, Gripper):
                    self.dists += [ti.field(dtype=self.simulator.dtype, shape=(self.n_particles,), needs_grad=True)]
                self.dists += [ti.field(dtype=self.simulator.dtype, shape=(self.n_particles,), needs_grad=True)]
            self.dists_start_idx.append(len(self.dists))

    def set_copy(self, is_copy: bool):
        self._is_copy = is_copy

    def initialize(self, cfg=None, target_path=None):
        if cfg is not None:
            from .shapes import Shapes
            from .primitive import Primitives
            self.cfg = cfg
            self.shapes = Shapes(self.cfg.SHAPES)
            self.init_particles, self.particle_colors = self.shapes.get()
            self.primitives.update_cfgs(cfg.PRIMITIVES)
            if self.has_loss and target_path is not None:
                self.load_target_x(target_path)
        self.n_particles = len(self.init_particles)
        # initialize all taichi variable according to configurations..
        self.primitives.initialize(self.cfg.ENV.cached_state_path)
        self.simulator.initialize(self.n_particles)
        self.renderer.initialize(self.cfg.RENDERER)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        self.init_particles, self.particle_colors = self.shapes.get()

    def load_target_x(self, path):
        self.target_x = np.load(path)
        self.tensor_target_x = torch.FloatTensor(self.target_x).to(self.device)

    def render(self, mode='human', **kwargs):
        assert self._is_copy, "The environment must be in the copy mode for render ..."
        if self.renderer_name == 'plb':
            if self.n_particles > 0:
                x = self.simulator.get_x(0)
                self.renderer.set_particles(x, self.particle_colors)
            img = self.renderer.render_frame(**kwargs)
            rgb = np.uint8(img[:, :, :3].clip(0, 1) * 255)

            if mode == 'human':
                cv2.imshow('x', rgb[..., ::-1])
                cv2.waitKey(1)
            elif mode == 'plt':
                import matplotlib.pyplot as plt
                plt.imshow(rgb)
                plt.show()
            else:
                return img
        elif self.renderer_name == 'tina':
            if self.n_particles > 0:
                x = self.simulator.get_x(0)
                self.renderer.set_particles(x, self.particle_colors)
            img = self.renderer.render(mode=mode, **kwargs)
            return img
        else:
            raise NotImplementedError

    def step(self, action=None):
        if action is not None:
            action = np.array(action)
        self.simulator.step(is_copy=self._is_copy, action=action)

    def get_state(self):
        assert self.simulator.cur == 0
        return {
            'state': self.simulator.get_state(0),
            'softness': self.primitives.get_softness(),
            'is_copy': self._is_copy
        }

    def set_state(self, state, softness=None, is_copy=None, **kwargs):
        if softness is None:
            softness = state['softness']
            is_copy = state['is_copy']
            state = state['state']

        self.n_particles = len(state[0])
        self.simulator.cur = 0
        self.simulator.set_state(0, state)
        self.primitives.set_softness(softness)
        self._is_copy = is_copy

    def set_primitive_state(self, state, softness, is_copy, **kwargs):
        self.simulator.set_primitive_state(0, state)

    def get_use_gripper_primitive(self):
        if not hasattr(self, 'use_gripper_primitive'):
            from plb.engine.primitive.primitives import Gripper
            self.use_gripper_primitive = False
            for i in self.primitives.primitives:
                if isinstance(i, Gripper):
                    self.use_gripper_primitive = True
        return self.use_gripper_primitive

    def get_contact_loss(self, shape, tool, contact_mask=None):
        if contact_mask is not None:
            assert len(shape) == len(contact_mask), f"pc shape: {shape.shape}, shape of contact mask: {contact_mask.shape}"
            shape = shape[contact_mask]

        if self.env_name == 'CutRearrange-v1' or self.env_name == 'CutRearrange-v2':
            center = tool[1][:3]
            gap = tool[1][7]
            z1 = center[2] - gap / 2 + 0.015  # surface of two board. contact loss to the inner center of the plate
            z2 = center[2] + gap / 2 - 0.015
            a = torch.stack([center[0], center[1], z1])
            b = torch.stack([center[0], center[1], z2])
            d1 = ((a[None, :] - shape[:, :3]) ** 2).sum(axis=1).min()
            d2 = ((b[None, :] - shape[:, :3]) ** 2).sum(axis=1).min()
            gripper_dist = (d1.clamp(0.00005, 1e9) + d2.clamp(0.00005, 1e9) - 0.0001)

            knife_dists = shape[:, 6:7]  # manipulator distance
            knife_dist = knife_dists.min(axis=0)[0].clamp(0, 1e9)
            dists = torch.cat([knife_dist, gripper_dist[None]]) # Not squared
            assert self.contact_loss_mask.shape == dists.shape
            dist = (self.contact_loss_mask * dists).sum()
        else:
            if self.get_use_gripper_primitive():
                dists = shape[:, -len(self.primitives) - 1:]  # manipulator distance
                gripper_dist = dists[:, 0] + dists[:, 1]
                dists = torch.cat([gripper_dist[:, None], dists[:, 2:]], dim=1)
            else:
                dists = shape[:, -len(self.primitives):]  # manipulator distance
            min_ = dists.min(axis=0)[0].clamp(0, 1e9)
            assert self.contact_loss_mask.shape == min_.shape
            min_ = self.contact_loss_mask * min_
            dist = (min_ ** 2).sum() * 1e-3   
        return dist

    # def compute_loss(self, idx, shape, tool, *args):  # shape: n x (6 + k), first 3 position, 3 velocity, dist to manipulator, k: number of manipulator
    #     """ Loss for traj_opt"""
    #     dist = self.get_contact_loss(shape)
    #     x = shape[:, :3]
    #
    #     if not hasattr(self, 'loss_fn'):
    #         self.loss_fn = SamplesLoss(loss='sinkhorn', p=1, blur=0.001)
    #
    #     import time
    #     st = time.time()
    #     dist += self.loss_fn(x.contiguous(), self.tensor_target_x)
    #     return dist
    def update_loss_fn(self, loss_type='emd'):
        self.loss_type = loss_type
        if loss_type == 'emd':
            self.loss_fn = create_emd_loss()
        elif loss_type == 'oneway_chamfer':
            self.loss_fn = create_chamfer_loss(bidirectional=False)
        elif loss_type == 'twoway_chamfer':
            self.loss_fn = create_chamfer_loss(bidirectional=True)

    def compute_loss(self, idxes, observations,
                     vel_loss_weight, state_mask=None, goal_mask=None, loss_type='emd'):  # shape: n x (6 + k), first 3 position, 3 velocity, dist to manipulator, k: number of manipulator
        """ Loss for traj_opt"""
        loss = 0
        if not hasattr(self, 'loss_fn'):
            self.update_loss_fn(loss_type=loss_type)
        xs = []
        for idx, (shape, tool, *args) in zip(idxes, observations):
            dist = self.get_contact_loss(shape, tool, contact_mask=state_mask)
            loss += dist
            if state_mask is not None:
                xs.append(shape[state_mask, :3])
            else:
                xs.append(shape[:, :3])

        sampled_idx = np.random.choice(xs[0].shape[0], 500, replace=False)
        xs = torch.stack(xs).contiguous()[:, sampled_idx]
        if goal_mask is not None:
            target_x = self.tensor_target_x[goal_mask][sampled_idx].repeat([len(xs), 1, 1])
        else:
            target_x = self.tensor_target_x[sampled_idx].repeat([len(xs), 1, 1])
           
        for i in range(len(xs)):
            if self.loss_type == 'emd':
                loss += self.loss_fn(xs[i], target_x[i])
            else:
                loss += self.loss_fn(target_x[i].unsqueeze(0), xs[i].unsqueeze(0))
        final_v = observations[-1][0]
        loss += vel_loss_weight * torch.sum(torch.mean(final_v[:, 3:6], dim=0) ** 2)
        return loss

    # def her_reward_fn(self, tool_state, particle_state, goal_particle_state):  # Redo for RL
    #     if not hasattr(self, 'loss_fn'):
    #         self.loss_fn = create_emd_loss()
    #     if not isinstance(tool_state, torch.Tensor):
    #         tool_state = torch.FloatTensor(tool_state).to('cuda', non_blocking=True)
    #         particle_state = torch.FloatTensor(particle_state).to('cuda', non_blocking=True)
    #         goal_particle_state = torch.FloatTensor(goal_particle_state).to('cuda', non_blocking=True)
    #
    #     emd = self.loss_fn(particle_state, goal_particle_state)
    #     dists = torch.min(torch.cdist(tool_state[:, :3][None], particle_state[None])[0], dim=1)[0]
    #     contact_loss = (self.contact_loss_mask * dists).sum() * 1e-3
    #     reward = -emd - contact_loss
    #     return reward.item()

    def set_init_emd(self):
        self.init_emd = self.get_curr_emd()
        if isinstance(self.init_emd, torch.Tensor):
            self.init_emd = self.init_emd.item()

    def get_curr_emd(self):
        if self.tensor_target_x is None:
            print("Generating target, skipping setting emd")
            return 0.
        if not hasattr(self, 'emd_loss_fn'):
            self.emd_loss_fn = create_emd_loss()
        curr_x = self.simulator.get_torch_x(0, self.device).contiguous()
        emd = self.emd_loss_fn(curr_x, self.tensor_target_x)
        return emd

    def get_reward_and_info(self):   ## TODO: add component matching option here
        if self.tensor_target_x is None:
            # Generating target
            return 0., {}
        else:
            if not hasattr(self, 'emd_loss_fn'):
                self.emd_loss_fn = create_emd_loss()
            curr_x = self.simulator.get_torch_x(0, self.device).contiguous()
            sampled_idx = np.random.choice(curr_x.shape[0], 500, replace=False)
            curr_x = curr_x[sampled_idx]
            target_x = self.tensor_target_x[sampled_idx]
            emd = self.emd_loss_fn(curr_x, target_x)
            primitive_state = torch.cat([i.get_state_tensor(0)[None, :3] for i in self.primitives], dim=0)
            dists = torch.min(torch.cdist(primitive_state[None], curr_x[None])[0], dim=1)[0]
            contact_loss = (self.contact_loss_mask * dists).sum() * 1e-3
            reward = -emd - contact_loss
            emd = emd.item()
            if not hasattr(self, 'init_emd'):
                normalized_performance = 0.
            elif self.init_emd == 0.:
                normalized_performance = 1.
            else:
                normalized_performance = (self.init_emd - emd) / self.init_emd
            info = {'info_emd': emd,
                    'info_normalized_performance': normalized_performance,
                    'info_contact_loss': contact_loss.item()}
            return reward.item(), info

    def get_geom_loss_fn(self):
        if not hasattr(self, 'emd_loss_fn'):
            self.emd_loss_fn = create_emd_loss()
        return self.emd_loss_fn

    def set_contact_loss_mask(self, mask):
        self.contact_loss_mask = mask

    @ti.kernel
    def _get_obs(self, s: ti.int32, x: ti.ext_arr(), c: ti.ext_arr()):
        for i in range(self.simulator.n_particles[None]):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.simulator.x[s, i][j]
                # x[i, j + self.dim] = self.simulator.v[s, i][j]
        for idx, i in ti.static(enumerate(self.primitives)):
            for j in ti.static(range(i.pos_dim)):
                c[idx, j] = i.position[s][j]
            for j in ti.static(range(i.rotation_dim)):
                c[idx, j + i.pos_dim] = i.rotation[s][j]
            if ti.static(i.state_dim == 8):
                c[idx, 7] = i.gap[s]

    def get_obs(self, s, device):
        f = s * self.simulator.substeps
        x = torch.zeros(size=(self.n_particles, self.dim * 2), device=device)
        x = x[:, :3]  # Only the position
        c = torch.zeros(size=(len(self.primitives), 8), device=device)  # hack for gripper
        self._get_obs(f, x, c)
        if self.return_dist:  # Not really used since this function should only be called when return_dist is True
            self.compute_min_dist(f)
            dists = torch.cat([i.to_torch(device)[:, None] for i in self.dists], 1)
            merged_dists = []
            for i in range(len(self.dists_start_idx) - 1):
                l, r = self.dists_start_idx[i], self.dists_start_idx[i + 1]
                merged_dists.append(torch.sum(dists[:, l:r], dim=1, keepdim=True))
            x = torch.cat((x, *merged_dists), 1)

        outputs = x.clone(), c.clone()
        return outputs

    @ti.kernel
    def compute_min_dist(self, f: ti.int32):
        for j in ti.static(range(self.simulator.n_primitive)):
            for i in range(self.simulator.n_particles[None]):
                v = ti.static(self.dists_start_idx[j])
                if ti.static(not isinstance(self.simulator.primitives[j], Gripper)):
                    self.dists[v][i] = self.simulator.primitives[j].sdf(f, self.simulator.x[f, i])
                else:
                    self.dists[v][i] = self.simulator.primitives[j].sdf_2(f, self.simulator.x[f, i], -1)
                    self.dists[v + 1][i] = self.simulator.primitives[j].sdf_2(f, self.simulator.x[f, i], 1)

    def get_tool_particles(self, s):
        tool_particles = []
        action_primitives = [prim for prim in self.primitives if prim.action_dim > 0]
        for prim in action_primitives:
            if hasattr(prim, 'init_points'):
                tool_particles.append(prim.get_surface_points(s))
            else:
                tool_particles.append(np.zeros((100,3)))
        return np.vstack(tool_particles)