import taichi as ti
import numpy as np
from .utils import length, qrot, qmul, w2quat
from ...config.utils import make_cls_config
from yacs.config import CfgNode as CN
from .utils import inv_trans, qrot
import torch
from plb.engine.primitive.utils import angvel
from scipy.spatial.transform import Rotation as R


@ti.data_oriented
class Primitive:
    # single primitive ..
    state_dim = 7

    def __init__(self, cfg=None, dim=3, max_timesteps=1024, dtype=ti.f32, **kwargs):
        """
        The primitive has the following functions ...
        """
        self.cfg = make_cls_config(self, cfg, **kwargs)
        print('Building primitive')
        print(self.cfg)

        self.dim = dim
        self.max_timesteps = max_timesteps
        self.dtype = dtype

        self.pos_dim = dim
        self.rotation_dim = 4
        self.angular_velocity_dim = 3
        self.max_primitives = 10  # Max number of primitives

        self.friction = ti.field(dtype, shape=())
        self.softness = ti.field(dtype, shape=())
        self.color = ti.Vector.field(3, ti.f32, shape=())  # positon of the primitive
        self.position = ti.Vector.field(3, dtype, needs_grad=True)  # positon of the primitive
        self.rotation = ti.Vector.field(4, dtype, needs_grad=True)  # quaternion for storing rotation

        self.v = ti.Vector.field(3, dtype, needs_grad=True)  # velocity
        self.w = ti.Vector.field(3, dtype, needs_grad=True)  # angular velocity

        ti.root.dense(ti.i, (self.max_timesteps,)).place(self.position, self.position.grad, self.rotation, self.rotation.grad,
                                                         self.v, self.v.grad, self.w, self.w.grad)
        self.xyz_limit = ti.Vector.field(3, dtype, shape=(2,))  # positon of the primitive

        self.action_dim = self.cfg.action.dim
        if self.action_dim > 0:
            self.action_buffer = ti.Vector.field(self.action_dim, dtype, needs_grad=True, shape=(max_timesteps,))
            self.action_scale = ti.Vector.field(self.action_dim, dtype, shape=())
            self.min_dist = ti.field(dtype, shape=(), needs_grad=True)  # record min distance to the point cloud..
            self.dist_norm = ti.field(dtype, shape=(), needs_grad=True)  # record min distance to the point cloud..

        self.inv_trans = inv_trans
        # self.collision_group = ti.Vector.field(self.max_primitives, dtype)
        self.num_rand_points = 100
    @ti.func
    def project(self, f, grid_pos):
        """ Project a point onto the surface of the shape """
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return qrot(self.rotation[f], self._project(f, grid_pos)) + self.position[f]

    @ti.func
    def _project(self, f, grid_pos):
        raise NotImplementedError

    @ti.func
    def _sdf(self, f, grid_pos):
        raise NotImplementedError

    @ti.func
    def _normal(self, f, grid_pos):
        raise NotImplementedError

    @ti.func
    def sdf(self, f, grid_pos):
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return self._sdf(f, grid_pos)

    @ti.func
    def normal(self, f, grid_pos):
        # n2 = self.normal2(f, grid_pos)
        # xx = grid_pos
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return qrot(self.rotation[f], self._normal(f, grid_pos))

    @ti.func
    def collider_v(self, f, grid_pos, dt):
        inv_quat = ti.Vector(
            [self.rotation[f][0], -self.rotation[f][1], -self.rotation[f][2], -self.rotation[f][3]]).normalized()
        relative_pos = qrot(inv_quat, grid_pos - self.position[f])
        new_pos = qrot(self.rotation[f + 1], relative_pos) + self.position[f + 1]
        collider_v = (new_pos - grid_pos) / dt  # TODO: revise
        return collider_v

    @ti.func
    def collide(self, f, grid_pos, v_out, dt, mass):
        dist = self.sdf(f, grid_pos)
        influence = min(ti.exp(-dist * self.softness[None]), 1)
        if (self.softness[None] > 0 and influence > 0.1) or dist <= 0:
            D = self.normal(f, grid_pos)
            collider_v_at_grid = self.collider_v(f, grid_pos, dt)

            input_v = v_out - collider_v_at_grid
            normal_component = input_v.dot(D)

            grid_v_t = input_v - min(normal_component, 0) * D

            grid_v_t_norm = length(grid_v_t)
            grid_v_t_friction = grid_v_t / grid_v_t_norm * max(0, grid_v_t_norm + normal_component * self.friction[None])
            flag = ti.cast(normal_component < 0 and ti.sqrt(grid_v_t.dot(grid_v_t)) > 1e-30, self.dtype)
            grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag)
            v_out = collider_v_at_grid + input_v * (1 - influence) + grid_v_t * influence

            # print(self.position[f],
            # print(grid_pos, collider_v, v_out, dist, self.friction, D)
            # if v_out[1] > 1000:
            # print(input_v, collider_v_at_grid, normal_component, D)

        return v_out

    # @ti.func
    # def collision_projection(self, f, grid_pos):
    #     min_dist = 0.
    #     min_normal = ti.Vector.zero(ti.f32, 3)
    #     for i in range(grid_pos.shape[1]):
    #         dist = self.sdf(f, grid_pos[f, i])
    #         normal = self.normal(f, grid_pos[f, i]).normalized()
    #         if dist < min_dist:
    #             min_dist = dist
    #             min_normal = normal
    #     shift = min_normal * min_dist
    #     self.position[f] += shift

    @ti.func
    def get_collision_idx(self, f, cid, grid_pos):
        min_dist = 0.
        idx = -1
        for i in range(grid_pos.shape[2]):
            dist = self.sdf(f, grid_pos[f, cid, i])
            if ti.atomic_min(min_dist, dist) > dist:
                idx = i
        return idx

    @ti.func
    def collision_projection(self, f, grid_pos):
        dist = self.sdf(f, grid_pos)
        normal = self.normal(f, grid_pos).normalized()
        shift = normal * dist
        self.position[f] += shift

    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        self.position[f + 1] = max(min(self.position[f] + self.v[f], self.xyz_limit[1]), self.xyz_limit[0])
        # rotate in world coordinates about itself.
        self.rotation[f + 1] = qmul(w2quat(self.w[f], self.dtype), self.rotation[f])

    # state set and copy ...
    @ti.func
    def copy_frame(self, source, target):
        self.position[target] = self.position[source]
        self.rotation[target] = self.rotation[source]

    @ti.kernel
    def get_state_kernel(self, f: ti.i32, controller: ti.ext_arr()):
        for j in ti.static(range(3)):
            controller[j] = self.position[f][j]
        for j in ti.static(range(4)):
            controller[j + self.dim] = self.rotation[f][j]

    @ti.kernel
    def set_state_kernel(self, f: ti.i32, controller: ti.ext_arr()):
        for j in ti.static(range(3)):
            self.position[f][j] = controller[j]
        for j in ti.static(range(4)):
            self.rotation[f][j] = controller[j + self.dim]

    def get_state_tensor(self, f):
        out = torch.zeros(7, device='cuda')
        self.get_state_kernel(f, out)
        return out

    def get_state(self, f):
        out = np.zeros((7), dtype=np.float64)
        self.get_state_kernel(f, out)
        return out

    def set_state(self, f, state):
        ss = self.get_state(f)
        ss[:len(state)] = state
        self.set_state_kernel(f, ss)
    
    def get_surface_points(self, f=0, state=None):
        if state is None:
            state = self.get_state(f)
        position, rotation = state[:3], state[3:]
        rot = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
        if len(state) == 8:
            points = self.init_points.copy()
            points[:50, 0] -= state[7] / 2
            points[50:, 0] += state[7] / 2
            transformed_points = rot.apply(points) + position
            return transformed_points

        transformed_points = rot.apply(self.init_points) + position
        return transformed_points
        
    @ti.kernel
    def set_position_kernel(self, pos: ti.ext_arr()):
        for i in ti.static(range(3)):
            self.position[0][i] = pos[i]

    @ti.kernel
    def set_rotation_kernel(self, rot: ti.ext_arr()):
        for i in ti.static(range(4)):
            self.rotation[0][i] = rot[i]

    def set_position(self, pos: np.ndarray):
        self.set_position_kernel(pos)

    def set_rotation(self, rot: np.ndarray):
        self.set_rotation_kernel(rot)

    @property
    def init_state(self):
        return self.cfg.init_pos + self.cfg.init_rot

    def initialize(self):
        cfg = self.cfg
        self.set_state(0, self.init_state)
        self.xyz_limit.from_numpy(np.array([cfg.lower_bound, cfg.upper_bound]))
        self.color[None] = cfg.color
        self.friction[None] = self.cfg.friction  # friction coefficient
        if self.action_dim > 0:
            self.action_scale[None] = cfg.action.scale

    def update_cfg(self, cfg):
        self.cfg = make_cls_config(self, cfg)
        self.set_state(0, self.init_state)

    @ti.kernel
    def set_action_kernel(self, s: ti.i32, action: ti.ext_arr()):
        for j in ti.static(range(self.action_dim)):
            self.action_buffer[s][j] = action[j]

    @ti.complex_kernel
    def no_grad_set_action_kernel(self, s, action):
        self.set_action_kernel(s, action)

    @ti.complex_kernel_grad(no_grad_set_action_kernel)
    def no_grad_set_action_kernel_grad(self, s, action):
        return

    @ti.kernel
    def get_action_grad_kernel(self, s: ti.i32, n: ti.i32, grad: ti.ext_arr()):
        for i in range(0, n):
            for j in ti.static(range(self.action_dim)):
                grad[i, j] = self.action_buffer.grad[s + i][j]

    @ti.kernel
    def set_velocity(self, s: ti.i32, n_substeps: ti.i32):
        # rewrite set velocity for different
        for j in range(s * n_substeps, (s + 1) * n_substeps):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k] * self.action_scale[None][k] / n_substeps
            if ti.static(self.action_dim > 3):
                for k in ti.static(range(3)):
                    self.w[j][k] = self.action_buffer[s][k + 3] * self.action_scale[None][k + 3] / n_substeps

    def set_action(self, s, n_substeps, action):
        # set actions for n_substeps ...
        if self.action_dim > 0:
            self.no_grad_set_action_kernel(s, action)  # HACK: taichi can't compute gradient to this.
            self.set_velocity(s, n_substeps)

    def get_action_grad(self, s, n):
        if self.action_dim > 0:
            grad = np.zeros((n, self.action_dim), dtype=np.float64)
            self.get_action_grad_kernel(s, n, grad)
            return grad
        else:
            return None

    @ti.func
    def get_surface_pos(self, f, rand_pos):
        # rand pos is in primitive space
        return qrot(self.rotation[f], self._project(f, rand_pos)) + self.position[f]

    def inv_action(self, curr_state, target_state, thr=1e-2):
        """ Return the inverse action that moves from curr_state to target_state, until each state dimension's error is less than thr.
        Return None if the condition is already satisfied.
        """
        thr = np.ones_like(curr_state) * thr
        if np.all(np.abs(curr_state - target_state) < thr):
            return None
        action = np.zeros(self.action_dim)

        # Translation
        dir = (target_state[:3] - curr_state[:3])  # Translation
        dir *= 40
        if np.any(np.abs(curr_state[:3] - target_state[:3]) > thr[:3]):
            action[:3] = dir
        # Rotation
        vel = angvel(curr_state[3:7], target_state[3:7])
        if vel[2] ==0.: # TODO Hack here. Maybe there is a bug in angvel. Will come back to this later.
            vel[2] = -vel[1] * 0.3
            vel[1] = 0.
        vel = vel * 100.
        if np.any(np.abs(curr_state[3:7] - target_state[3:7]) > thr[3:7]):
            action[3:6] = vel
        return action

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.shape = ''
        cfg.init_pos = (0.3, 0.3, 0.3)  # default color
        cfg.init_rot = (1., 0., 0., 0.)  # default color
        cfg.color = (0.3, 0.3, 0.3)  # default color
        cfg.lower_bound = (0., 0., 0.)  # default color
        cfg.upper_bound = (1., 1., 1.)  # default color
        cfg.friction = 0.9  # default color
        # cfg.variations = None  # TODO: not support now
        cfg.collision_group = [0., 0., 0.]

        action = cfg.action = CN()
        action.dim = 0  # in this case it can't move ...
        action.scale = ()
        return cfg
