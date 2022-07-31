import taichi as ti
import torch
import numpy as np
import yaml
from .primive_base import Primitive, angvel
from .utils import inverse_rot, quat_equal, w2quat, qrot, qmul, rotate, relative_rot_q, relative_rot_w, trans, q_mul, inverse_rot, relative_rot
from yacs.config import CfgNode as CN
from scipy.spatial.transform import Rotation as R
from .utils import qrot, qmul, w2quat
import copy


@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-14)


@ti.func
def normalize(n):
    return n / length(n)


class Sphere(Primitive):
    def __init__(self, **kwargs):
        super(Sphere, self).__init__(**kwargs)
        self.radius = self.cfg.radius

    @ti.func
    def sdf(self, f, grid_pos):
        return length(grid_pos - self.position[f]) - self.radius

    @ti.func
    def normal(self, f, grid_pos):
        return normalize(grid_pos - self.position[f])

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.radius = 1.
        return cfg


class Capsule(Primitive):
    def __init__(self, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.h = self.cfg.h
        self.r = self.cfg.r

    def initialize(self):
        super(Capsule, self).initialize()
        self.generate_init_points()
        # self.sample_points_on_surface(-1)

    @ti.func
    def _sdf(self, f, grid_pos):
        p2 = grid_pos
        p2[1] += self.h / 2
        p2[1] -= min(max(p2[1], 0.0), self.h)
        return length(p2) - self.r

    @ti.func
    def _normal(self, f, grid_pos):
        p2 = grid_pos
        p2[1] += self.h / 2
        p2[1] -= min(max(p2[1], 0.0), self.h)
        return normalize(p2)

    def generate_init_points(self):
        n = int(np.sqrt(self.num_rand_points))
        points = []
        linsp1 = np.linspace(-self.h/2, self.h/2, n)
        linsp2 = np.linspace(0, 2*np.pi, n)
        for k in range(n):
            for l in range(n):
                points.append(np.array([self.r*np.cos(linsp2[k]), linsp1[l], self.r*np.sin(linsp2[k])]))
        self.init_points = np.vstack(points)
    # @ti.kernel
    # def sample_points_on_surface(self, f:ti.i32):
    #     n = ti.static(int(ti.sqrt(self.num_rand_points)))
    #     for k in range(n):
    #         for l in range(n):
    #             pos = ti.Vector([self.r*ti.cos(self.linsp2[k]), self.linsp1[l], self.r*ti.sin(self.linsp2[k])])
    #             pos = trans(pos, self.position[f+1], self.rotation[f+1])
    #             self.rand_points[f+1, k*n+l] = pos

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.06
        cfg.r = 0.03
        return cfg

    # @ti.func
    # def copy_frame(self, source, target):
    #     self.position[target] = self.position[source]
    #     self.rotation[target] = self.rotation[source]
    #     for i in ti.static(range(self.num_rand_points)):
    #         self.rand_points[target, i] = self.rand_points[source, i]


class RollingPin(Capsule):
    # rollingpin's capsule...
    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        vel = self.v[f]
        dw = vel[0]  # rotate about object y
        dth = vel[1]  # rotate about the world w
        dy = vel[2]  # decrease in y coord...
        y_dir = qrot(self.rotation[f], ti.Vector([0., -1., 0.]))
        x_dir = ti.Vector([0., 1., 0.]).cross(y_dir) * dw * 0.03  # move toward x, R=0.03 is hand crafted...
        x_dir[1] = dy  # direction
        self.rotation[f + 1] = qmul(
            w2quat(ti.Vector([0., -dth, 0.]), self.dtype),
            qmul(self.rotation[f], w2quat(ti.Vector([0., dw, 0.]), self.dtype))
        )
        # print(self.rotation[f+1], self.rotation[f+1].dot(self.rotation[f+1]))
        self.position[f + 1] = max(min(self.position[f] + x_dir, self.xyz_limit[1]), self.xyz_limit[0])


class RollingPinExt(Capsule):
    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        vel = self.v[f]
        w = self.w[f]
        dw = vel[0]  # rotate about object y
        dth = vel[1]  # rotate about the world w
        dy = vel[2]  # decrease in y coord...
        y_dir = qrot(self.rotation[f], ti.Vector([0., -1., 0.]))
        x_dir = ti.Vector([0., 1., 0.]).cross(y_dir) * (dw * 0.03 + w[0])
        x_dir[1] = dy  # direction
        self.rotation[f + 1] = qmul(
            w2quat(ti.Vector([0., -dth, 0.]), self.dtype),
            qmul(self.rotation[f], w2quat(ti.Vector([0., dw, 0.]), self.dtype))
        )
        # print(self.rotation[f+1], self.rotation[f+1].dot(self.rotation[f+1]))
        self.position[f + 1] = max(min(self.position[f] + x_dir, self.xyz_limit[1]), self.xyz_limit[0])

    def inv_action(self, curr_state, target_state, thr=1e-2):
        """ Return the inverse action that moves from curr_state to target_state, until each state dimension's error is less than thr.
        Return None if the condition is already satisfied.
        """
        thr = np.ones_like(curr_state) * thr
        if np.all(np.abs(curr_state - target_state)[:3] < thr[:3]): # Only match the translation
            return None
        action, ret_action = np.zeros(self.action_dim), np.zeros(self.action_dim) # Rolling pin uses a different action space
        # Translation
        dir = (target_state[:3] - curr_state[:3])  # Translation
        dir *= 40
        if np.any(np.abs(curr_state[:3] - target_state[:3]) > thr[:3]):
            action[:3] = dir
        ret_action[2] = action[1] * 50
        ret_action[0] = - action[0] * 0.2
        return ret_action

    # From Carl, rotate the roller when resetting
    # def inv_action(self, curr_state, target_state, thr=5e-3):
    #     thr = np.ones_like(curr_state) * thr
    #     if np.all(np.abs(curr_state[:3] - target_state[:3]) < thr[:3]) and quat_equal(curr_state[3:7], target_state[3:7], thr[0]):
    #         return None
    #     action = np.zeros(self.action_dim)
    #
    #     # First figure out translation. Check whether following roll_dir can get us to the desired location
    #     dir = (target_state[:3] - curr_state[:3])
    #     dy = dir[1]
    #     dir = np.array([dir[0], 0, dir[2]])
    #     y_dir_curr = R.from_quat([curr_state[4], curr_state[5], curr_state[6], curr_state[3]]).apply([0, -1, 0])
    #     y_dir_targ = R.from_quat([target_state[4], target_state[5], target_state[6], target_state[3]]).apply([0, -1, 0])
    #     roll_dir = np.cross(np.array([0,1,0]), y_dir_curr)
    #     # print("roll_dir:", roll_dir)
    #     # print("translation_dir:", dir)
    #     assert roll_dir[1] == 0 # rolling pin has to be flat
    #
    #     # If no need for translation: simply rotate and done.
    #     if np.all(np.abs(curr_state[:3] - target_state[:3]) < thr[:3]):
    #         rad1 = 0
    #         dl = 0
    #         dy = 0
    #         # Rotation around the global Y-axis
    #         rad2 = -np.arccos(np.dot(y_dir_curr, y_dir_targ)) * np.sign(np.cross(y_dir_curr, y_dir_targ)[1])
    #     elif np.abs(dy) > thr[0]:
    #         rad1 = 0
    #         dl = 0
    #         rad2 = 0
    #     # direction is right, rolling
    #     elif math.isclose(np.abs(dir.dot(roll_dir) / np.linalg.norm(dir) / np.linalg.norm(roll_dir)), 1, abs_tol=1e-3):
    #         # Rotation around the rolling pin itself
    #         rel_rot = relative_rot(curr_state[3:7], target_state[3:7])
    #         rad1 = rel_rot.as_euler('XYZ')[2] + rel_rot.as_euler('XYZ')[0]
    #         # dl is the rolling length (signed)
    #         dl = np.linalg.norm(dir) * np.sign(dir.dot(roll_dir))
    #         dl -= rad1 * 0.03
    #         rad2 = 0
    #     # turning rolling pin to the correct rolling direction
    #     else:
    #         y_dir_targ = np.cross(np.array([0, -1, 0]),  dir / np.linalg.norm(dir))
    #         y_dir_targ = y_dir_targ * np.sign(y_dir_targ.dot(y_dir_curr))
    #         rad1 = 0
    #         dl = 0
    #         # Rotation around the global Y-axis
    #         rad2 = -np.arccos(np.dot(y_dir_curr, y_dir_targ)) * np.sign(np.cross(y_dir_curr, y_dir_targ)[1])
    #
    #     action[0] = rad1 * 1. / 0.7
    #     action[1] = rad2 * 10
    #     action[2] = dy * 40
    #     action[3] = dl * 40
    #     print(curr_state, target_state)
    #     print(action)
    #     return action

# From Carl. Directly reset SE(3) pose
# class RollingPinExt2(Capsule):
#     @ti.kernel
#     def forward_kinematics(self, f: ti.i32):
#         v, w = self.v[f], self.w[f]
#         self.position[f + 1] = max(min(self.position[f] + v, self.xyz_limit[1]), self.xyz_limit[0])
#         q = w2quat(w, self.dtype)
#         self.rotation[f + 1] = qmul(q, self.rotation[f])
class Chopsticks(Capsule):
    state_dim = 8

    def __init__(self, **kwargs):
        super(Chopsticks, self).__init__(**kwargs)
        self.gap = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.gap_vel = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.h = self.cfg.h
        self.r = self.cfg.r
        self.minimal_gap = self.cfg.minimal_gap
        assert self.action_dim == 7  # 3 linear, 3 angle, 1 for grasp ..

    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        self.gap[f + 1] = max(self.gap[f] - self.gap_vel[f], self.minimal_gap)
        self.position[f + 1] = max(min(self.position[f] + self.v[f], self.xyz_limit[1]), self.xyz_limit[0])
        self.rotation[f + 1] = qmul(self.rotation[f], w2quat(self.w[f], self.dtype))
        # print(self.rotation[f+1])

    @ti.kernel
    def set_velocity(self, s: ti.i32, n_substeps: ti.i32):
        # rewrite set velocity for different
        for j in range(s * n_substeps, (s + 1) * n_substeps):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k] * self.action_scale[None][k] / n_substeps
            for k in ti.static(range(3)):
                self.w[j][k] = self.action_buffer[s][k + 3] * self.action_scale[None][k + 3] / n_substeps
            self.gap_vel[j] = self.action_buffer[s][6] * self.action_scale[None][6] / n_substeps

    @ti.func
    def _sdf(self, f, grid_pos):
        delta = ti.Vector([self.gap[f] / 2, 0., 0.])
        p = grid_pos - ti.Vector([0., -self.h / 2, 0.])
        a = super(Chopsticks, self)._sdf(f, p - delta)  # grid_pos - (mid + delta)
        b = super(Chopsticks, self)._sdf(f, p + delta)  # grid_pos - (mid - delta)
        return ti.min(a, b)

    @ti.func
    def _normal(self, f, grid_pos):
        delta = ti.Vector([self.gap[f] / 2, 0., 0.])
        p = grid_pos - ti.Vector([0., -self.h / 2, 0.])
        a = super(Chopsticks, self)._sdf(f, p - delta)  # grid_pos - (mid + delta)
        b = super(Chopsticks, self)._sdf(f, p + delta)  # grid_pos - (mid - delta)
        a_n = super(Chopsticks, self)._normal(f, p - delta)  # grid_pos - (mid + delta)
        b_n = super(Chopsticks, self)._normal(f, p + delta)  # grid_pos - (mid + delta)
        m = ti.cast(a <= b, self.dtype)
        return m * a_n + (1 - m) * b_n

    @property
    def init_state(self):
        return self.cfg.init_pos + self.cfg.init_rot + (self.cfg.init_gap,)

    def get_state(self, f):
        return np.append(super(Chopsticks, self).get_state(f), self.gap[f])

    @ti.func
    def copy_frame(self, source, target):
        super(Chopsticks, self).copy_frame(source, target)
        self.gap[target] = self.gap[source]

    def set_state(self, f, state):
        assert len(state) == 8
        super(Chopsticks, self).set_state(f, state[:7])
        self.gap[f] = state[7]

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.06
        cfg.r = 0.03
        cfg.minimal_gap = 0.06
        cfg.init_gap = 0.06
        return cfg


class Cylinder(Primitive):
    def __init__(self, **kwargs):
        super(Cylinder, self).__init__(**kwargs)
        self.h = self.cfg.h
        self.r = self.cfg.r

    @ti.func
    def _sdf(self, f, grid_pos):
        # convert it to a 2D box .. and then call the sdf of the 2d box..
        d = ti.abs(ti.Vector([length(ti.Vector([grid_pos[0], grid_pos[2]])), grid_pos[1]])) - ti.Vector([self.h, self.r])
        return min(max(d[0], d[1]), 0.0) + length(max(d, 0.0))  # if max(d, 0) < 0 or if max(d, 0) > 0

    @ti.func
    def _normal(self, f, grid_pos):
        p = ti.Vector([grid_pos[0], grid_pos[2]])
        l = length(p)
        d = ti.Vector([l, ti.abs(grid_pos[1])]) - ti.Vector([self.h, self.r])

        # if max(d) > 0, normal direction is just d
        # other wise it's 1 if d[1]>d[0] else -d0
        # return min(max(d[0], d[1]), 0.0) + length(max(d, 0.0))
        f = ti.cast(d[0] > d[1], self.dtype)
        n2 = max(d, 0.0) + ti.cast(max(d[0], d[1]) <= 0., self.dtype) * ti.Vector([f, 1 - f])  # normal should be always outside ..
        n2_ = n2 / length(n2)
        p2 = p / l
        n3 = ti.Vector([p2[0] * n2_[0], n2_[1] * (ti.cast(grid_pos[1] >= 0, self.dtype) * 2 - 1), p2[1] * n2_[0]])
        return normalize(n3)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.2
        cfg.r = 0.1
        return cfg


class Torus(Primitive):
    def __init__(self, **kwargs):
        super(Torus, self).__init__(**kwargs)
        self.tx = self.cfg.tx
        self.ty = self.cfg.ty

    @ti.func
    def _sdf(self, f, grid_pos):
        q = ti.Vector([length(ti.Vector([grid_pos[0], grid_pos[2]])) - self.tx, grid_pos[1]])
        return length(q) - self.ty

    @ti.func
    def _normal(self, f, grid_pos):
        x = ti.Vector([grid_pos[0], grid_pos[2]])
        l = length(x)
        q = ti.Vector([length(x) - self.tx, grid_pos[1]])

        n2 = q / length(q)
        x2 = x / l
        n3 = ti.Vector([x2[0] * n2[0], n2[1], x2[1] * n2[0]])
        return normalize(n3)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.tx = 0.2
        cfg.ty = 0.1
        return cfg


class Box(Primitive):
    def __init__(self, **kwargs):
        super(Box, self).__init__(**kwargs)
        self.size = ti.Vector.field(3, self.dtype, shape=())

    def initialize(self, cached_state_path=''):
        super(Box, self).initialize()
        self.generate_init_points(cached_state_path)
        self.size[None] = self.cfg.size

    @ti.func
    def _project(self, f, grid_pos):
        """ Project a point onto the surface of the shape """
        return max(min(grid_pos, self.size), -self.size)

    @ti.func
    def _sdf(self, f, grid_pos):
        # p: vec3,b: vec3
        q = ti.abs(grid_pos) - self.size[None]
        out = length(max(q, 0.0))
        out += min(max(q[0], max(q[1], q[2])), 0.0)
        return out

    @ti.func
    def _normal(self, f, grid_pos):
        # TODO: replace it with analytical normal later..
        d = ti.cast(1e-4, ti.float32)
        n = ti.Vector.zero(self.dtype, self.dim)
        for i in ti.static(range(self.dim)):
            inc = grid_pos
            dec = grid_pos
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self._sdf(f, inc) - self._sdf(f, dec))
        return n / length(n)

    def generate_init_points(self, cached_state_path):
        # only generate points on the top part for now
        n = int(np.sqrt(self.num_rand_points))
        size = self.size[None]
        points = []

        if 'liftspread' in cached_state_path:
            linsp1 = np.linspace(-size[0], size[0], n)
            linsp2 = np.linspace(-size[1], size[1], n)
            for k in range(n):
                for l in range(n):
                    points.append(np.array([linsp1[k], linsp2[l], -size[2]/2]))
        elif 'cutrearrangespread' in cached_state_path:
            linsp1 = np.linspace(-size[1], size[1], n)
            linsp2 = np.linspace(-size[2], size[2], n)
            for k in range(n):
                for l in range(n):
                    points.append(np.array([-size[0]/2, linsp1[k], linsp2[l]]))
        else:
            linsp1 = np.linspace(-size[0], size[0], n)
            linsp2 = np.linspace(-size[1], size[1], n)
            for k in range(n):
                for l in range(n):
                    points.append(np.array([linsp1[k], linsp2[l], -size[2]/2]))
        self.init_points = np.vstack(points)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.size = (0.1, 0.1, 0.1)
        return cfg


class Gripper(Box):
    state_dim = 8

    def __init__(self, **kwargs):
        super(Gripper, self).__init__(**kwargs)
        self.gap = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.gap_vel = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.size = ti.Vector.field(3, self.dtype, shape=())
        self.size[None] = self.cfg.size
        self.minimal_gap = self.cfg.minimal_gap
        self.maximal_gap = self.cfg.maximal_gap

    def generate_init_points(self, cached_state_path=''):
        # only generate points on the top part for now
        n = int(np.sqrt(self.num_rand_points))
        size = self.size[None]
        points = []
        linsp1 = np.linspace(-size[1], size[1], n)
        linsp2 = np.linspace(-size[2], size[2], n)
        for k in range(n):
            for l in range(n):
                # print(self.gap[None])
                points.append(np.array([size[0], linsp1[l], linsp2[k]]))
        for k in range(n):
            for l in range(n):
                points.append(np.array([-size[0], linsp1[l], linsp2[k]]))
        self.init_points = np.vstack(points[::2])

    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        self.gap[f + 1] = min(max(self.gap[f] - self.gap_vel[f], self.minimal_gap), self.maximal_gap)
        self.position[f + 1] = max(min(self.position[f] + self.v[f], self.xyz_limit[1]), self.xyz_limit[0])
        self.rotation[f + 1] = qmul(self.rotation[f], w2quat(self.w[f], self.dtype))

    @ti.kernel
    def set_velocity(self, s: ti.i32, n_substeps: ti.i32):
        for j in range(s * n_substeps, (s + 1) * n_substeps):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k] * self.action_scale[None][k] / n_substeps
            for k in ti.static(range(3)):
                self.w[j][k] = self.action_buffer[s][k + 3] * self.action_scale[None][k + 3] / n_substeps
            self.gap_vel[j] = self.action_buffer[s][6] * self.action_scale[None][6] / n_substeps

    @ti.func
    def get_pos(self, f, flag):
        return self.position[f] + qrot(self.rotation[f], ti.Vector([self.gap[f] / 2 * flag, 0., 0.]))

    @ti.func
    def sdf_2(self, f, grid_pos, flag):
        grid_pos = self.inv_trans(grid_pos, self.get_pos(f, flag), self.rotation[f])
        return Box._sdf(self, f, grid_pos)

    @ti.func
    def normal_2(self, f, grid_pos, flag):
        grid_pos = self.inv_trans(grid_pos, self.get_pos(f, flag), self.rotation[f])
        return qrot(self.rotation[f], Box._normal(self, f, grid_pos))

    @ti.func
    def sdf(self, f, grid_pos):
        return ti.min(self.sdf_2(f, grid_pos, -1), self.sdf_2(f, grid_pos, 1))

    @ti.func
    def normal(self, f, grid_pos):
        a = self.sdf_2(f, grid_pos, -1)
        b = self.sdf_2(f, grid_pos, 1)
        a_n = self.normal_2(f, grid_pos, -1)
        b_n = self.normal_2(f, grid_pos, 1)
        m = ti.cast(a <= b, self.dtype)
        return m * a_n + (1 - m) * b_n

    @ti.func
    def collider_v(self, f, grid_pos, dt, flag):
        inv_quat = ti.Vector(
            [self.rotation[f][0], -self.rotation[f][1], -self.rotation[f][2], -self.rotation[f][3]]).normalized()
        relative_pos = qrot(inv_quat, grid_pos - self.get_pos(f, flag))
        new_pos = qrot(self.rotation[f + 1], relative_pos) + self.get_pos(f + 1, flag)
        collider_v = (new_pos - grid_pos) / dt  # TODO: revise
        return collider_v, relative_pos

    @ti.func
    def collide(self, f, grid_pos, v_out, dt, mass):
        v_out1 = self.collide2(f, grid_pos, v_out, dt, mass, -1)
        v_out2 = self.collide2(f, grid_pos, v_out1, dt, mass, 1)
        return v_out2

    @ti.func
    def collide2(self, f, grid_pos, v_out, dt, mass, flag):
        dist = self.sdf_2(f, grid_pos, flag)
        influence = min(ti.exp(-dist * self.softness[None]), 1)
        if (self.softness[None] > 0 and influence > 0.1) or dist <= 0:
            D = self.normal_2(f, grid_pos, flag)

            v_in = v_out

            collider_v_at_grid, relative_pos = self.collider_v(f, grid_pos, dt, flag)

            input_v = v_out - collider_v_at_grid
            normal_component = input_v.dot(D)

            grid_v_t = input_v - min(normal_component, 0) * D

            grid_v_t_norm = length(grid_v_t)
            grid_v_t_friction = grid_v_t / grid_v_t_norm * max(0,
                                                               grid_v_t_norm + normal_component * self.friction[None])
            flag = ti.cast(normal_component < 0 and ti.sqrt(grid_v_t.dot(grid_v_t)) > 1e-30, self.dtype)
            grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag)
            v_out = collider_v_at_grid + input_v * (1 - influence) + grid_v_t * influence

        return v_out

    @property
    def init_state(self):
        return self.cfg.init_pos + self.cfg.init_rot + (self.cfg.init_gap,)

    def get_state(self, f):
        return np.append(super(Gripper, self).get_state(f), self.gap[f])

    @ti.func
    def copy_frame(self, source, target):
        super(Gripper, self).copy_frame(source, target)
        self.gap[target] = self.gap[source]

    def set_state(self, f, state):
        assert len(state) == 8
        super(Gripper, self).set_state(f, state[:7])
        self.gap[f] = state[7]

    def inv_action(self, curr_state, target_state, thr=1e-2):
        action = super(Gripper, self).inv_action(curr_state, target_state, thr)
        if action is None:
            return None
        else:
            action[-1] = (- target_state[-1] + curr_state[-1]) * 50 # Somehow positive action means close gripper
        return action

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        # cfg.h = 0.06
        # cfg.r = 0.03
        cfg.size = (0.03, 0.06, 0.03)
        cfg.minimal_gap = 0.06
        cfg.maximal_gap = 1.
        cfg.init_gap = 0.06
        cfg.round = 0
        return cfg


class Gripper2(Capsule):
    state_dim = 8

    def __init__(self, **kwargs):
        super(Gripper2, self).__init__(**kwargs)
        self.gap = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.gap_vel = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        #self.size = ti.Vector.field(3, self.dtype, shape=())
        #self.size[None] = self.cfg.size
        self.minimal_gap = self.cfg.minimal_gap
        self.maximal_gap = self.cfg.maximal_gap

    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        self.gap[f + 1] = min(max(self.gap[f] - self.gap_vel[f], self.minimal_gap), self.maximal_gap)
        self.position[f + 1] = max(min(self.position[f] + self.v[f], self.xyz_limit[1]), self.xyz_limit[0])
        self.rotation[f + 1] = qmul(self.rotation[f], w2quat(self.w[f], self.dtype))

    @ti.kernel
    def set_velocity(self, s: ti.i32, n_substeps: ti.i32):
        for j in range(s * n_substeps, (s + 1) * n_substeps):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k] * self.action_scale[None][k] / n_substeps
            for k in ti.static(range(3)):
                self.w[j][k] = self.action_buffer[s][k + 3] * self.action_scale[None][k + 3] / n_substeps
            self.gap_vel[j] = self.action_buffer[s][6] * self.action_scale[None][6] / n_substeps

    @ti.func
    def get_pos(self, f, flag):
        return self.position[f] + qrot(self.rotation[f], ti.Vector([self.gap[f] / 2 * flag, 0., 0.]))

    @ti.func
    def sdf_2(self, f, grid_pos, flag):
        grid_pos = self.inv_trans(grid_pos, self.get_pos(f, flag), self.rotation[f])
        return Capsule._sdf(self, f, grid_pos)

    @ti.func
    def normal_2(self, f, grid_pos, flag):
        grid_pos = self.inv_trans(grid_pos, self.get_pos(f, flag), self.rotation[f])
        return qrot(self.rotation[f], Capsule._normal(self, f, grid_pos))

    @ti.func
    def sdf(self, f, grid_pos):
        return ti.min(self.sdf_2(f, grid_pos, -1), self.sdf_2(f, grid_pos, 1))

    @ti.func
    def normal(self, f, grid_pos):
        a = self.sdf_2(f, grid_pos, -1)
        b = self.sdf_2(f, grid_pos, 1)
        a_n = self.normal_2(f, grid_pos, -1)
        b_n = self.normal_2(f, grid_pos, 1)
        m = ti.cast(a <= b, self.dtype)
        return m * a_n + (1 - m) * b_n

    @ti.func
    def collider_v(self, f, grid_pos, dt, flag):
        inv_quat = ti.Vector(
            [self.rotation[f][0], -self.rotation[f][1], -self.rotation[f][2], -self.rotation[f][3]]).normalized()
        relative_pos = qrot(inv_quat, grid_pos - self.get_pos(f, flag))
        new_pos = qrot(self.rotation[f + 1], relative_pos) + self.get_pos(f + 1, flag)
        collider_v = (new_pos - grid_pos) / dt  # TODO: revise
        return collider_v, relative_pos

    @ti.func
    def collide(self, f, grid_pos, v_out, dt, mass):
        v_out1 = self.collide2(f, grid_pos, v_out, dt, mass, -1)
        v_out2 = self.collide2(f, grid_pos, v_out1, dt, mass, 1)
        return v_out2

    @ti.func
    def collide2(self, f, grid_pos, v_out, dt, mass, flag):
        dist = self.sdf_2(f, grid_pos, flag)
        influence = min(ti.exp(-dist * self.softness[None]), 1)
        if (self.softness[None] > 0 and influence > 0.1) or dist <= 0:
            D = self.normal_2(f, grid_pos, flag)

            v_in = v_out

            collider_v_at_grid, relative_pos = self.collider_v(f, grid_pos, dt, flag)

            input_v = v_out - collider_v_at_grid
            normal_component = input_v.dot(D)

            grid_v_t = input_v - min(normal_component, 0) * D

            grid_v_t_norm = length(grid_v_t)
            grid_v_t_friction = grid_v_t / grid_v_t_norm * max(0,
                                                               grid_v_t_norm + normal_component * self.friction[None])
            flag = ti.cast(normal_component < 0 and ti.sqrt(grid_v_t.dot(grid_v_t)) > 1e-30, self.dtype)
            grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag)
            v_out = collider_v_at_grid + input_v * (1 - influence) + grid_v_t * influence

        return v_out

    @property
    def init_state(self):
        return self.cfg.init_pos + self.cfg.init_rot + (self.cfg.init_gap,)

    def get_state(self, f):
        return np.append(super(Gripper2, self).get_state(f), self.gap[f])

    @ti.func
    def copy_frame(self, source, target):
        super(Gripper2, self).copy_frame(source, target)
        self.gap[target] = self.gap[source]

    def set_state(self, f, state):
        assert len(state) == 8
        super(Gripper2, self).set_state(f, state[:7])
        self.gap[f] = state[7]

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.06
        cfg.r = 0.015
        #cfg.size = (0.03, 0.06, 0.03)
        cfg.minimal_gap = 0.06
        cfg.maximal_gap = 1.
        cfg.init_gap = 0.06
        cfg.round = 0
        return cfg


class Prism(Primitive):
    def __init__(self, **kwargs):
        super(Prism, self).__init__(**kwargs)
        self.h = ti.Vector.field(2, self.dtype, shape=())
        self.prot = ti.Vector.field(4, self.dtype, shape=())

    def initialize(self):
        super(Prism, self).initialize()
        self.h[None] = self.cfg.h
        self.prot[None] = self.cfg.prot

    @ti.func
    def _sdf(self, f, grid_pos):
        inv_quat = ti.Vector([self.prot[None][0], -self.prot[None][1],
                              -self.prot[None][2], -self.prot[None][3]]).normalized()
        grid_pos = qrot(inv_quat, grid_pos)
        q = ti.abs(grid_pos)
        return max(q[2] - self.h[None][1],
                   max(q[0] * 0.866025 + grid_pos[1] * 0.5, -grid_pos[1]) - self.h[None][0] * 0.5)

    @ti.func
    def _normal(self, f, grid_pos):
        # TODO: replace it with analytical normal later..
        d = ti.cast(1e-4, ti.float32)
        n = ti.Vector.zero(self.dtype, self.dim)
        for i in ti.static(range(self.dim)):
            inc = grid_pos
            dec = grid_pos
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self._sdf(f, inc) - self._sdf(f, dec))
        return n / length(n)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = (0.1, 0.1)
        cfg.prot = (1.0, 0.0, 0.0, 0.0)
        return cfg

class Knife(Primitive): # TODO: write generate_init_points for knife
    def __init__(self, **kwargs):
        super(Knife, self).__init__(**kwargs)
        self.prism = Prism(h=self.cfg.h, prot=self.cfg.prot)
        self.box = Box(size=self.cfg.size)

    def initialize(self):
        super(Knife, self).initialize()
        self.prism.initialize()
        self.box.initialize()
        self.generate_init_points()

    def generate_init_points(self):
        h = self.prism.h[None]
        size = self.box.size[None]
        n = int(np.sqrt(self.num_rand_points))
        prism_points1, prism_points2 = [], []
        offset_y = h[0]
        linsp1 = np.linspace(0, size[0], n)
        linsp2 = np.linspace(-size[2], size[2], n)
        for k in range(n):
            for l in range(n):
                prism_points1.append(np.array([-linsp1[l], linsp1[l]*np.sqrt(3)-offset_y,  linsp2[k]]))
                prism_points2.append(np.array([ linsp1[l], linsp1[l]*np.sqrt(3)-offset_y,  linsp2[k]]))

        prism_points1 = prism_points1[::2]
        prism_points2 = prism_points2[::2]
        self.init_points = np.vstack([prism_points1, prism_points2])

    @ti.func
    def _sdf(self, f, grid_pos):
        q1 = self.prism._sdf(f, grid_pos)
        q2 = self.box._sdf(f, grid_pos)
        return max(q1, q2)

    #@ti.func
    #def _sdf_helper(self, f, grid_pos, h):
    #    q = ti.abs(grid_pos)
    #    return max(q[2] - h[None][1], max(q[0] * 0.866025 + grid_pos[1] * 0.5, -grid_pos[1]) - h[None][0] * 0.5)

    def inv_action(self, curr_state, target_state, thr=1e-2):
        thr = np.ones_like(curr_state) * thr
        if np.all(np.abs(curr_state - target_state) < thr):
            return None
        action = np.zeros(self.action_dim)

        # Translation
        # print(f'Reset state: {target_state}, curr_state {curr_state}')
        dir = (target_state[:3] - curr_state[:3])  # Translation
        dir *= np.array([10., 40., 10.])
        if abs(target_state[1] - curr_state[1]) > 0.01:  # First lift the knife
            dir[0] = 0.
        if np.any(np.abs(curr_state[:3] - target_state[:3]) > thr[:3]):
            action[:3] = dir
        return action

    @ti.func
    def _normal(self, f, grid_pos):
        # TODO: replace it with analytical normal later..
        d = ti.cast(1e-4, ti.float32)
        n = ti.Vector.zero(self.dtype, self.dim)
        for i in ti.static(range(self.dim)):
            inc = grid_pos
            dec = grid_pos
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self._sdf(f, inc) - self._sdf(f, dec))
        return n / length(n)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = (0.1, 0.1)
        cfg.size = (0.1, 0.1, 0.1)
        cfg.prot = (1.0, 0.0, 0.0, 0.0)
        return cfg




class Primitives:
    def __init__(self, cfgs, max_timesteps=1024):
        outs = []
        self.primitives = []
        for i in cfgs:
            if isinstance(i, CN):
                cfg = i
            else:
                cfg = CN(new_allowed=True)
                cfg = cfg._load_cfg_from_yaml_str(yaml.safe_dump(i))
            outs.append(cfg)

        self.action_dims = [0]
        print("pimirives: num primitive:", len(outs))
        for i in outs:
            primitive = eval(i.shape)(cfg=i, max_timesteps=max_timesteps)
            self.primitives.append(primitive)
            self.action_dims.append(self.action_dims[-1] + primitive.action_dim)
        self.n = len(self.primitives)

    def update_cfgs(self, cfgs):
        outs = []
        for i in cfgs:
            if isinstance(i, CN):
                cfg = i
            else:
                cfg = CN(new_allowed=True)
                cfg = cfg._load_cfg_from_yaml_str(yaml.safe_dump(i))
            outs.append(cfg)
        for i, cfg in enumerate(outs):
            self.primitives[i].update_cfg(cfg)

    @property
    def action_dim(self):
        return self.action_dims[-1]

    @property
    def state_dim(self):
        return sum([i.state_dim for i in self.primitives])
    @property
    def state_dims(self):
        return [i.state_dim for i in self.primitives]

    def set_action(self, s, n_substeps, action):
        action = np.asarray(action).reshape(-1).clip(-1, 1)
        assert len(action) == self.action_dims[-1]
        for i in range(self.n):
            self.primitives[i].set_action(s, n_substeps, action[self.action_dims[i]:self.action_dims[i + 1]])

    def get_grad(self, n):
        grads = []
        for i in range(self.n):
            grad = self.primitives[i].get_action_grad(0, n)
            if grad is not None:
                grads.append(grad)
        return np.concatenate(grads, axis=1)

    def set_softness(self, softness=666.):
        for i in self.primitives:
            i.softness[None] = softness

    def get_softness(self):
        return self.primitives[0].softness[None]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[0]
        return self.primitives[item]

    def __len__(self):
        return len(self.primitives)

    def initialize(self, cached_state_path=''):
        for i in self.primitives:
            if i.__class__ is Box:
                i.initialize(cached_state_path=cached_state_path)
            else:
                i.initialize()