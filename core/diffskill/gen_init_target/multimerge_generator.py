import numpy as np
from core.gen_init_target.state_generator import StateGenerator
from functools import partial
from pyquaternion import Quaternion
from math import atan2


def set_rot(env, q):
    state = env.taichi_env.primitives[0].get_state(0)
    env.taichi_env.primitives[0].set_state(0, [state[0], state[1], state[2], q.w, q.x, q.y, q.z, 0.4])


def rotate(q, axis, angle):
    """Rotate q around axis by angle"""
    q = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    rot = Quaternion(axis=axis, angle=angle)
    return rot * q


def gen_rand(l, r):
    return l + np.random.rand() * (r - l)


class MultimergeGenerator(StateGenerator):
    def __init__(self, *args, **kwargs):
        super(MultimergeGenerator, self).__init__(*args, **kwargs)
        if self.mode =='train':
            self.range_x = (0.35, 0.55)  # x-location
            self.range_y = (0.35, 0.55)  # y-location
        else:
            self.range_x = (0.32, 0.58)  # x-location
            self.range_y = (0.32, 0.58)  # y-location
        self.range_r = (0.03, 0.04)
        self.init_rot = (0.5, 0.5, -0.5, 0.5)
        self.init_gap = 0.45

    def check_collision(self, all_c, all_r):
        for i in range(len(all_c)):
            for j in range(i + 1, len(all_c)):
                dist = np.sqrt((all_c[i][0] - all_c[j][0]) ** 2 + (all_c[i][2] - all_c[j][2]) ** 2)
                if dist < all_r[i] + all_r[j] + 0.04:
                    return False
        return True

    def set_gripper_pre_merge(self, c1, c2):
        center = (np.array(c1) + np.array(c2)) / 2.
        theta = np.pi / 2 - atan2(c2[2] - c1[2], c2[0] - c1[0])
        q = rotate(self.init_rot, (0, 1, 0), theta)
        self.env.taichi_env.primitives[0].set_state(0, [center[0], center[1], center[2], q.w, q.x, q.y, q.z, self.init_gap])

    def case(self, cfg, all_c, all_r):
        cfg.SHAPES[0]['all_pos'] = all_c
        cfg.SHAPES[0]['all_r'] = all_r

    def sample_sphere(self):
        x, y, r = gen_rand(*self.range_x), gen_rand(*self.range_y), gen_rand(*self.range_r)
        c = (x, r, y)
        return c, r

    def merge(self):
        for j in range(35):  # Grasp in the other direction
            action = np.array([0] * 7)
            action[6] = ((j < 25) - 0.5) * 2
            self.env.step(action)

    def sample_spheres(self, N):
        # Sample collision free spheres
        while True:
            all_c, all_r = [], []
            for i in range(N):
                c, r = self.sample_sphere()
                all_c.append(c)
                all_r.append(r)
            if self.check_collision(all_c, all_r):  # No collision
                break
        return all_c, all_r

    def _generate_test(self):
        N = 5
        # Case 1: 4 -> 3
        np.random.seed(0)
        for i in range(N):
            all_c, all_r = self.sample_spheres(4)
            c1, c2, c3, c4 = all_c
            self.env.reset(target_cfg_modifier=partial(self.case, all_c=all_c, all_r=all_r))
            self.set_gripper_pre_merge(c1, c2)
            self.save_init(idx=i)
            self.merge()
            self.save_target(idx=i)

        # Case 2: 4 -> 2
        cnt = N
        for i in range(N):
            all_c, all_r = self.sample_spheres(4)
            c1, c2, c3, c4 = all_c
            self.env.reset(target_cfg_modifier=partial(self.case, all_c=all_c, all_r=all_r))
            self.set_gripper_pre_merge(c1, c2)
            self.save_init(idx=cnt)
            self.merge()
            self.save_target(idx=cnt)
            cnt += 1
            self.set_gripper_pre_merge((np.array(c1) + np.array(c2)) / 2., c3)
            self.save_init(idx=cnt)
            self.merge()
            self.save_target(idx=cnt)
            cnt += 1

    def _generate_train(self):
        N = 1000
        # Case 1: 2 -> 1
        np.random.seed(0)
        for i in range(N):
            all_c, all_r = self.sample_spheres(2)
            c1, c2 = all_c
            self.env.reset(target_cfg_modifier=partial(self.case, all_c=all_c, all_r=all_r))
            self.set_gripper_pre_merge(c1, c2)
            self.save_init(idx=i)
            self.merge()
            self.save_target(idx=i)

        # Case 2: 3 -> 2
        for i in range(N):
            all_c, all_r = self.sample_spheres(3)
            c1, c2, c3 = all_c
            self.env.reset(target_cfg_modifier=partial(self.case, all_c=all_c, all_r=all_r))
            self.set_gripper_pre_merge(c1, c2)
            self.save_init(idx=i + N)
            self.merge()
            self.save_target(idx=i + N)

        # Case 3: 3 -> 1
        cnt = 2 * N
        for i in range(N):
            all_c, all_r = self.sample_spheres(3)
            c1, c2, c3 = all_c
            self.env.reset(target_cfg_modifier=partial(self.case, all_c=all_c, all_r=all_r))
            self.set_gripper_pre_merge(c1, c2)
            self.save_init(idx=cnt)
            self.merge()
            self.save_target(idx=cnt)
            cnt += 1
            self.set_gripper_pre_merge((np.array(c1) + np.array(c2)) / 2., c3)
            self.save_init(idx=cnt)
            self.merge()
            self.save_target(idx=cnt)
            cnt += 1

    def _generate(self):
        if self.mode == 'train':
            self._generate_train()
        else:
            self._generate_test()
