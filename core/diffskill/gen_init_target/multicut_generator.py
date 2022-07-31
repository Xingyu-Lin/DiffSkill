import numpy as np
from core.gen_init_target.state_generator import StateGenerator
import copy
import transforms3d
from functools import partial

from plb.utils.visualization_utils import save_numpy_as_gif, save_rgb

def rand(a, b):
    return np.random.random() * (b - a) + a


class MulticutGenerator(StateGenerator):
    def __init__(self, *args, **kwargs):
        super(MulticutGenerator, self).__init__(*args, **kwargs)
        self.N = 3
        self.N2 = 3
        self.range_x = (0.3, 0.7)  # Dough location on the cutting board
        self.range_z = (0.4, 0.6)
        self.range_l1 = (0.12, 0.18)
        self.range_l2 = (0.25, 0.3)
        self.range_l3 = (0.08, 0.1)
        self.range_w = (0.08, 0.1)
        self.range_ang = (-np.pi/2, np.pi/2)
        self.range_cut = (-0.5, 0.5)
        self.cur_pos = None
        self.cur_rot = None
    
    def check_collision(self, all_c, all_r):
        for i in range(len(all_c)):
            for j in range(i + 1, len(all_c)):
                dist = np.sqrt((all_c[i][0] - all_c[j][0]) ** 2 + (all_c[i][2] - all_c[j][2]) ** 2)
                if dist < all_r[i][0]/2 + all_r[j][0]/2 + 0.04:
                    return False
        return True

    def sample_box(self, long=False):
        lrange = self.range_l1 if not long else self.range_l2
        range_x = self.range_x if not long else (0.4, 0.6)
        range_z = self.range_z if not long else (0.45, 0.55)
        x, y, l, w, ang = rand(*range_x), rand(*range_z), rand(*lrange), rand(*self.range_w), rand(*self.range_ang)
        pos = (x, w, y)
        width = (l, w, w)
        rot = transforms3d.quaternions.axangle2quat([0, 1, 0], ang)
        return pos, width, rot

    def sample_boxes(self, N, long=False):
        # Sample collision free spheres
        while True:
            all_c, all_w, all_pose = [], [], []
            for i in range(N):
                pos, width, pose = self.sample_box(long=long)
                all_c.append(pos)
                all_w.append(width)
                all_pose.append(pose)
            if self.check_collision(all_c, all_w):  # No collision
                break
        return all_c, all_w, all_pose
    
    def case1(self, cfg, num_boxes=1, long=False):  # 1->2
        pos, widths, init_rots = self.sample_boxes(num_boxes, long=long)
        cut_prop = rand(*self.range_cut)
        self.cur_pos = pos
        self.cur_rot = init_rots
        cfg.SHAPES[0]['all_pos'] = pos
        cfg.SHAPES[0]['all_width'] = widths
        cfg.SHAPES[0]['all_rot'] = init_rots
        
        i = np.random.choice(len(pos))
        vec, ang = transforms3d.quaternions.quat2axangle(init_rots[i])
        ang = vec[1] * ang
        dx, dz = np.cos(-ang) * (cut_prop * widths[i][0] / 4 - widths[i][0] / 8), np.sin(-ang) * (cut_prop * widths[i][0] / 4 - widths[i][0] / 8)
        cfg.PRIMITIVES[0]['init_pos'] = (float(dx + pos[i][0]), 0.3, float(dz + pos[i][2]))
        cfg.PRIMITIVES[0]['init_rot'] = (float(init_rots[i][0]), float(init_rots[i][1]), float(init_rots[i][2]), float(init_rots[i][3]))
    
    def generate12(self, N, start_idx):
        np.random.seed(0)
        # 1->2
        for i in range(N):
            self.env.reset(target_cfg_modifier=partial(self.case1))
            direction = np.random.binomial(1, 0.5) * 2 - 1
            # img = self.taichi_env.render(mode='rgb', img_size=128)
            for _ in range(20):
                self.env.step(np.array([0] * 3))  # Wait for dough to drop
            self.save_init(start_idx+i)
            # Cut
            for j in range(20):
                action = [0., -1., 0.]
                self.env.step(action)
            # Move a little bit towards the normal direction
            quat = self.taichi_env.primitives[0].get_state(0)[3:]
            vec, ang = transforms3d.quaternions.quat2axangle(quat)
            ang = vec[1] * ang
            for j in range(12):
                dx, dz = np.cos(-ang)*0.5, np.sin(-ang)*0.5
                action = np.array([dx, 0.,dz]) * direction
                self.env.step(action)
            self.save_target(start_idx+i)

    def generate23(self, N, start_idx):
        np.random.seed(0)
        # 2->3
        for i in range(N):
            self.env.reset(target_cfg_modifier=partial(self.case1, num_boxes=2))
            direction = np.random.binomial(1, 0.5) * 2 - 1
            for _ in range(20):
                self.env.step(np.array([0] * 3))  # Wait for dough to drop

            self.save_init(start_idx+i)
            # Cut
            for j in range(20):
                action = [0., -1., 0.]
                self.env.step(action)
            # Move a little bit towards the normal direction
            quat = self.taichi_env.primitives[0].get_state(0)[3:]
            vec, ang = transforms3d.quaternions.quat2axangle(quat)
            ang = vec[1] * ang
            # assert np.allclose(vec, [0., -1., 0])
            for j in range(12):
                dx, dz = np.cos(-ang)*0.5, np.sin(-ang)*0.5
                action = np.array([dx, 0.,dz]) * direction
                self.env.step(action)
            self.save_target(start_idx+i)

    def generate13(self, N, start_idx):
        np.random.seed(0)
        # 1->3
        for i in range(N):
            self.env.reset(target_cfg_modifier=partial(self.case1, long=True))
            direction = np.random.binomial(1, 0.5) * 2 - 1
            orig_primitive_state = self.taichi_env.primitives[0].get_state(0)
            for _ in range(20):
                self.env.step(np.array([0] * 3))  # Wait for dough to drop
            self.save_init(start_idx+i)
            # Cut
            # imgs = []
            for j in range(20):
                action = [0., -1., 0.]
                self.env.step(action)
            # Move a little bit towards the normal direction
            quat = self.taichi_env.primitives[0].get_state(0)[3:]
            vec, ang = transforms3d.quaternions.quat2axangle(quat)
            ang = vec[1] * ang
            for j in range(12):
                dx, dz = np.cos(-ang)*0.5, np.sin(-ang)*0.5
                action = np.array([dx, 0.,dz]) * direction
                self.env.step(action)
            # Move the tool back
            self.taichi_env.primitives[0].set_state(0, orig_primitive_state)
            # Move the tool to cut the other piece
            for j in range(18 + 6*direction):
                dx, dz = np.cos(-ang)*0.5, np.sin(-ang)*0.5
                action = np.array([dx, 0.,dz])
                self.env.step(action)
            # Cut
            for j in range(20):
                action = [0., -1., 0.]
                self.env.step(action)
            # Move a little bit towards the normal direction
            for j in range(12):
                dx, dz = np.cos(-ang)*0.5, np.sin(-ang)*0.5
                action = np.array([dx, 0.,dz]) * 1
                self.env.step(action)
            for _ in range(20):
                self.env.step(np.array([0] * 3))  # Wait for dough to drop
            self.save_target(start_idx+i)

    def generate34(self, N, start_idx):
        np.random.seed(0)
        # 3->4
        for i in range(N):
            self.env.reset(target_cfg_modifier=partial(self.case1,num_boxes=3))
            direction = np.random.binomial(1, 0.5) * 2 - 1
            for _ in range(20):
                self.env.step(np.array([0] * 3))  # Wait for dough to drop
            self.save_init(start_idx+i)
            # Cut
            for j in range(20):
                action = [0., -1., 0.]
                self.env.step(action)
            # Move a little bit towards the normal direction
            quat = self.taichi_env.primitives[0].get_state(0)[3:]
            vec, ang = transforms3d.quaternions.quat2axangle(quat)
            ang = vec[1] * ang
            # assert np.allclose(vec, [0., -1., 0])
            for j in range(12):
                dx, dz = np.cos(-ang)*0.5, np.sin(-ang)*0.5
                action = np.array([dx, 0.,dz]) * direction
                self.env.step(action)
            self.save_target(start_idx+i)

    def generate24(self, N, start_idx):
        np.random.seed(0)
        # 2->4
        for i in range(N):
            self.env.reset(target_cfg_modifier=partial(self.case1,num_boxes=2))
            direction = np.random.binomial(1, 0.5) * 2 - 1
            orig_primitive_state = self.taichi_env.primitives[0].get_state(0)
            for _ in range(20):
                self.env.step(np.array([0] * 3))  # Wait for dough to drop
            self.save_init(start_idx+i)
            # Cut
            # imgs = []
            for j in range(20):
                action = [0., -1., 0.]
                self.env.step(action)
                # imgs.append(self.taichi_env.render(mode='rgb', img_size=128))
            # Move a little bit towards the normal direction
            quat = self.taichi_env.primitives[0].get_state(0)[3:]
            vec, ang = transforms3d.quaternions.quat2axangle(quat)
            ang = vec[1] * ang
            for j in range(12):
                dx, dz = np.cos(-ang)*0.5, np.sin(-ang)*0.5
                action = np.array([dx, 0.,dz]) * direction
                self.env.step(action)
                # imgs.append(self.taichi_env.render(mode='rgb', img_size=128))
            # Move the tool to the other piece
            reset_i = 0
            for idx, rot in enumerate(self.cur_rot):
                if not np.allclose(rot, orig_primitive_state[3:]):
                    reset_i = idx
            state = np.concatenate([self.cur_pos[reset_i], self.cur_rot[reset_i]])
            state[1] = 0.3
            self.taichi_env.primitives[0].set_state(0, state)
            vec, ang = transforms3d.quaternions.quat2axangle(self.cur_rot[reset_i])
            ang = vec[1] * ang
            # Cut
            for j in range(20):
                action = [0., -1., 0.]
                self.env.step(action)
                # imgs.append(self.taichi_env.render(mode='rgb', img_size=128))
            # Move a little bit towards the normal direction
            for j in range(12):
                dx, dz = np.cos(-ang)*0.5, np.sin(-ang)*0.5
                action = np.array([dx, 0.,dz]) * 1
                self.env.step(action)
                # imgs.append(self.taichi_env.render(mode='rgb', img_size=128))
            for _ in range(20):
                self.env.step(np.array([0] * 3))  # Wait for dough to stop
            # import os
            # gifpath = os.path.join('debug', f'vis_{i}.gif')
            # save_numpy_as_gif(np.array(imgs)[:, :, :, :3], gifpath)
            self.save_target(start_idx+i)   




    def _generate_train(self):
        count = 0
        N = 1000
        self.generate12(N, count)
        count += N
        self.generate23(N, count)
        count += N
        self.generate13(N, count)
        count += N
        self.generate34(N, count)
        count += N

    def _generate_test(self):
        count = 0
        N = 1000
        self.generate24(N, count)
        count += N



    def _generate(self):
        if self.mode == 'train':
            self._generate_train()
        else:
            self._generate_test()
        