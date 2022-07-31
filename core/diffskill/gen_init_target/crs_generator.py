import numpy as np
from core.gen_init_target.state_generator import StateGenerator
import copy

def rand(a, b):
    return np.random.random() * (b - a) + a

def cut(state, cut_loc, slide_a, slide_b):
    ns = copy.deepcopy(state)
    flag = ns['state'][0][:, 0] <= cut_loc
    ns['state'][0][flag, 0] -= slide_a
    ns['state'][0][np.logical_not(flag), 0] += slide_b
    return ns, flag


def move_cluster(state, flag, dx, dz, dy=0):
    # need to move forward
    new_state = copy.deepcopy(state)
    mean = new_state['state'][0][flag].mean(axis=0)
    new_state['state'][0][flag] += np.array([dx, mean[1] + dy, dz]) - mean
    return new_state

def move_to(state, target_x, target_y, target_z):
    new_state = copy.deepcopy(state)
    mean = new_state['state'][0].mean(axis=0)
    diff = np.array([target_x, target_y, target_z]) - mean
    new_state['state'][0] += diff
    return new_state

class CRSGenerator(StateGenerator):
    def __init__(self, *args, **kwargs):
        super(CRSGenerator, self).__init__(*args, **kwargs)
        self.env.reset()
        for i in range(50):
            self.env.step(np.zeros((15,)))
        self.initial_state = self.env.get_state()
        self.N = 200
        self.all_cutloc = {}

    def cut_state_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        size = np.array([rand(0.2, 0.35), 0.08, rand(0.06, 0.08)])
        x = 0.9-size[0]*0.5
        y = 0.0669   # copied from cutrearrange
        z = 0.5+rand(-0.2, 0.2)

        # Set knife location to be on top of dough
        knife_state = state['state'][-3]
        knife_state[0] = x
        knife_state[2] = z

        # Set dough state
        state['state'][0] = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size) + np.array([x, y, z])
        if save_init:
            self.save_init(i, state)

        da = 0.1  # move left fixed length
        db = 0
        left_end = x-size[0]*0.5
        cut_loc = left_end + rand(0.1, 0.12) # Randomize cut
        self.all_cutloc[str(i)] = cut_loc
        cutted, flag = cut(state, cut_loc, da, db)
        return cutted, flag, [x,y,z,size,cut_loc]

    def move_state_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)

        size = np.array([rand(0.1, 0.12), 0.08, rand(0.06, 0.08)])
        x = rand(0.5, 0.65)
        y = 0.0669
        z = 0.5+rand(-0.2, 0.2)
        
        # Set pusher location to be on the right of dough
        box_state = state['state'][-2]
        box_state[0] = x+size[0]/2+0.02
        box_state[1] = 0.05
        box_state[2] = z

        # Set dough state
        state['state'][0] = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size) + np.array([x, y, z])
        if save_init:
            self.save_init(i+self.N, state)

        # move the dough
        target_x = rand(0.25, 0.3)
        target_y = y
        target_z = 0.5 + rand(-0.2, 0.2)
        moved = move_to(state, target_x, target_y, target_z)
        return moved
    
    def spread_state_case(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        size = np.array([rand(0.1, 0.12), 0.08, rand(0.06, 0.08)])
        x = rand(0.22, 0.32)
        y = 0.0669
        z = 0.5+rand(-0.2, 0.2)

        # Set roller location to be on top of dough
        roller_state = state['state'][-1]
        roller_state[0] = x
        roller_state[2] = z
        # Set dough state
        state['state'][0] = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size) + np.array([x, y, z])
        if save_init:
            self.save_init(i+self.N*2, state)
        
        # roll the dough
        volume = np.prod(size)
        h = rand(0.016, 0.018)
        w = 0.13
        l = volume / h / w
        new_size = np.array([l, h, w])
        new_state = copy.deepcopy(state)
        new_state['state'][0] = (np.random.random((len(new_state['state'][0]), 3)) * 2 - 1) * (0.5 * new_size) + np.array([x, 0.0269+h/2, z])
        return new_state
    

    def spread_state_test(self, init_state, flag, info):
        old_x, old_y, old_z, old_size, cut_loc = info
        # Set knife location back
        knife_state = init_state['state'][-3]
        knife_state[0] = 0.8
        knife_state[2] = 0.1

        x = rand(0.22, 0.32)
        y = 0.0669
        z = 0.5 + rand(-0.2, 0.2)
        new_state = move_cluster(init_state, flag, x-old_x, z-old_z)
        left_len = new_state['state'][0][flag].shape[0]
        left_size = np.array([cut_loc-(old_x-old_size[0]/2), old_size[1], old_size[2]])
        
        # roll the dough
        volume = np.prod(left_size)
        h = rand(0.016, 0.018)
        w = 0.13
        l = volume / h / w
        new_size = np.array([l, h, w])
        new_state['state'][0][flag, :] = (np.random.random((left_len, 3)) * 2 - 1) * (0.5 * new_size) + np.array([x, 0.0269+h/2, z])
        return new_state

    def test_case2(self, i, init_state, save_init=True):
        state = copy.deepcopy(init_state)
        size = np.array([rand(0.34, 0.35), 0.08, rand(0.06, 0.08)])
        x = 0.9-size[0]*0.5
        y = 0.0669   # copied from cutrearrange
        z = 0.5+rand(-0.15, 0.15)

        # Set knife location to be on top of dough
        knife_state = state['state'][-3]
        knife_state[0] = x
        knife_state[2] = z

        # Set dough state
        state['state'][0] = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size) + np.array([x, y, z])
        if save_init:
            self.save_init(i, state)

        da = 0.1  # move left fixed length
        db = 0
        left_end = x-size[0]*0.5
        cut_loc = left_end + rand(0.2, 0.24) # Randomize cut
        self.all_cutloc[str(i)] = cut_loc
        state, flag = cut(state, cut_loc, da, db)
        cut_loc1 = left_end-da+0.11
        state, flag1 = cut(state, cut_loc1, da, db)
        flag2 = np.logical_xor(flag, flag1)

        size_cut = np.array([cut_loc-left_end, size[1], size[2]])
        vol_cut = np.prod(size_cut)
        size1 = np.array([0.11, size[1], size[2]])
        vol1 = np.prod(size1)
        vol2 = vol_cut - vol1

        h1 = rand(0.016, 0.018)
        w1 = 0.13
        l1 = vol1 / h1 / w1
        new_size1 = np.array([l1, h1, w1])
        h2 = rand(0.016, 0.018)
        w2 = 0.13
        l2 = vol2 / h2 / w2
        new_size2 = np.array([l2, h2, w2])

        # Set knife location back
        knife_state = state['state'][-3]
        knife_state[0] = 0.8
        knife_state[2] = 0.1

        while True:
            z1, z2 = 0.5+rand(-0.15, 0.15), 0.5+rand(-0.15, 0.15)
            if abs(z2 - z1) > w1 + 0.12 and abs(z2 - z1) < w1 + 0.2:
                break
        x1, x2 = rand(0.22, 0.32), rand(0.22, 0.32)
        
        # roll the dough
        new_size1 = np.array([l1, h1, w1])
        state['state'][0][flag1, :] = (np.random.random((len(np.where(flag1)[0]), 3)) * 2 - 1) * (0.5 * new_size1) + np.array([x1, 0.0269+h1/2, z1])
        new_size2 = np.array([l2, h2, w2])
        state['state'][0][flag2, :] = (np.random.random((len(np.where(flag2)[0]), 3)) * 2 - 1) * (0.5 * new_size2) + np.array([x2, 0.0269+h2/2, z2])
        return state

    def _generate(self):
        if self.mode == 'train':
            for i in range(self.N):
                cutted, flag, _ = self.cut_state_case(i, self.initial_state)
                # self.save_init(i*2+1, cutted)
                self.save_target(i, cutted)
            for i in range(self.N):
                moved = self.move_state_case(i, self.initial_state)
                # self.save_init(i*2+1+self.N*2, moved)
                self.save_target(i+self.N, moved)
            for i in range(self.N):
                rolled = self.spread_state_case(i, self.initial_state)
                # self.save_init(i*2+1+self.N*4, rolled)
                self.save_target(i+self.N*2, rolled)
        # else:
            count = self.N*3
            # in-distribution cases
            for i in range(10):
                cutted, flag, info = self.cut_state_case(i+count, self.initial_state, save_init=True)
                rolled = self.spread_state_test(cutted, flag, info)
                self.save_target(i+count, rolled)    
            count += 10
            # cut-twice cases
            for i in range(10):
                final_state = self.test_case2(i+count, self.initial_state, save_init=True)
                self.save_target(i+count, final_state)

        import os 
        import pickle
        with open(os.path.join(self.init_dir, 'cut_loc.pkl'), 'wb') as f:
            pickle.dump(self.all_cutloc, f, protocol=4)