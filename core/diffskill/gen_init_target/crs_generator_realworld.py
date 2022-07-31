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
        size = np.array([0.26, 0.08, 0.08])
        x = 0.9-size[0]*0.5
        y = 0.0669   # copied from cutrearrange
        z = 0.5+rand(-0.1, 0.1)

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
    
    def spread_state_test(self, init_state, flag, info):
        old_x, old_y, old_z, old_size, cut_loc = info
        # Set knife location back
        knife_state = init_state['state'][-3]
        knife_state[0] = 0.8
        knife_state[2] = 0.1

        x = rand(0.25, 0.28)
        y = 0.0669
        z = 0.5 + rand(-0.1, 0.1)
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



    def _generate(self):
        # in-distribution cases
        for i in range(4):
            cutted, flag, info = self.cut_state_case(i, self.initial_state, save_init=True)
            rolled = self.spread_state_test(cutted, flag, info)
            self.save_target(i, rolled)