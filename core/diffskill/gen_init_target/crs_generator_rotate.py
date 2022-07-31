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

class CRSGeneratorRot(StateGenerator):
    def __init__(self, *args, **kwargs):
        super(CRSGeneratorRot, self).__init__(*args, **kwargs)
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
        gen_state = (np.random.random((len(state['state'][0]), 3)) * 2 - 1) * (0.5 * size) 
        for t in range(3):
            state_cpy = gen_state.copy()
            ang = i*3 + t
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            xz = np.hstack([state_cpy[:, 0:1], state_cpy[:, 2:3]]).T
            new_xz = rot@xz
            state_cpy[:, 0] = new_xz[0]
            state_cpy[:, 2] = new_xz[1]
            state_cpy += np.array([x, y, z])
            state['state'][0] = state_cpy
            self.save_init(620+i*3+t, state)
            
        
        # roll the dough
        volume = np.prod(size)
        h = rand(0.016, 0.018)
        w = 0.13
        l = volume / h / w
        new_size = np.array([l, h, w])
        new_state = copy.deepcopy(state)
        new_state['state'][0] = (np.random.random((len(new_state['state'][0]), 3)) * 2 - 1) * (0.5 * new_size) + np.array([x, 0.0269+h/2, z])
        return new_state


    def _generate(self):
        self.angles = np.load('angles.npy')
        if self.mode == 'train':
            for i in range(self.N):
                cutted, flag, _ = self.cut_state_case(i, self.initial_state, save_init=False)
                # self.save_init(i*2+1, cutted)
                # self.save_target(i, cutted)
            for i in range(self.N):
                moved = self.move_state_case(i, self.initial_state, save_init=False)
                # self.save_init(i*2+1+self.N*2, moved)
                # self.save_target(i+self.N, moved)
            for i in range(self.N):
                rolled = self.spread_state_case(i, self.initial_state)
                self.save_target(620+i*3, rolled)
                self.save_target(620+i*3+1, rolled)
                self.save_target(620+i*3+2, rolled)