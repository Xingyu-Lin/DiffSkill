import numpy as np
from core.gen_init_target.state_generator import StateGenerator
from functools import partial


class GathermoveGenerator(StateGenerator):
    def __init__(self, *args, **kwargs):
        super(GathermoveGenerator, self).__init__(*args, **kwargs)
        self.N = 100
        self.xs = np.linspace(0.36, 0.4, self.N)
        self.rs = np.linspace(0.04, 0.07, self.N // 10)

    def case(self, cfg, i):  # Sphere
        # Disable collision checking during generation. Need to disable both, or the collision will not be disabled
        cfg.PRIMITIVES[0]['collision_group'] = (0, 0, 0)
        cfg.PRIMITIVES[1]['collision_group'] = (0, 0, 0)
        cfg.SHAPES[0]['seed'] = i

    def case2(self, cfg, i):
        cfg.SHAPES[0]['shape'] = 'sphere'
        del cfg.SHAPES[0]['pos_max'], cfg.SHAPES[0]['pos_min'], cfg.SHAPES[0]['seed']
        pos = (0.65, 0.08, 0.5)
        r = self.rs[i * 11117771 % 12837119 % self.N // 10]
        cfg.SHAPES[0]['radius'] = r
        new_pos = (self.xs[i], r + 0.08, pos[2])
        cfg.SHAPES[0]['init_pos'] = new_pos

    def _generate(self):
        for i in range(self.N):
            self.env.reset(target_cfg_modifier=partial(self.case, i=i))
            for _ in range(10):
                self.env.step(np.array([0] * 13))  # Wait for dough to drop
            self.save_init(2 * i)
            if i % 2 ==0:
                n = 25
            else:
                n = 10
            for j in range(n):  # Grasp in one direction
                action = np.array([0] * 7 + [0.] * 6)
                action[6] = ((j < 15) - 0.5) * 2
                self.env.step(action)
            self.taichi_env.primitives[0].set_state(0, [0.7, 0.06, 0.5, 0.707, 0.707, 0., 0., 0.4])
            for j in range(25):  # Grasp in the other direction
                action = np.array([0] * 7 + [0.] * 6)
                action[6] = ((j < 15) - 0.5) * 2
                self.env.step(action)
            self.taichi_env.primitives[0].set_state(0, [0.7, 0.06, 0.5, 0.5, 0.5, -0.5, 0.5, 0.4])
            self.save_init(2 * i + 1)
            self.save_target(i)

        for i in range(self.N):
            self.env.reset(target_cfg_modifier=partial(self.case2, i=i))
            self.save_target(i + self.N)
