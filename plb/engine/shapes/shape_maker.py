from re import L
import numpy as np
import copy

COLORS = [
    (127 << 16) + 127,
    (127 << 8),
    127,
    127 << 16,
]


class Shapes:
    # make shapes from the configuration
    def __init__(self, cfg):
        self.objects = []
        self.colors = []

        self.dim = 3

        state = np.random.get_state()

        for i in cfg:
            kwargs = {key: eval(val) if isinstance(val, str) else val for key, val in i.items() if key != 'shape'}
            if i['shape'] == 'box':
                self.add_box(**kwargs)
            elif i['shape'] == 'multibox':
                self.add_multibox(**kwargs)
            elif i['shape'] == 'sphere':
                self.add_sphere(**kwargs)
            elif i['shape'] == 'multisphere':
                self.add_multisphere(**kwargs)
            elif i['shape'] == 'capsule':
                self.add_capsule(**kwargs)
            elif i['shape'] == 'cylinder':
                self.add_cylinder(**kwargs)
            elif i['shape'] == 'scatter':
                np.random.seed(kwargs['seed'])  # fix seed 0
                pos_min, pos_max = kwargs['pos_min'], kwargs['pos_max']
                N = 40
                particle_pos = []
                for pmin, pmax in zip(pos_min, pos_max):
                    particle_pos.append(np.random.uniform(pmin, pmax, size=N).reshape([N, 1]))
                particle_pos = np.hstack(particle_pos)

                multiply = 50
                noise = np.random.normal(0, scale=0.004, size=N * 3 * multiply).reshape(multiply, N, 3)
                noise[:, :, 1] = noise[:, :, 1] * 0.2 + 0.1  # Add height
                particle_pos = (particle_pos.reshape([1, N, 3]) + noise).reshape(multiply * N, 3)

                self.add_object(particle_pos, color=kwargs['color'])
            else:
                raise NotImplementedError(f"Shape {i['shape']} is not supported!")
        np.random.set_state(state)

    def get_n_particles(self, volume):
        return max(int(volume / (0.1 ** 3) * 30000), 1)

    def add_object(self, particles, color=None, init_rot=None):
        if init_rot is not None:
            import transforms3d
            q = transforms3d.quaternions.quat2mat(init_rot)
            origin = particles.mean(axis=0)
            particles = (particles[:, :self.dim] - origin) @ q.T + origin
        self.objects.append(particles[:, :self.dim])
        if color is None or isinstance(color, int):
            tmp = COLORS[len(self.objects) - 1] if color is None else color
            color = np.zeros(len(particles), np.int32)
            color[:] = tmp
        self.colors.append(color)

    def add_multibox(self, all_pos, all_width, all_rot, color):
        volumes = [np.prod(width) for width in all_width]
        total_volume = sum(volumes)
        print(volumes)
        n_particles = min(30000, self.get_n_particles(total_volume))
        print('n particles:', n_particles)
        for (init_pos, init_width, init_rot, vol) in zip(all_pos, all_width, all_rot, volumes):
            self.add_box(init_pos, init_width, n_particles=int(n_particles * vol / total_volume), color=color, init_rot=init_rot)

    def add_box(self, init_pos, width, n_particles=None, color=None, init_rot=None):
        # pass
        if isinstance(width, float):
            width = np.array([width] * self.dim)
        else:
            width = np.array(width)
        if n_particles is None:
            n_particles = self.get_n_particles(np.prod(width))
        # print('------------------------ making box with width:', width)
        p = (np.random.random((n_particles, self.dim)) * 2 - 1) * (0.5 * width) + np.array(init_pos)
        self.add_object(p, color, init_rot=init_rot)

    def add_multisphere(self, all_pos, all_r, color):
        volumes = [(r ** 3) * 4 * np.pi / 3 for r in all_r]
        total_volume = sum(volumes)
        print(volumes)
        n_particles = self.get_n_particles(total_volume)
        print('n particles:', n_particles)
        for (init_pos, r, vol) in zip(all_pos, all_r, volumes):
            print('     each:', int(n_particles * vol / total_volume))
            self.add_sphere(init_pos, r, n_particles=int(n_particles * vol / total_volume))
            # self.add_sphere(init_pos, r, n_particles=30000/len(all_pos)))

    def add_sphere(self, init_pos, radius, n_particles=None, color=None, init_rot=None):
        if n_particles is None:
            if self.dim == 3:
                volume = (radius ** 3) * 4 * np.pi / 3
            else:
                volume = (radius ** 2) * np.pi
            n_particles = self.get_n_particles(volume)

        p = np.random.normal(size=(n_particles, self.dim))
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        u = np.random.random(size=(n_particles, 1)) ** (1. / self.dim)
        p = p * u * radius + np.array(init_pos)[:self.dim]
        self.add_object(p, color, init_rot=init_rot)

    def add_capsule(self, init_pos, radius, height, n_particles=None, color=None, init_rot=None):
        if n_particles is None:
            if self.dim == 3:
                volume = (radius ** 3) * 4 * np.pi / 3
            else:
                volume = (radius ** 2) * np.pi
            n_particles = self.get_n_particles(volume)

        assert self.dim == 3
        v_sphere = (radius ** 3) * 4 * np.pi / 3
        v_cylinder = (radius ** 2) * np.pi * height
        n1 = int(n_particles * v_sphere / (v_sphere + v_cylinder))
        n2 = n_particles - n1
        p = np.random.normal(size=(n1, self.dim))
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        u = np.random.random(size=(n1, 1)) ** (1. / self.dim)
        p = p * u * radius
        p[:, 2] += np.sign(p[:, 2]) * height / 2.
        p += np.array(init_pos)[:self.dim]

        # For the cylinder
        r = radius * np.sqrt(np.random.random([n2, 1]))
        theta = np.random.random([n2, 1]) * 2 * np.pi
        x, y = (np.cos(theta) * r).reshape(-1, 1), (np.sin(theta) * r).reshape(-1, 1)
        h = np.random.random([n2, 1]) * height - height / 2.
        p_c = np.hstack([x, y, h]) + init_pos
        p = np.vstack([p, p_c])

        self.add_object(p, color, init_rot=init_rot)

    def add_cylinder(self, init_pos, radius, height, n_particles=None, color=None, init_rot=None):
        if n_particles is None:
            if self.dim == 3:
                volume = (radius ** 3) * 4 * np.pi / 3
            else:
                volume = (radius ** 2) * np.pi
            n_particles = self.get_n_particles(volume)

        assert self.dim == 3

        # For the cylinder
        r = radius * np.sqrt(np.random.random([n_particles, 1]))
        theta = np.random.random([n_particles, 1]) * 2 * np.pi
        x, y = (np.cos(theta) * r).reshape(-1, 1), (np.sin(theta) * r).reshape(-1, 1)
        h = np.random.random([n_particles, 1]) * height - height / 2.
        p = np.hstack([x, y, h]) + init_pos

        self.add_object(p, color, init_rot=init_rot)

    def get(self):
        assert len(self.objects) > 0, "please add at least one shape into the scene"
        return np.concatenate(self.objects), np.concatenate(self.colors)
