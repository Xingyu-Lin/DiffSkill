import taichi as ti
import transforms3d
import tina
import numpy as np
import cv2


class TinaRenderer:
    def __init__(self, cfg, primitives=(), **kwargs):
        # overwrite configurations
        for i, v in kwargs.items():
            cfg[i] = v
        print("Initialize Tina Renderer")
        print(str(cfg).replace('\n', '  \n'))
        self.cfg = cfg

        init_camera_info = {
            'center': cfg.cam_center,
            'theta': cfg.cam_theta,
            'phi': cfg.cam_phi,
            'radius': cfg.cam_radius
        }
        self.scene = tina.Scene(res_x=cfg.tina_img_res, smoothing=True, texturing=True, taa=True, cfg=cfg, primitives=primitives,
                                **init_camera_info)
        self.primitives = primitives

        self.pars = tina.SimpleParticles()
        self.material = tina.Diffuse()

        # pc = np.load('/home/xingyu/Projects/PlasticineLab/data/particles.npy')
        # self.pars.set_particles(pc)

        self.scene.add_object(self.pars, self.material, raster='particle_sdf')
        self.scene.lighting.set_ambient_light([0.1, 0.1, 0.1])

        self.gui = None
        self.binds = {}
        self.verbose = False

    def update_camera(self, center, theta, phi, radius):
        self.scene.update_control(center=center, theta=theta, phi=phi, radius=radius, is_ortho=False)

    def state2mat(self, state):
        a = transforms3d.quaternions.quat2mat(state[3:7])  # notice ...
        x = np.eye(4)
        x[:3, :3] = a
        x[:3, 3] = state[:3]
        return x

    def clear_obj(self, pid):
        if pid in self.binds:
            o = self.binds[pid][0]
            if o in self.scene.objects:
                del self.scene.objects[o]

    def bind(self, pid, mesh, material, init_trans=None):
        self.clear_obj(pid)
        obj = tina.MeshTransform(mesh)
        self.binds[pid] = (obj, material, init_trans)
        if pid >= 0:
            self.visualize_primitive[pid] = 0

    def unbind(self, pid):
        self.clear_obj(pid)
        if pid in self.binds:
            del self.binds[pid]

    def unbind_all(self):
        for pid in self.binds:
            self.clear_obj(pid)
        self.binds = {}

    @property
    def visualize_primitive(self):
        return self.scene.particle_sdf_raster.visualize_primitive

    def render_object(self, idx, not_show_flag=1, state_fn=None):
        o, m, init = self.binds[idx]
        if o in self.scene.objects:
            del self.scene.objects[o]
        if not not_show_flag:
            if state_fn is not None:
                mat = self.state2mat(state_fn())
            else:
                mat = np.eye(4)

            if self.verbose:
                print(mat)
            if init is not None:
                mat = mat @ init
            o.set_transform(mat)

            if self.verbose:
                print(o, init)
                print(o.get_verts().min(axis=0))
                print(o.get_verts().max(axis=0))

            self.scene.add_object(o, m)

    def render(self, mode='human', img_size=64, **kwargs):
        vp = self.scene.particle_sdf_raster.visualize_primitive
        if -1 in self.binds:
            self.scene.particle_sdf_raster.ground[None] = -0.38
            self.render_object(-1, 0)
        else:
            self.scene.particle_sdf_raster.ground[None] = 3. / 128

        for idx, flag in enumerate(self.scene.particle_sdf_raster.visualize_primitive.to_numpy()):
            if idx not in self.binds:
                vp[idx] = 1
                continue
            self.render_object(idx, flag, lambda: self.primitives[idx].get_state(0))

        if mode == 'human':
            if self.gui is None:
                self.gui = tina.ti.GUI('primitives', res=self.scene.res)
            while self.gui.running:
                self.scene.input(self.gui, refresh=True)
                self.scene.render()
                self.gui.set_image(self.scene.img)
                self.gui.show()
        elif mode == 'rgb':
            if self.gui is None:
                self.gui = tina.ti.GUI('primitives', res=self.scene.res, show_gui=False)
            self.scene.input(self.gui, refresh=True)
            self.scene.render()
            img = self.scene.img.to_numpy()
            depth = self.local2global(self.scene.engine.depth.to_numpy().astype(np.float64) / self.scene.engine.maxdepth)
            depth = np.clip(depth, 0., 3.)
            ret_img = np.concatenate([img, depth[:, :, None]], axis=-1)
            if ret_img.shape[0] != img_size:
                ret_img = cv2.resize(ret_img[:, :, :], dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
            return ret_img

    def local2global(self, depth):
        return depth
        engine = self.scene.engine
        p = np.stack(np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]))[::-1], axis=-1)
        p = (p + engine.bias.to_numpy()) / np.array(engine.res) * 2 - 1
        v2w = engine.V2W.to_numpy()
        eye = np.ones_like(p[..., -1:])
        origin = np.concatenate((p, eye * 0, eye), -1) @ v2w.T
        reached = np.concatenate((p, depth[..., None], eye), -1) @ v2w.T
        origin = origin[..., :3] / origin[..., 3:4]
        reached = reached[..., :3] / reached[..., 3:4]
        return np.linalg.norm(reached - origin, axis=-1)

    def set_particles(self, x, color):
        self.pars.set_particles(x)

    def set_target_density(self, target_density=None):
        pass

    def initialize(self, cfg):
        # TODO actually update the cfg
        self.update_camera(cfg.cam_center, cfg.cam_theta, cfg.cam_phi, cfg.cam_radius)
