from ..common import *

"""
Adapted from taichi element
"""
import taichi as ti
import numpy as np
import math
import time
from plb.engine.renderer.renderer_utils import ray_aabb_intersection, inf, out_dir

DIFFUSE = 0
SPECULAR = 1

fov = 0.23
dist_limit = 100

exposure = 1.5
light_direction_noise = 0.03
light_color = [1.0, 1.0, 1.0]


# def color_1to3(color):
#     return V3(float(color >> 16) / 255., float(color >>8) / 255.)

@ti.data_oriented
class ParticleSDFRaster:
    def __init__(self, engine, coloring=True,
                 clipping=True, cfg=None, primitives=(), **kwargs):

        # overwrite configurations
        for i, v in kwargs.items():
            cfg[i] = v
        self.dx = cfg.dx
        self.spp = cfg.spp
        self.voxel_res = cfg.voxel_res
        self.max_num_particles = cfg.max_num_particles
        self.bake_size = cfg.bake_size
        self.max_ray_depth = cfg.max_ray_depth
        self.use_directional_light = cfg.use_directional_light
        self.light_direction = cfg.light_direction
        self.use_roulette = cfg.use_roulette

        self.vignette_strength = 0.9
        self.vignette_radius = 0.0
        self.vignette_center = [0.5, 0.5]

        self.inv_dx = 1 / self.dx
        self.sdf_threshold = ti.field(dtype=ti.f32, shape=())
        self.ground = ti.field(dtype=ti.f32, shape=())

        self.sdf_threshold[None] = cfg.sdf_threshold
        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.camera_rot = ti.Vector.field(2, dtype=ti.f32, shape=())

        # plb part
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32)
        self.depth_buffer = ti.field(dtype=ti.f32)
        self.bbox = ti.Vector.field(3, dtype=ti.f32, shape=2)

        self.particle_x = ti.Vector.field(3, dtype=ti.f32)
        self.particle_color = ti.field(dtype=ti.i32)

        self.volume = ti.field(dtype=ti.int64)
        self.sdf = ti.field(dtype=ti.f32)
        self.sdf_copy = ti.field(dtype=ti.f32)

        self.target_res = cfg.target_res
        self.target_density = ti.field(dtype=ti.f32, shape=self.target_res)
        self.target_density2 = ti.field(dtype=ti.f32, shape=self.target_res)
        self.color_vec = ti.Vector.field(3, dtype=ti.f32)
        self.target_density_color = ti.Vector([0.1, 0.3, 0.9])

        self.primitives = primitives

        ti.root.dense(ti.l, self.max_num_particles).place(self.particle_x, self.particle_color)
        ti.root.dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, [i // 4 for i in self.voxel_res]).place(self.volume, self.sdf,
                                                                                               self.sdf_copy,
                                                                                               self.color_vec)
        self.verts, self.colors = self.particle_x, self.particle_color

        # flags
        self.visualize_target = ti.field(dtype=ti.i32, shape=())
        self.visualize_primitive = ti.field(dtype=ti.i32, shape=(len(self.primitives)))
        self.visualize_shape = ti.field(dtype=ti.i32, shape=())
        self.visualize_shape[None] = 1

        # Tina part
        self.engine = engine
        self.res = self.engine.res
        self.coloring = coloring
        self.clipping = clipping

        self.occup = ti.field(int, self.res)
        self.npars = ti.field(int, ())
        maxpars = cfg.max_num_particles

        @ti.materialize_callback
        def init_pars():
            if self.coloring:
                self.colors.fill(1)
                self.ground[None] = 3. / 128
                self.visualize_primitive.fill(1)

    # Tina part
    # -----------------------------------------------------
    @ti.func
    def get_particles_range(self):
        for i in range(self.npars[None]):
            yield i

    @ti.func
    def get_particle_position(self, f):
        return self.verts[f]

    @ti.func
    def get_particle_radius(self, f):
        return self.sizes[f]

    @ti.func
    def get_particle_color(self, f):
        return self.colors[f]

    @ti.kernel
    def set_particles(self, verts: ti.ext_arr()):
        self.npars[None] = min(verts.shape[0], self.verts.shape[0])
        for i in range(self.npars[None]):
            for k in ti.static(range(3)):
                self.verts[i][k] = verts[i, k]

    # @ti.kernel
    # def set_particle_radii(self, sizes: ti.ext_arr()):
    #     for i in range(self.npars[None]):
    #         self.sizes[i] = sizes[i]
    #
    # @ti.kernel
    # def set_particle_colors(self, colors: ti.ext_arr()):
    #     ti.static_assert(self.coloring)
    #     for i in range(self.npars[None]):
    #         for k in ti.static(range(3)):
    #             self.colors[i][k] = colors[i, k]

    @ti.kernel
    def set_object(self, pars: ti.template()):
        pars.pre_compute()
        self.npars[None] = pars.get_npars()
        for i in range(self.npars[None]):
            vert = pars.get_particle_position(i)
            self.verts[i] = vert
            if ti.static(self.coloring):
                color = V3(1.)
                # ti.ti_print('set object: ', (int(color[0] * 255) << 16) + (int(color[1] * 255) << 8) + int(color[0] * 255))
                self.colors[i] = (int(color[0] * 255) << 16) + (int(color[1] * 255) << 8) + int(color[0] * 255)

    @ti.kernel
    def render_occup(self):
        pass
        # for P in ti.grouped(self.occup):
        #     self.occup[P] = -1
        # for f in ti.smart(self.get_particles_range()):
        #     Al = self.get_particle_position(f)
        #     Rl = self.get_particle_radius(f)
        #     Av = self.engine.to_viewspace(Al)
        #     if ti.static(self.clipping):
        #         if not -1 <= Av.z <= 1:
        #             continue
        #
        #     DXl = mapply_dir(self.engine.V2W[None], V(1., 0., 0.)).normalized()
        #     DYl = mapply_dir(self.engine.V2W[None], V(0., 1., 0.)).normalized()
        #     Rv = V(0., 0.)
        #     Rv.x = self.engine.to_viewspace(Al + DXl * Rl).x - Av.x
        #     Rv.y = self.engine.to_viewspace(Al + DYl * Rl).y - Av.y
        #     Bv = [
        #         Av - V(Rv.x, 0., 0.),
        #         Av + V(Rv.x, 0., 0.),
        #         Av - V(0., Rv.y, 0.),
        #         Av + V(0., Rv.y, 0.),
        #     ]
        #     a = self.engine.to_viewport(Av)
        #     b = [self.engine.to_viewport(Bv) for Bv in Bv]
        #
        #     bot, top = ifloor(min(b[0], b[2])), iceil(max(b[1], b[3]))
        #     bot, top = max(bot, 0), min(top, self.res - 1)
        #     for P in ti.grouped(ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1))):
        #         p = float(P) + self.engine.bias[None]
        #         Pv = V23(self.engine.from_viewport(p), Av.z)
        #         Pl = self.engine.from_viewspace(Pv)
        #         if (Pl - Al).norm_sqr() > Rl ** 2:
        #             continue
        #
        #         depth_f = Av.z
        #         depth = int(depth_f * self.engine.maxdepth)
        #         if ti.atomic_min(self.engine.depth[P], depth) > depth:
        #             if self.engine.depth[P] >= depth:
        #                 self.occup[P] = f

    @ti.kernel
    def render_color(self, shader: ti.template()):
        for P in ti.grouped(self.occup):
            p = float(P) + self.engine.bias[None]
            Pv = V23(self.engine.from_viewport(p), 1.)
            Pl = self.engine.from_viewspace(Pv)

            Pcamv = V23(self.engine.from_viewport(p), 0.)
            Pcaml = self.engine.from_viewspace(Pcamv)

            pos = Pcaml
            d = (Pl - Pcaml).normalized()
            # ti.ti_print('pos:', pos)
            # ti.ti_print('d:', d)

            closest, normal, c, roughness, material = self.next_hit(pos, d)
            # if c.max() > 0.2:
            #     ti.ti_print('color:', c)
            c *= 3
            f = 0  # Random number
            # color = self.get_particle_color(f)
            #reached = self.engine.to_viewspace(pos + closest * d) - Pcamv
            #closest = ti.sqrt(reached.dot(reached))
            #print(dd, closest, reached, Pcamv)

            depth = closest * self.engine.maxdepth
            #print(depth, self.engine.depth[P], closest, self.engine.maxdepth)
            # This only works in orthogonal projection..
            if ti.atomic_min(self.engine.depth[P], depth) > depth:
                shader.shade_color(self.engine, P, p, f, pos, normal, V(0., 0.), c)
                #ti.atomic_min(self.engine.depth[P], depth)

    # -----------------------------------------------------
    # build sdf from particles
    # -----------------------------------------------------
    @ti.func
    def smooth(self, volume, volume_out, res: ti.template()):
        a, b, c = ti.static(res)
        for id in ti.grouped(volume):
            if id[0] >= 1 and id[1] >= 1 and id[2] >= 1 and id[0] < a - 1 and id[1] < b - 1 and id[2] < c - 1:
                sum = 0.0
                for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    sum += volume[id + ti.Vector([i, j, k])]
                volume_out[id] = sum / 27.
            else:
                volume_out[id] = 1.

    @ti.kernel
    def build_sdf_from_particles(self):
        # bake
        size = ti.static(self.bake_size)
        resx, resy, resz = ti.static(self.voxel_res)
        for i in ti.grouped(self.volume):
            self.volume[i] = (self.volume[i] + 1) * 2 - 1

        num_p = self.npars[None]
        for id, i, j, k in ti.ndrange(num_p, (-size - 1, size + 1), (-size - 1, size + 1), (-size - 1, size + 1)):
            p = (self.particle_x[id] - self.bbox[0]) * self.inv_dx  # 0 is the lower bound
            color = self.particle_color[id]
            coord = p.cast(ti.i32)

            idx = coord + ti.Vector([i, j, k])
            if idx[0] >= 0 and idx[1] >= 0 and idx[2] >= 0 and \
              idx[0] < resx and idx[1] < resy and idx[2] < resz:
                dist = (idx - p).norm()
                dist = min(max(0, (255 * 0.2 * dist)), 255)
                char = (ti.cast(dist, ti.int64) << 24) + color
                ti.atomic_min(self.volume[idx], char)

        for i in ti.grouped(self.volume):
            c = self.volume[i]
            for j in ti.static(range(2, -1, -1)):
                self.color_vec[i][j] = (c & 255) / 255.
                c = c >> 8
            self.sdf[i] = (c & 255) / 255.

        for _ in ti.static(range(1)):
            self.smooth(self.sdf, self.sdf_copy, self.voxel_res)
            self.smooth(self.sdf_copy, self.sdf, self.voxel_res)

    # -----------------------------------------------------
    # sample textures
    # -----------------------------------------------------
    @ti.func
    def sample_tex(self, tex, pos, res: ti.template()):
        # bilinear interpolation to sample a tex from texture
        a, b, c = ti.static(res)
        pos = pos * ti.Vector([a, b, c])
        base = ti.min(ti.cast(pos, ti.i32), ti.Vector([a, b, c]) - 1)  # clip
        fx = pos - base

        x, y, z = base[0], base[1], base[2]
        x1, y1, z1 = min(x + 1, a - 1), min(y + 1, b - 1), min(z + 1, c - 1)  # clip again..
        c00 = tex[base] * (1 - fx[0]) + tex[x1, y, z] * fx[0]
        c01 = tex[x, y, z1] * (1 - fx[0]) + tex[x1, y, z1] * fx[0]
        c10 = tex[x, y1, z] * (1 - fx[0]) + tex[x1, y1, z] * fx[0]
        c11 = tex[x, y1, z1] * (1 - fx[0]) + tex[x1, y1, z1] * fx[0]

        c0 = c00 * (1 - fx[1]) + c10 * fx[1]
        c1 = c01 * (1 - fx[1]) + c11 * fx[1]

        return c0 * (1 - fx[2]) + c1 * fx[2]

    @ti.func
    def sample_sdf(self, pos):
        pos = (pos - self.bbox[0]) / (self.bbox[1] - self.bbox[0])
        out = 0.0
        if pos.min() >= 0 and pos.max() <= 1:
            out = self.sample_tex(self.sdf, pos, self.voxel_res) - self.sdf_threshold  # 0.35 sdf_threshold is similar to the particle radius
        return out

    @ti.func
    def sample_color(self, pos):
        pos = (pos - self.bbox[0]) / (self.bbox[1] - self.bbox[0])
        out = ti.Vector([0., 0., 0.])
        if pos.min() >= 0 and pos.max() <= 1:
            out = self.sample_tex(self.color_vec, pos, self.voxel_res)
        return out

    @ti.func
    def sample_normal(self, sample_sdf_func: ti.template(), p):
        d = 1e-3  # this seems to be important ... otherwise it won't be smooth..
        n = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            inc = p
            dec = p
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (sample_sdf_func(inc) - sample_sdf_func(dec))
        return n.normalized()

    @ti.func
    def ground_color(self, p):
        color = ti.Vector([0.3, 0.5, 0.7])
        if p[0] <= 1 and p[0] >= 0 and p[2] <= 1 and p[2] >= 0:
            #color *= ((ti.cast(p[0] / 0.25, ti.int32) + ti.cast(p[2] / 0.25, ti.int32)) % 2) * 0.2 + 0.35
            color *= 0.3
        else:
            # color *= ((ti.cast(p[0] / 0.25, ti.int32) + ti.cast(p[2] / 0.25, ti.int32)) % 2) * 0.2 + 0.35
            color *= 0.3
            # color = ti.Vector([0., 0., 0.])
        return color

    @ti.func
    def sample_target_density(self, p):
        return self.sample_tex(self.target_density, p, self.target_res)

    # -----------------------------------------------------
    # collision handler
    # -----------------------------------------------------
    @ti.func
    def next_hit(self, o, d):
        normal = ti.Vector([0., 0., 0.])
        color = ti.Vector([0., 0., 0.])
        closest = inf
        roughness = 0.05
        material = DIFFUSE  # diffuse

        # add background
        # if d[2] != 0:
        #     ray_closest = -(o[2] + 5.5) / d[2]
        #     # ray_closest = (0. - o[2])/d[2]
        #     if ray_closest > 0 and ray_closest < closest:  # and o[1] + d[1] * ray_closest <=1:
        #         closest = ray_closest
        #         normal = ti.Vector([0.0, 0.0, 1.0])
        #         color = ti.Vector([0.6, 0.7, 0.7])
        #         roughness = 0.0

        # add ground...
        # ground_dist = (o[1] - self.ground[None]) / (-d[1])
        ground_dist = .99
        # ti.ti_print(ground_dist)
        if ground_dist < dist_limit and ground_dist < closest:
            closest = ground_dist
            normal = ti.Vector([0., 1., 0.])
            color = self.ground_color(o + d * closest)
            roughness = 0.0
            material = DIFFUSE  # specular

        if ti.static(len(self.primitives) > 0):
            # copy from ray match
            j = 0
            dist = 0.0
            dist_z = 0.0
            sdf_val = inf
            sdf_id = 0

            pp = o
            while j < 200 and dist < dist_limit and sdf_val > 1e-8:
                pp = o + dist * d

                sdf_val = inf
                for i in ti.static(range(len(self.primitives))):
                    if self.visualize_primitive[i] > 0:
                        dd = ti.cast(self.primitives[i].sdf(0, pp), ti.f32)
                        if dd < sdf_val:
                            sdf_val = dd
                            sdf_id = i
                dist += sdf_val
                j += 1
            dist_z = self.engine.to_viewspace(pp).z
            if dist_z < closest and dist_z < dist_limit:
                closest = dist_z
                for i in ti.static(range(len(self.primitives))):
                    if sdf_id == i:
                        normal = ti.cast(self.primitives[i].normal(0, o + dist * d), ti.f32)
                        color = self.primitives[i].color
                roughness = 0.
                material = DIFFUSE

        # ------------------------ plasticine --------------------------------
        # shoot function
        if self.visualize_shape[None]:
            intersect, tnear, tfar = ray_aabb_intersection(self.bbox[0], self.bbox[1], o, d)

            if intersect:
                # ti.ti_print('bbox:', self.bbox[0], self.bbox[1])
                tnear = max(tnear, 0.)
                pos = o + d * (tnear + 1e-4)
                step = ti.Vector([0., 0., 0.])

                for j in range(500):
                    s = self.sample_sdf(pos)
                    if s < 0:
                        back_step = step
                        for k in range(20):
                            back_step = back_step * 0.5
                            if self.sample_sdf(pos - back_step) < 0:
                                pos -= back_step

                        dist_z = self.engine.to_viewspace(pos).z
                        if dist_z < closest:
                            closest = dist_z
                            normal = self.sample_normal(self.sample_sdf, pos)
                            color = self.sample_color(pos)
                            material = DIFFUSE
                        break
                    else:
                        step = d * max(s * 0.005, 0.002)
                        pos += step
        # ti.ti_print('closest:', closest)
        # if self.visualize_target[None]:
        #     # ------------------------ target density ----------------------------
        #     intersect, tnear, tfar = ray_aabb_intersection(ti.Vector([0.0, 0.0, 0.0]), ti.Vector([1.0, 1.0, 1.0]), o, d)
        #     if intersect:
        #         tnear = max(tnear, 0.)
        #         pos = o + d * (tnear + 1e-4)
        #         step = ti.Vector([0., 0., 0.])
        #         total_forward = 0.0
        #
        #         for j in range(500):
        #             if total_forward + tnear > tfar:
        #                 break
        #             s = self.sample_target_density(pos)
        #             if s < 0:
        #                 back_step = step
        #                 for k in range(20):
        #                     back_step = back_step * 0.5
        #                     if self.sample_target_density(pos - back_step) < 0:
        #                         pos -= back_step
        #
        #                 dist = (o - pos).norm()
        #                 if dist < closest:
        #                     closest = dist
        #                     normal = self.sample_normal(self.sample_target_density, pos)
        #                     color = self.target_density_color
        #                     material = DIFFUSE
        #                 break
        #             else:
        #                 step_length = (1.0 / self.target_res[0])
        #                 step = d * step_length
        #                 total_forward += step_length
        #                 pos += step

        return closest, normal, color, roughness, material

    # -----------------------------------------------------
    # ray tracing
    # -----------------------------------------------------
    @ti.func
    def sample_sphere(self):
        u = ti.random(ti.f32)
        v = ti.random(ti.f32)
        x = u * 2 - 1
        phi = v * 2 * 3.14159265358979
        yz = ti.sqrt(1 - x * x)
        return ti.Vector([x, yz * ti.cos(phi), yz * ti.sin(phi)])

    # @ti.func
    # def sky_color(self, direction):
    #     coeff1 = direction.dot(ti.Vector([0.8, 0.65, 0.15])) * 0.5 + 0.5
    #     coeff1 = ti.max(ti.min(coeff1, 1.), 0.)
    #     light = coeff1 * ti.Vector([0.9, 0.9, 0.9]) + (1. - coeff1) * ti.Vector([0.7, 0.7, 0.8])
    #     return light * 1.5

    # @ti.kernel
    # def copy(self, img: ti.ext_arr(), samples: ti.i32):
    #     for i, j in self.color_buffer:
    #         u = 1.0 * i / self.image_res[0]
    #         v = 1.0 * j / self.image_res[1]
    #
    #         darken = 1.0 - self.vignette_strength * max((ti.sqrt(
    #             (u - self.vignette_center[0]) ** 2 +
    #             (v - self.vignette_center[1]) ** 2) - self.vignette_radius), 0)
    #
    #         for c in ti.static(range(3)):
    #             img[i, j, c] = ti.sqrt(self.color_buffer[i, j][c] * darken *
    #                                    exposure / samples)
    #         img[i, j, 3] = ti.sqrt(self.depth_buffer[i, j] / samples)

    # @ti.kernel
    # def render(self):
    #     ti.block_dim(128)
    #     # print(self.sample_sdf(self.bbox[0] + 0.05))
    #     # return
    #     mat = ti.Matrix([
    #         [ti.cos(self.camera_rot[None][1]), 0.0000000, ti.sin(self.camera_rot[None][1])],
    #         [0.0000000, 1.0000000, 0.0000000],
    #         [-ti.sin(self.camera_rot[None][1]), 0.0000000, ti.cos(self.camera_rot[None][1])],
    #     ]) @ ti.Matrix([
    #         [1.0000000, 0.0000000, 0.0000000],
    #         [0.0000000, ti.cos(self.camera_rot[None][0]), ti.sin(self.camera_rot[None][0])],
    #         [0.0000000, -ti.sin(self.camera_rot[None][0]), ti.cos(self.camera_rot[None][0])],
    #     ])
    #     for u, v in self.color_buffer:
    #         pos = self.camera_pos[None]
    #         d = ti.Vector([
    #             (2 * fov * (u + ti.random(ti.f32)) / self.image_res[1] -
    #              fov * self.aspect_ratio - 1e-5),
    #             2 * fov * (v + ti.random(ti.f32)) / self.image_res[1] - fov - 1e-5, -1.0
    #         ])
    #         d = mat @ d.normalized()
    #         contrib = self.trace(pos, d)
    #         depth = self.next_hit(pos, d)[0]
    #         self.color_buffer[u, v] += contrib
    #         self.depth_buffer[u, v] += depth

    @ti.kernel
    def initialize_particles_kernel(self):
        self.bbox[0] = [inf, inf, inf]
        self.bbox[1] = [-inf, -inf, -inf]
        for i in range(self.npars[None]):
            for c in ti.static(range(3)):
                v = (ti.floor(self.particle_x[i][c] * self.inv_dx) - 6.0) * self.dx
                ti.atomic_min(self.bbox[0][c], v)
                ti.atomic_max(self.bbox[1][c], v)
        # ti.ti_print(self.bbox[0], self.bbox[1])

    def plb_set_particles(self):
        # assume that num_part and particle_x is all calculated ...
        self.initialize_particles_kernel()
        # update box..
        bbox = self.bbox.to_numpy()
        desired_res = (bbox[1] - bbox[0]) / self.dx
        for a, b in zip(desired_res, self.voxel_res):
            assert a < b, f"the sdf {bbox} should be smaller {desired_res} < {self.voxel_res}"
        bbox[1] = bbox[0] + np.array(self.voxel_res) * self.dx
        self.bbox.from_numpy(bbox)

        # reset ..
        self.volume.fill(int((1 << 31) - 1))
        self.build_sdf_from_particles()

    # def render_frame(self, spp=None, **kwargs):
    #     if spp is None:
    #         spp = self.spp
    #
    #     last_t = 0
    #     visualize_target = kwargs.get('target', 1)
    #     target_interval = round(1 / kwargs.get('target_opacity', 0.5))  # opacity
    #     self.visualize_shape[None] = kwargs.get('shape', 1)
    #     self.visualize_primitive[None] = kwargs.get('primitive', 1)
    #     self.color_buffer.fill(0)
    #     self.depth_buffer.fill(0)
    #
    #     for i in range(1, 1 + spp):
    #         # Opacity=50%
    #         self.visualize_target[None] = int(i % target_interval == 0) * visualize_target
    #         self.render()
    #
    #         interval = 20
    #         if i % interval == 0:
    #             if last_t != 0:
    #                 ti.sync()
    #             last_t = time.time()
    #
    #     img = np.zeros((self.image_res[0], self.image_res[1], 4), dtype=np.float32)
    #     self.copy(img, spp)
    #     img[:, :, :3] = np.clip(img[:, :, :3], 0., 1.)
    #     return img[:, ::-1].transpose(1, 0, 2)  # opencv format for render..

    @ti.kernel
    def smooth_target_density(self):
        for I in ti.grouped(self.target_density):
            self.target_density2[I] = -self.target_density2[I] + 3
        self.smooth(self.target_density2, self.target_density, self.target_res)

    @ti.kernel
    def fill_target_density(self, val: ti.f32):
        for I in ti.grouped(self.target_density):
            self.target_density[I] = val
            self.target_density2[I] = val

    def set_target_density(self, target_density=None):
        if target_density is not None:
            self.target_density2.from_numpy(target_density.astype(np.float32))
            self.smooth_target_density()
        else:
            self.fill_target_density(0.0)

    def initialize(self, cfg):
        self.camera_pos.from_numpy(np.array([float(i) for i in cfg.camera_pos]))
        self.camera_rot.from_numpy(np.array(cfg.camera_rot))
        print('initialize:', self.camera_pos, self.camera_rot)
