import taichi as ti
import tina
import numpy as np

ti.init(ti.cuda)
scene = tina.Scene(res_x=1024, smoothing=True, texturing=True, taa=True)

pars = tina.SimpleParticles()
pars2 = tina.SimpleParticles()
material = tina.Diffuse()
pc = np.load('/home/xingyu/Projects/PlasticineLab/data/particles.npy')

pc[:, 1] -= np.min(pc[:, 1])
pc[:, 1] += 1 / 64.
# pc -= np.mean(pc, axis=0)
# pc += np.array((0.5, 0.5, 0.5)).reshape(1, 3)  # Normalize to 0
pars.set_particles(pc)
pars2.set_particles(pc)
scene.add_object(pars, material, raster='particle_sdf')
# scene.add_object(pars2, material)

scene.lighting.set_ambient_light([0.1, 0.1, 0.1])

gui = tina.ti.GUI('primitives', res=scene.res)
scene.init_control(gui, center=(0.5, 0, 0.5),
                   theta=3.14/3., phi=3.14/3., radius=1.)
while gui.running:
    scene.input(gui, refresh=True)

    scene.render()
    gui.set_image(scene.img)
    gui.show()
