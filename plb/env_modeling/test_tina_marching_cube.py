import taichi as ti
import tina
import numpy as np


def get_mesh(grid):
    from marching_cubes import march
    # extract the mesh where the values are larger than or equal to 1
    # everything else is ignored
    verts, normals, faces = march(grid, 2)  # zero smoothing rounds
    smooth_vertices, smooth_normals, faces = march(grid, 3)  # 3 smoothing rounds
    m = np.mean(verts, axis=0)
    verts -= m
    return verts, faces, normals


def get_tina_faces(verts, faces, normals):
    nf = faces.shape[0]
    tina_verts = verts[faces.flatten()].reshape(nf, 3, 3)
    tina_normals = normals[faces.flatten()].reshape(nf, 3, 3)
    tina_coors = np.zeros_like(tina_verts)
    tina_face = np.concatenate([tina_verts[:, :, None, :], tina_normals[:, :, None, :], tina_coors[:, :, None, :]], axis=2)
    return tina_face


grid = np.load('/home/xingyu/Projects/PlasticineLab/datasets/0706_PushSpread/target/target_19.npy')
grid = np.pad(grid, 5)
grid[np.where(grid > 0)] = 1.

ti.init(ti.cuda)
scene = tina.Scene(res_x=1024, smoothing=True, texturing=True, taa=True)

verts, faces, normals = get_mesh(grid)
tina_face = get_tina_faces(verts, faces, normals)
mesh = tina.PrimitiveMesh(tina_face)
# mesh = tina.PrimitiveMesh.sphere()
# mesh = tina.PrimitiveMesh.cylinder()
scene.add_object(mesh, material=tina.Diffuse(color=(255 / 255., 250 / 255., 231 / 255.)))
scene.lighting.set_ambient_light([0.1, 0.1, 0.1])
# scene.lighting.add_light(dir=[0, 0, 1], color=[1, 1, 1])
# scene.lighting.add_light(dir=[0, 1, 0], color=[1, 1, 1])
# scene.lighting.add_light(dir=[1, 0, 0], color=[1, 1, 1])
gui = tina.ti.GUI('primitives', res=scene.res)

while gui.running:
    scene.input(gui)
    scene.render()
    # verts, faces, normals = get_mesh(grid)
    # tina_face = get_tina_faces(verts, faces, normals)
    # mesh = tina.PrimitiveMesh(tina_face)
    # mesh = tina.PrimitiveMesh.sphere()
    # mesh = tina.PrimitiveMesh.cylinder()
    # scene.add_object(mesh, material=tina.Diffuse(color=(255 / 255., 250 / 255., 231 / 255.)))
    gui.set_image(scene.img)
    gui.show()
