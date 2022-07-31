import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

grid = np.load('datasets/0706_PushSpread/target/target_18.npy')
# grid[20:30, 20:30, 20:30] = 1
grid[np.where(grid > 0)] = 1.
grid = np.pad(grid, 2)
if False:
    from skimage import measure

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(grid, method='lorensen')
    print(verts.shape)
    print(faces.shape)
    print(normals.shape)
else:
    from marching_cubes import march

    # extract the mesh where the values are larger than or equal to 1
    # everything else is ignored
    verts, normals, faces = march(grid, 1)  # zero smoothing rounds
    smooth_vertices, smooth_normals, faces = march(grid, 3)  # 3 smoothing rounds

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis: a = 6 per ellipsoid")
ax.set_ylabel("y-axis: b = 10")
ax.set_zlabel("z-axis: c = 16")

ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(0, 20)  # b = 10
ax.set_zlim(0, 32)  # c = 16

plt.tight_layout()
plt.show()
