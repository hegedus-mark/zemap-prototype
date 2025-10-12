import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict

# Create a geodesic sphere
subdivisions = 2  # increase for higher resolution
radius = 1.0
geo = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

vertices = geo.vertices
faces = geo.faces

# Step 1: Count how many faces each vertex belongs to
vertex_faces = defaultdict(list)
for i, f in enumerate(faces):
    for v in f:
        vertex_faces[v].append(i)

vertex_counts = {v: len(fs) for v, fs in vertex_faces.items()}

# Step 2: Identify pentagon vertices (connected to 5 triangles)
pentagon_vertices = set(v for v, count in vertex_counts.items() if count == 5)

# Step 3: Assign each triangle to pentagon or hexagon
triangle_colors = []
for f in faces:
    # If any vertex in the triangle is a pentagon vertex, color it red
    if any(v in pentagon_vertices for v in f):
        triangle_colors.append('red')  # pentagon
    else:
        triangle_colors.append('blue')  # hexagon

# Step 4: Plot triangles
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for f, color in zip(faces, triangle_colors):
    tri_verts = vertices[f]
    tri = Poly3DCollection([tri_verts], facecolor=color, edgecolor='k')
    ax.add_collection3d(tri)

ax.set_box_aspect([1, 1, 1])
ax.auto_scale_xyz([-radius, radius], [-radius, radius], [-radius, radius])
plt.show()
