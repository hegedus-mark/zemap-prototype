import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi
from mpl_toolkits.mplot3d import Axes3D

# --- Parameters ---
world_size = 1000
num_large_continents = 3
num_small_islands = 12
sphere_radius = 1

# --- Generate large continent seeds (clustered Gaussian) ---
large_centers = np.random.rand(num_large_continents, 2) * world_size
large_points = []
for center in large_centers:
    point = center + np.random.randn(1, 2) * (world_size / 10)
    large_points.append(point)
large_points = np.vstack(large_points)

# --- Generate small island seeds (skewed distribution) ---
x = np.random.rand(num_small_islands)**1.5 * world_size
y = np.random.rand(num_small_islands)**1.5 * world_size
small_points = np.column_stack((x, y))

# --- Combine seeds ---
points_2d = np.vstack((large_points, small_points))

# --- Convert 2D points to spherical coordinates ---
# Map 0..world_size to lat/lon
lon = (points_2d[:, 0] / world_size) * 360 - 180   # -180 to 180
lat = (points_2d[:, 1] / world_size) * 180 - 90    # -90 to 90

# Convert lat/lon to 3D unit sphere coordinates
lon_rad = np.radians(lon)
lat_rad = np.radians(lat)
x3d = sphere_radius * np.cos(lat_rad) * np.cos(lon_rad)
y3d = sphere_radius * np.cos(lat_rad) * np.sin(lon_rad)
z3d = sphere_radius * np.sin(lat_rad)
points_3d = np.column_stack((x3d, y3d, z3d))

# --- Compute spherical Voronoi ---
sv = SphericalVoronoi(points_3d, radius=sphere_radius, center=[0,0,0])
sv.sort_vertices_of_regions()

# --- Plot the sphere with Voronoi edges ---
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Voronoi edges
for region in sv.regions:
    vertices = sv.vertices[region]
    ax.plot(vertices[:,0], vertices[:,1], vertices[:,2], 'k-')

# Plot seeds
ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], color='red', s=50)

ax.set_box_aspect([1,1,1])
ax.set_title("Spherical Voronoi World (3D)")
plt.show()

# --- Project sphere to 2D (equirectangular projection) ---
x2d = np.degrees(np.arctan2(y3d, x3d))   # longitude
y2d = np.degrees(np.arcsin(z3d / sphere_radius))  # latitude

fig, ax = plt.subplots(figsize=(12,6))

# Plot Voronoi edges projected to 2D
for region in sv.regions:
    vertices = sv.vertices[region]
    # Project vertices to 2D
    lon_proj = np.degrees(np.arctan2(vertices[:,1], vertices[:,0]))
    lat_proj = np.degrees(np.arcsin(vertices[:,2] / sphere_radius))
    # Close polygon loop
    lon_proj = np.append(lon_proj, lon_proj[0])
    lat_proj = np.append(lat_proj, lat_proj[0])
    ax.plot(lon_proj, lat_proj, 'k-')

# Plot seeds
ax.scatter(x2d, y2d, color='red', s=50)

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Procedural World Map (Equirectangular Projection)")
plt.show()
