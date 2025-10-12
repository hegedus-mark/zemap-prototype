import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon

# --- Step 1: Flat hex map setup ---
width, height = 12, 6      # number of hexes in x and y
R_hex = 1.0                 # hex “radius” (distance from center to corner)

# Axial coordinate system for hexes
def hex_corners(q, r, radius=1.0):
    """Return 6 corners of a flat hex at axial coordinates q,r."""
    x_offset = radius * 3/2 * q
    y_offset = radius * np.sqrt(3) * (r + 0.5 * (q % 2))
    corners = []
    for i in range(6):
        angle = np.pi/3 * i
        x = x_offset + radius * np.cos(angle)
        y = y_offset + radius * np.sin(angle)
        corners.append((x, y))
    return corners

# --- Step 2: Project flat coordinates to sphere ---
R_sphere = 5.0  # sphere radius

def project_to_sphere(x, y, width, height, R=1.0):
    """Convert flat map (x, y) to spherical coordinates (x, y, z)."""
    lon = x / (width * 3/2 * R_hex) * 2*np.pi - np.pi       # -180° to 180°
    lat = np.pi/2 - y / (height * np.sqrt(3) * R_hex) * np.pi  # 90° to -90°
    xs = R * np.cos(lat) * np.cos(lon)
    ys = R * np.cos(lat) * np.sin(lon)
    zs = R * np.sin(lat)
    return np.array([xs, ys, zs])

# --- Step 3: Generate hexes ---
flat_hexes = []
sphere_hexes = []

for r in range(height):
    for q in range(width):
        corners = hex_corners(q, r, R_hex)
        flat_hexes.append(corners)
        # Project each corner to sphere
        sphere_corners = [project_to_sphere(x, y, width, height, R_sphere) for x, y in corners]
        sphere_hexes.append(sphere_corners)

# --- Step 4: Plotting ---
fig = plt.figure(figsize=(12, 6))

# Flat view
ax1 = fig.add_subplot(121)
for hex_c in flat_hexes:
    hex_poly = Polygon(hex_c, edgecolor='k', facecolor='skyblue', alpha=0.6)
    ax1.add_patch(hex_poly)
ax1.set_aspect('equal')
ax1.set_title('Flat Hex Map')
ax1.autoscale_view()

# Sphere view
ax2 = fig.add_subplot(122, projection='3d')
for hex_c in sphere_hexes:
    poly3d = Poly3DCollection([hex_c], facecolor='skyblue', edgecolor='k', alpha=0.6)
    ax2.add_collection3d(poly3d)
ax2.set_box_aspect([1,1,1])
ax2.auto_scale_xyz([-R_sphere, R_sphere], [-R_sphere, R_sphere], [-R_sphere, R_sphere])
ax2.set_title('Sphere Projection')

plt.show()
