import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def create_square_facing_out(center, size=1.0):
    """
    Create a square centered at 'center' facing outward from the origin.
    """
    x, y, z = center
    n = np.array([x, y, z])
    n = n / np.linalg.norm(n)  # normalize normal

    # Choose an arbitrary up vector
    up = np.array([0, 0, 1])
    if np.allclose(n, up):
        up = np.array([0, 1, 0])

    right = np.cross(up, n)
    right /= np.linalg.norm(right)
    up_corrected = np.cross(n, right)

    hw = size / 2
    corners = (
        -hw*right - hw*up_corrected,
         hw*right - hw*up_corrected,
         hw*right + hw*up_corrected,
        -hw*right + hw*up_corrected
    )
    corners = np.array(corners) + center
    return corners

# Parameters
R = 5          # radius of sphere
cols = 16      # number of squares along equator
rows = 8       # number of squares from pole to pole
equator_size = 1.0  # square size at equator

# Compute vertical size to match equator squares (roughly)
vertical_size = np.pi * R / rows  # arc length per row
horizontal_size = 2 * np.pi * R / cols

# Pick the smaller to avoid gaps
square_size = min(equator_size, horizontal_size, vertical_size)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for r in range(rows):
    phi = np.arccos(1 - 2*(r + 0.5)/rows)  # geodesic-ish latitude

    # compute number of columns for this row
    N = max(1, round(cols * np.sin(phi)))  # at least 1 to avoid zero
    for c in range(N):
        theta = 2 * np.pi * c / N
        x = R * np.sin(phi) * np.cos(theta)
        y = R * np.sin(phi) * np.sin(theta)
        z = R * np.cos(phi)

        square = create_square_facing_out((x, y, z), square_size)
        poly = Poly3DCollection([square], facecolor='cyan', edgecolor='black', alpha=0.7)
        ax.add_collection3d(poly)

# Set limits
ax.set_xlim(-R-2, R+2)
ax.set_ylim(-R-2, R+2)
ax.set_zlim(-R-2, R+2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1,1,1])

plt.show()