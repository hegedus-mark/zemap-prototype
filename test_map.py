import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

R = 5        # radius of sphere
layers = 10   # number of latitude layers
width = 0.8  # square width
height = 0.8 # square height

def create_square_facing_out(center, size=1.0):
    x, y, z = center
    # Normal vector pointing out
    n = np.array([x, y, z])
    n = n / np.linalg.norm(n)  # normalize

    # Find two perpendicular vectors for the square plane
    # Take an arbitrary up vector
    up = np.array([0, 0, 1])
    if np.allclose(n, up):  # avoid colinear
        up = np.array([0, 1, 0])

    # right vector = cross(up, normal)
    right = np.cross(up, n)
    right /= np.linalg.norm(right)

    # corrected up vector = cross(normal, right)
    up_corrected = np.cross(n, right)

    # Square corners in local space
    hw = size / 2
    corners = (
        -hw*right - hw*up_corrected,
         hw*right - hw*up_corrected,
         hw*right + hw*up_corrected,
        -hw*right + hw*up_corrected
    )
    corners = np.array(corners) + center
    return corners

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(layers):
    phi = np.pi * (i + 0.5) / layers
    N = int(2 * layers * np.sin(phi) + 4)
    for j in range(N):
        theta = 2 * np.pi * j / N
        x = R * np.sin(phi) * np.cos(theta)
        y = R * np.sin(phi) * np.sin(theta)
        z = R * np.cos(phi)

        square = create_square_facing_out((x, y, z), width)
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