import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a, b, c = 1, 1, 0.95



theta = np.linspace(0, np.pi, 40)
phi = np.linspace(0, 2*np.pi, 80)
theta, phi = np.meshgrid(theta, phi)

x = a * np.sin(theta) * np.cos(phi)
y = b * np.sin(theta) * np.sin(phi)
z = c * np.cos(theta)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='c', edgecolor='k')  # grid lines
ax.set_box_aspect([a, b, c]) 
plt.show()
