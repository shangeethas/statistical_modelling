#libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

R = np.arange(-4, 4, 0.1)
X, Y = np.meshgrid(R, R)

Z = np.sum(np.exp(-0.5 * (X**2 + Y**2)))
P = (1/Z) * np.exp(-0.5 * (X**2 + Y**2))

invalid_xy = (X**2 + Y**2) < 1
P[invalid_xy] = 0

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X, Y, P, s=0.5, alpha=0.5)
plt.show()
