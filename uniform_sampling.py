from bacteria_model import Fluorescent_bacteria_spline_fn, SpherocylindricalBacteria
from shapely.geometry import LineString
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
R = 2
l = 3
r = 0.5
#theta_max = np.pi/2 - np.arccos(l/(2*R))
#thetas = np.random.uniform(-theta_max, theta_max, 1000)
#phis = np.random.uniform(0, 2*np.pi, 1000)
#ps = r*np.sqrt(np.random.uniform(1, 1, 1000))
#x = np.multiply(np.sin(thetas), R+np.multiply(ps, np.cos(phis)))
#y = np.multiply(np.cos(thetas), R+np.multiply(ps, np.cos(phis)))-R
#z = np.multiply(ps, np.sin(phis))

def f(x):
    return np.sqrt(R**2-x**2)-R

x_max, x_min = l/2+r, -l/2-r
y_max, y_min = r, f(l/2)-r
z_max, z_min = r, -r
phi_max, phi_min = np.arctan((l/2)/(R+f(l/2))), np.arctan((-l/2)/(R+f(-l/2)))

samples = []
for i in range(10000):
    # lengthscale is alawys gonna be bigger
    x_sample = np.random.uniform(x_min, x_max)
    y_sample = np.random.uniform(y_min, y_max)
    z_sample = np.random.uniform(z_min, z_max)
    sample = np.array([x_sample, y_sample, z_sample])
    phi = np.arctan(x_sample/(R+y_sample))
    x_int = R*np.sin(phi)
    y_int = R*(np.cos(phi)-1)
    if phi < phi_min:
        if np.linalg.norm(sample-np.array([-l/2, f(-l/2), 0.0])) < r:
            samples.append(sample)
    elif phi > phi_max:
        if np.linalg.norm(sample-np.array([l/2, f(l/2), 0.0])) < r:
            samples.append(sample)
    else:
        if np.linalg.norm(sample-np.array([x_int, y_int, 0.0])) < r:
            samples.append(sample)
xsamples, ysamples, zsamples = list(zip(*samples))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xsamples, ysamples, zsamples, s=1, c='r', marker='o')
dx = 0.01
x = np.arange(-l/2, l/2+dx, dx)
ax.plot(x, [f(x_i) for x_i in x])
#plt.scatter(x, y)
plt.axis('scaled')
plt.show()
spline = LineString([(x_i, f(x_i)) for x_i in x])
boundary = spline.buffer(r)
x_b, y_b = list(zip(*list(boundary.exterior.coords)))
plt.plot(x_b, y_b)
plt.scatter(xsamples, ysamples)
plt.plot(x, [f(x_i) for x_i in x], c='k')
plt.axis('scaled')
plt.show()
#bacteria = SpherocylindricalBacteria(r, l, R, 0, 0, 0, 50)
#bacteria.plot_3D()
#bacteria.plot_2D()
