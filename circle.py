import shapely.geometry as geo
import matplotlib.pyplot as plt
import numpy as np

r = 0.5
dx = 0.01
point = geo.Point(0.0, 0.0)
circle = point.buffer(r)
[minx, miny, maxx, maxy] = circle.bounds
i=0
samples = []
while(i<10000):
    sample = geo.Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
    if sample.within(circle):
        samples.append(sample)
        i += 1
samples_x = [sample.x for sample in samples]
samples_y = [sample.y for sample in samples]
fig, axs = plt.subplots(2, sharex=True)
axs[0].scatter(samples_x, samples_y, s=1)
axs[0].plot(*circle.exterior.xy)
bins = np.arange(-r, r+dx, dx)
bin_centers = np.arange(-r+dx/2, r, dx)
hist, bin_edges = np.histogram(samples_x, bins=bins)
axs[1].hist(samples_x, bins=bin_edges)
sigma = 0.177
gx = np.arange(-3*sigma, 3*sigma, dx)
gaussian = np.exp(-(gx/sigma)**2/2)
result = np.convolve(hist, gaussian, mode="full")
result = result*np.sum(hist)/np.sum(result)
new_x = np.arange(-r+dx/2-3*sigma, r+3*sigma, dx)
axs[1].plot(new_x, result)
plt.show()
