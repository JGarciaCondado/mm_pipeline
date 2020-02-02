import shapely.geometry as geo
import matplotlib.pyplot as plt
import numpy as np

r_values = np.arange(0.05, 2.0, 0.1)
r_conv = []
dx = 0.01
point = geo.Point(0.0, 0.0)
for r in r_values:
    circle = point.buffer(r)
    [minx, miny, maxx, maxy] = circle.bounds
    d_conv = 0
    for i in range(10):
        i=0
        samples = []
        while(i<10000):
            sample = geo.Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if sample.within(circle):
                samples.append(sample)
                i += 1
        samples_x = [sample.x for sample in samples]
        samples_y = [sample.y for sample in samples]
        bins = np.arange(-r, r, dx)
        bin_centers = np.arange(-r+dx/2, r, dx)
        hist, bin_edges = np.histogram(samples_x, bins=bins)
        sigma = 0.177
        gx = np.arange(-3*sigma, 3*sigma, dx)
        gaussian = np.exp(-(gx/sigma)**2/2)
        result = np.convolve(hist, gaussian, mode="full")
        result = result*np.sum(hist)/np.sum(result)
        threshold = np.max(result)*0.03
        new_x = np.arange(-r+dx/2-3*sigma, r+3*sigma, dx)
        [[low_limit], [upper_limit]] = np.argwhere(result > threshold)[[0,-1]]
        d_conv += new_x[upper_limit+1] - new_x[low_limit-1]
    r_conv.append(d_conv/10)

plt.plot(r_values*2, r_conv-r_values*2)
plt.xlabel("2R")
plt.ylabel("2R'-2R")
plt.show()
plt.plot(r_values*2, r_conv)
plt.plot(np.arange(0.0, 4.5, 0.5), np.arange(0.0, 4.5, 0.5))
plt.xlabel("2R")
plt.ylabel("2R'")
plt.show()
