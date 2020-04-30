import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from rmatching import circle_px, discretized_circle_px, convolved_circle_px, optimum_r_and_psf


#Constants
image_size = 26
r_c = 3.5
h = 16.7 #center of circel in x-direction
sigma = 4.2

#Uniform sampling within circle
r = r_c*np.sqrt(np.random.uniform(0,1,10000))
theta = np.random.uniform(0, 2*np.pi, 10000)
x = r*np.cos(theta) + h
y = r*np.sin(theta)
plt.scatter(x, y)
plt.title("Sampling a circle with radius {} and center at x = {} ".format(r_c, h))
plt.axis('scaled')
plt.show()

#Plot samples and px
xt = np.arange(-r_c+h, r_c+h, 0.01)
plt.xlabel("x (in pixels)")
plt.ylabel("P(x)")
plt.plot(xt, [circle_px(x, r_c, h) for x in xt], label="Analytical")
plt.hist(x, bins=20, density=True, label="Samples")
plt.legend()
plt.show()

#Plot discretized px and convolution
bins_circle, counts = discretized_circle_px(r_c, h, image_size)
bins_convolution, convolution = convolved_circle_px(r_c, h, sigma, image_size)
plt.xlabel("x (in pixels)")
plt.ylabel("P(x)")
x_resample = np.random.normal(x, sigma)
plt.hist(bins_circle[:-1], bins_circle, weights=counts, label = "Discretized p(x)")
plt.hist(bins_convolution[:-1], bins_convolution, weights=convolution, label = "p(y) by convolution", alpha=0.85)
plt.hist(x_resample, bins = bins_convolution, density = True, label = "Sampled p(y)", alpha=0.5)
x = np.arange(0, image_size, 0.1)
g_x = np.exp(-(x-h)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
plt.plot(x, g_x, label="Gaussian")
spl = splrep(range(len(convolution)), convolution)
x2 = np.linspace(0, len(convolution), 100)
y2 = splev(x2, spl)
plt.plot(x2, y2, label = "Spline approximation of p(y)")
plt.legend()
plt.show()

#Test optimum finder
#TODO use this kind of thinking for unit test
print("Distribution r: {}".format(r_c))
print("Distribution sigma: {}".format(sigma))
print("Distribution h: {}".format(h))
opt_r, opt_sigma, opt_h = optimum_r_and_psf(convolution, np.arange(3.0, 10, 0.1), np.arange(2.0, 5.0, 0.1))
#print("MSE: {}".format(mse))
print("Optimum r: {}".format(opt_r))
print("Optimum sigma: {}".format(opt_sigma))
print("Optimum h: {}".format(opt_h))
plt.xlabel("x (in pixels)")
plt.ylabel("P(x)")
plt.hist(bins_convolution[:-1], bins_convolution, weights=convolution, label = "True p(x)")
bins_opt, opt = convolved_circle_px(opt_r, opt_h, opt_sigma, image_size)
plt.hist(bins_opt[:-1], bins_opt, weights=opt, label = "Optimum p(x)", alpha=0.65)
plt.legend()
plt.show()
