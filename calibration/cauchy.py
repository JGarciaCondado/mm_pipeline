import matplotlib.pyplot as plt
import numpy as np
from rmatching import convolve_cauchy, optimum_cauchy, model_cauchy, model, optimum_r_and_psf

dist = np.load("px.npy")
rc, gc, hc = optimum_cauchy(dist)
rg, gg, hg = optimum_r_and_psf(dist)
bins = np.arange(-0.5, len(dist))
print(rc,gc,hc)
print(rg,gg,hg)
plt.plot(np.arange(0, 26, 0.01), model_cauchy(np.arange(0,26, 0.01), rc, hc, gc))
plt.plot(np.arange(0, 26, 0.01), model(np.arange(0,26, 0.01), rg, hg, gg))
plt.hist(bins[:-1], bins, weights=dist)
plt.show()
