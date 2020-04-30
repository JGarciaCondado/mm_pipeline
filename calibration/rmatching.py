import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit

def circle_px(x, r, h):
    if abs(x-h) >= r:
        return 0.0
    Z = np.pi*r**2/2
    return np.sqrt(r**2-(x-h)**2)/Z

def discretized_circle_px(r, h, n_bins):
    bin_edges = np.arange(int(n_bins+1))-0.5
    counts = [cumulative_dist(bin_edges[i+1], r, h)-cumulative_dist(bin_edges[i], r, h) for i in range(len(bin_edges)-1)]
    return bin_edges, counts

def cumulative_dist(x, r, h):
    if x-h <= -r:
        return 0.0
    elif x-h >= r:
        return 1.0
    else:
        #Normalization constant
        Z  = np.pi*r**2/2
        return ((x-h)*np.sqrt(r**2-(x-h)**2)/2+r**2*np.arcsin((x-h)/r)/2 - \
           (r**2*-np.pi/4))/Z


def convolved_circle_px(r, h, sigma, n_bins):
    #TODO doesnt work well when part of the circle is outside the image i.e. r+h or r-h is smaller
    # number of bins. This is because discretized_circle_px doesnt give back full circle px -> exapnds
    # so that it always give full px even if it has to give higher bins -> maybe convolution of 
    # of px and gaussian by moving in h direction? conolve (x-h) for example


    #TODO above can be solved as well by making p(x) circle around and the gaussian being moved by h
    # this could solve discretization problem as well
    bin_edges, counts = discretized_circle_px(r, h, n_bins)
    g_x = np.exp(-np.arange(-int(n_bins), int(n_bins)+1)**2/(2*sigma**2))
    convolution = np.convolve(counts, g_x)[n_bins:n_bins*2]
    #Normalize to get a distribution
    convolution = convolution/np.sum(convolution)
    return bin_edges, convolution

def compare_distributions(distribution, r, h, sigma):
    bin_edges, convolution = convolved_circle_px(r, h, sigma, len(distribution))
    MSE = np.mean((distribution - convolution)**2)
    return MSE

def match_r_and_psf(distribution, r_range, sigma_range):
    """ Grid search for r and psf """
    # The center of the circle is in the peak of the distribution
    # Use the fact that it is a concave function
#    dist_diff = np.diff(distribution)
    # Only first zero crossing is of interes
#    zero_crossing = np.where(np.diff(np.signbit(dist_diff)))[0][0]
#    m = dist_diff[zero_crossing+1] - dist_diff[zero_crossing]
#    c = dist_diff[zero_crossing] - m*(zero_crossing+0.5)
#    h = - c / m
    #TODO check that there is only one peak by number of roots
    spl = splrep(range(len(distribution)), distribution)
    x2 = np.linspace(0, len(distribution), 1000)
    y2 = splev(x2, spl)
    h = x2[np.argmax(y2)]
    #TODO make faster by not recalculating the discretized px or gx for each r and sigma
    # i.e. can do this in terms of Linear algebra with matrices
    min_mse = 0
    optimum_r = 0.0
    optimum_sigma = 0.0
    for r in r_range:
        for sigma in sigma_range:
            mse = compare_distributions(distribution, r, h, sigma)
            if 1/mse > min_mse:
                 min_mse = 1/mse
                 optimum_r = r
                 optimum_sigma = sigma

    return optimum_r, optimum_sigma, 1/min_mse, h


def model_spline(r, h, sigma):
    bins, conv = convolved_circle_px(r,h,sigma, 26)
    x = np.arange(0, 26)
    return splrep(x, conv)

def model(x, r, h, sigma):
    spl = model_spline(r, h, sigma)
    y = splev(x, spl)
    return y

def optimum_r_and_psf(distribution):
    #TODO check that there is only one peak by number of roots
    spl = splrep(range(len(distribution)), distribution)
    x2 = np.linspace(0, len(distribution), 1000)
    y2 = splev(x2, spl)
    h = x2[np.argmax(y2)]

    #TODO initial values
    popt, pcov = curve_fit(model, range(len(distribution)), distribution, p0 = [5, h, 2], bounds= ([4, 0, 1], [7, len(distribution), 5]))
    opt_r, opt_h, opt_sigma = popt

    return opt_r, opt_sigma, opt_h

def cauchy(x, gamma, x0):
    return 1/(np.pi*gamma) * (gamma**2/((x-x0)**2+gamma**2))

def convolve_cauchy(r, h, gamma, n_bins):
    bin_edges, counts = discretized_circle_px(r, h, n_bins)
    cx = cauchy(np.arange(-int(n_bins), int(n_bins)+1), gamma, 0)
    convolution = np.convolve(counts, cx)[n_bins:n_bins*2]
    #Normalize to get a distribution
    convolution = convolution/np.sum(convolution)
    return bin_edges, convolution

def model_spline_cauchy(r, h, gamma):
    bins, conv = convolve_cauchy(r,h,gamma, 26)
    x = np.arange(0, 26)
    return splrep(x, conv)

def model_cauchy(x, r, h, gamma):
    spl = model_spline_cauchy(r, h, gamma)
    y = splev(x, spl)
    return y

def optimum_cauchy(distribution):
    #TODO check that there is only one peak by number of roots
    spl = splrep(range(len(distribution)), distribution)
    x2 = np.linspace(0, len(distribution), 1000)
    y2 = splev(x2, spl)
    h = x2[np.argmax(y2)]

    #TODO initial values
    popt, pcov = curve_fit(model_cauchy, range(len(distribution)), distribution, p0 = [5, h, 2], bounds= ([1, 10, 1], [7, 20, 5]))
    opt_r, opt_h, opt_gamma = popt

    return opt_r, opt_gamma, opt_h

def circle_pdf(x, r, h):
    px = np.zeros(x.shape)
    Z = np.pi*r**2/2
    px[np.argwhere(np.abs(x-h) < r)] = np.sqrt(r**2-(x[np.argwhere(np.abs(x-h) < r)]-h)**2)/Z
    return px


def gaussian_pdf(x, sigma):
    return np.exp(-x**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

#def model_pdf(x, r, h, sigma, dx = 0.01):
#    f = circle_pdf( 
#    f = 
#    for m in 




