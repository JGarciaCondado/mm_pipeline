import numpy as np
from scipy.ndimage.filters import gaussian_filter

def compare_images(original_im, generated_im, blur = True, sigma = 1.0):
    if original_im.shape != generated_im.shape:
        raise ValueError("Images must be the same shape")
    if blur:
        orignal_im = gaussian_filter(original_im, sigma = sigma)
        generated_im = gaussian_filter(generated_im, sigma = sigma)
    #normalize images
    original_im = (original_im-np.amin(original_im))/(np.amax(original_im)-np.min(original_im))
    generated_im = (generated_im-np.amin(generated_im))/(np.amax(generated_im)-np.min(generated_im))
    diff = original_im-generated_im
    square_diff = np.square(diff)
    sum_org = np.sum(original_im)
    sum_square_diff = np.sum(square_diff)
    metric = sum_square_diff/sum_org
    return metric
