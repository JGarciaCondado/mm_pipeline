import numpy as np

def compare_images(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must be the same shape")
    diff = image1-image2
    square_diff = np.square(diff)
    sum_image1 = np.sum(image1)
    sum_square_diff = np.sum(square_diff)
    metric = sum_square_diff/sum_image1
    return metric
