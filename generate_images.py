from microscope_models import Fluorescent_microscope_spline
from bacteria_model import Fluorescent_bacteria_spline_fn
import numpy as np

def main():
    # measurments in micrometers

    n_b = 5000  # number of samples
    n = 2

    ex_wv = 0.8  # emmitted wavelength by microscope for excitation
    em_wv = 0.59  # emitted wavelength due to fluorescence
    pixel_size = 4.4  # pixel size
    NA = 0.95  # Numerical aperture
    magnification = 40  # magnification

    for i in range(n):
        r = np.random.uniform(0.3, 0.7)  # radius of cylinder caps in micrometers
        l = np.random.uniform(1.5, 4.5)  # total length of cylindrical body excluding the caps
        R = np.random.uniform(4, 40)
        theta = np.random.uniform(-15, 15)

        def spline_fn_curvature(x):
            return np.sqrt(R**2 - (x-l/2)**2) - np.sqrt(R**2-l**2/4)

        # Create bacteria model
        bacteria = Fluorescent_bacteria_spline_fn(r, l, 0.01, spline_fn_curvature, theta, ex_wv, em_wv, n_b)
        # Create microscope model
        microscope = Fluorescent_microscope_spline(
            magnification, NA, ex_wv, em_wv, pixel_size)
        # Create image
        image = microscope.image_bacteria_conv(bacteria)
        # Image ground truth
        image_gt = microscope.image_ground_truth_pixels(bacteria)
        # Save both
        np.save('../dataset/im_r%s_l%s_R%s_O%s.npy' % (r, l, R, theta), image)
        np.save('../dataset/im_gt_r%s_l%s_R%s_O%s.npy' % (r, l, R, theta), image_gt)
        # Flip and save
        image = np.flip(image, axis = 1)
        image_gt = np.flip(image_gt, axis=1)
        # Save both
        np.save('../dataset/im_r%s_l%s_R%s_O%s_fy.npy' % (r, l, R, theta), image)
        np.save('../dataset/im_gt_r%s_l%s_R%s_O%s_fy.npy' % (r, l, R, theta), image_gt)
        # Flip again and save
        image = np.flip(image, axis = 0)
        image_gt = np.flip(image_gt, axis=0)
        # Save both
        np.save('../dataset/im_r%s_l%s_R%s_O%s_fyx.npy' % (r, l, R, theta), image)
        np.save('../dataset/im_gt_r%s_l%s_R%s_O%s_fyx.npy' % (r, l, R, theta), image_gt)
        # Flip again to obtain same as a single flip along axis 0 and save
        image = np.flip(image, axis = 1)
        image_gt = np.flip(image_gt, axis=1)
        # Save both
        np.save('../dataset/im_r%s_l%s_R%s_O%s_fx.npy' % (r, l, R, theta), image)
        np.save('../dataset/im_gt_r%s_l%s_R%s_O%s_fx.npy' % (r, l, R, theta), image_gt)

if __name__ == "__main__":
    main()
