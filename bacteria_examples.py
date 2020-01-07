from microscope_models import Fluorescent_microscope_spline
from bacteria_model import Fluorescent_bacteria_spline_fn
import numpy as np

def main():
    # measurments in micrometers

    n_b = 5000  # number of samples

    # values from subdiffaction-llimit study of kaede diffusion (Somenath et
    # al)
    r_b = 0.5  # radius of cylinder caps in micrometers
    l_b = 4  # total length of cylindrical body excluding the caps

    # Ball park values
    ex_wv = 0.8  # emmitted wavelength by microscope for excitation
    em_wv = 0.59  # emitted wavelength due to fluorescence
    pixel_size = 4.4  # pixel size
    NA = 0.95  # Numerical aperture
    magnification = 40  # magnification

    def spline_fn_curvature(x, R=10, l=l_b):
        return np.sqrt(R**2 - (x-l/2)**2) - np.sqrt(R**2-l**2/4)

    # Create bacteria model
    bacteria =  Fluorescent_bacteria_spline_fn(r_b, l_b, 0.01, spline_fn_curvature, 10, ex_wv, em_wv, n_b)
    # Show 3D dots from Rejection sampling
    bacteria.plot_3D()
    # Show 2D dots by ignoring z-coordinate
    bacteria.plot_2D()
    # Create microscope
    microscope = Fluorescent_microscope_spline(
            magnification, NA, ex_wv, em_wv, pixel_size)
    # Create image
    image = microscope.image_bacteria_conv(bacteria)

    # Display image with convolution
    microscope.display_image(image)

    # Display image with contour
    microscope.display_image_with_boundary(image, bacteria)

if __name__ == "__main__":
    main()
