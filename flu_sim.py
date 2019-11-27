from microscope_models import Fluorescent_microscope
from bacteria_model import Fluorescent_bacteria

# measurments in micrometers

n_b = 5000  # number of samples

# values from subdiffaction-llimit study of kaede diffusion (Somenath et al)
r_b = 0.4  # radius of cylinder caps in micrometers
l_b = 3  # total length of cylindrical body excluding the caps
ex_wv = 0.8  # emmitted wavelength by microscope for fluroescence excitation
em_wv = 0.59  # emitted wavelength due to fluorescence
pixel_size = 1  # pixel size
NA = 0.95  # Numerical aperture
magnification = 40  # magnification


if __name__ == "__main__":
    # Create bacteria model
    bacteria = Fluorescent_bacteria(r_b, l_b, ex_wv, em_wv, n_b)
    # Create microscope model
    microscope = Fluorescent_microscope(
        magnification, NA, ex_wv, em_wv, pixel_size)
    # Show 3D dots from Rejection sampling
    bacteria.plot_3D()
    # Show 2D dots by ignoring z-coordinate
    bacteria.plot_2D()
    # Create image
    image = microscope.image_bacteria(bacteria)
    # Display image
    microscope.display_image(image)
