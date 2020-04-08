# Make folders in directory above available
import sys
sys.path.append('../')

from microscope_models import Microscope
from bacteria_model import SpherocylindricalBacteria
import matplotlib.pyplot as plt
import numpy as np

def main():
    # measurments in micrometers

    n_b = 5000  # number of samples

    # values from subdiffaction-llimit study of kaede diffusion (Somenath et
    # al)
    r_b = 0.3309786038590506
  # radius of cylinder caps in micrometers
    l_b = 2.9239029503218905
  # total length of cylindrical body excluding the caps
    theta = 12.03 #theta
    R_b = 10 #Radius of curvature
    density = 1000
    centroid = (16.2, 18)
    shape = (34 ,26)
    ex_wv = 0.8  # emmitted wavelength by microscope for excitation
    em_wv = 0.59  # emitted wavelength due to fluorescence
    pixel_size = 4.4  # pixel size
    NA = 0.95  # Numerical aperture
    magnification = 40  # magnification


    # Create bacteria model
    bacteria = SpherocylindricalBacteria(r_b, l_b, R_b, theta, ex_wv, em_wv, density)
    # Show 3D dots from Rejection sampling
    bacteria.plot_3D()
    # Show 2D dots by ignoring z-coordinate
    bacteria.plot_2D()
    # Create microscope
    microscope = Microscope(magnification, NA, ex_wv, em_wv, pixel_size)
    # Create image
    cell_conv = microscope.image_bacteria_conv(bacteria, centroid, shape)
    cell_sampling = microscope.image_bacteria_sample(bacteria, centroid, shape)

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2)

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    # use unified colormap max value of 700?
    axarr[0].imshow(cell_conv)
    axarr[1].imshow(cell_sampling)
    plt.show()

if __name__ == "__main__":
    main()
