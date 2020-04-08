#!/usr/bin/env python3
"""Parse command line options and arguments for the Mother Machine pipeline.

 This script parses options and arguments specified on the command line, and
 runs the pipleine.

 Usage
 -----
 Show help: mm_pipeline.py -h
 Run pipeline: mm_pipeline.py -r
 """

from microscope_models import Fluorescent_microscope_spline, Microscope
from bacteria_model import Fluorescent_bacteria_spline_fn, SpherocylindricalBacteria

import matplotlib.pyplot as plt
import getopt
import sys
import numpy as np

def main(arg_list):
    """Parse the command line options and arguments specified in arg_list.

    """
    usage_message = ("Usage:\n"
                     "Show help: mm_pipeline.py -h\n"
                     "Run: mm_pipleine.py\n")
    try:
        options, arguments = getopt.getopt(arg_list, "hc:")
    except getopt.GetoptError:
        print("Error: invalid command line arguments\n")
        print(usage_message)
        sys.exit()

    # measurments in micrometers

    n_b = 5000  # number of samples

    # values from subdiffaction-llimit study of kaede diffusion (Somenath et
    # al)
    r_b = 0.7  # radius of cylinder caps in micrometers
    l_b = 4.5  # total length of cylindrical body excluding the caps

    # Ball park values
    ex_wv = 0.8  # emmitted wavelength by microscope for excitation
    em_wv = 0.59  # emitted wavelength due to fluorescence
    pixel_size = 4.4  # pixel size
    NA = 0.95  # Numerical aperture
    magnification = 40  # magnification

    def spline_fn_curvature(x, R=100, l=l_b):
        return np.sqrt(R**2 - (x-l/2)**2) - np.sqrt(R**2-l**2/4)

    for option, path in options:
        if option == "-h":  # print the usage message
            print(usage_message)
            sys.exit()
    if not options:
        # Create bacteria model
#        bacteria = Fluorescent_bacteria_spline_fn(r_b, l_b, 0.01, spline_fn_curvature, -15, ex_wv, em_wv, n_b)
        bacteria = SpherocylindricalBacteria(r_b, l_b, 10, -15, ex_wv, em_wv, 50)
        bacteria2 = SpherocylindricalBacteria(0.3, l_b, 100, 5, ex_wv, em_wv, 50)
        # Create microscope model
        microscope = Microscope(
            magnification, NA, ex_wv, em_wv, pixel_size)
        # Show 3D dots from Rejection sampling
#        bacteria.plot_3D()
        # Show 2D dots by ignoring z-coordinate
##        bacteria.plot_2D()
        # Create image
        image = microscope.image_bacteria_conv(bacteria, (15, 30), (40, 40))
        image = microscope.image_trench([bacteria, bacteria2, bacteria, bacteria2], [(13, 30), (13, 80), (13, 130), (13, 180)])
#        print(image)
#       image_gt = microscope.image_ground_truth_pixels(bacteria)
#        plt.imshow(image_gt)
#        plt.show()
#        microscope.display_image(image_gt)
        # Display image
        microscope.display_image(image)


if __name__ == "__main__":
    main(sys.argv[1:])
