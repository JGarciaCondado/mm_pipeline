#!/usr/bin/env python3
"""Parse command line options and arguments for the Mother Machine pipeline.

 This script parses options and arguments specified on the command line, and
 runs the pipleine.

 Usage
 -----
 Show help: mm_pipeline.py -h
 Run pipeline: mm_pipeline.py -r
 """

from microscope_models import Fluorescent_microscope
from bacteria_model import Fluorescent_bacteria

import getopt
import sys


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
    r_b = 0.4  # radius of cylinder caps in micrometers
    l_b = 3  # total length of cylindrical body excluding the caps

    # Ball park values
    ex_wv = 0.8  # emmitted wavelength by microscope for excitation
    em_wv = 0.59  # emitted wavelength due to fluorescence
    pixel_size = 1  # pixel size
    NA = 0.95  # Numerical aperture
    magnification = 40  # magnification

    for option, path in options:
        if option == "-h":  # print the usage message
            print(usage_message)
            sys.exit()
    if not options:
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


if __name__ == "__main__":
    main(sys.argv[1:])
