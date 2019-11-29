"""Models and builds an epi-iluminated fluroescent microscope .

 Used in the Mother Machien image-processing pipeline to generate
 photorealistic images from bacteria models.

 Classes
 -------
 Fluorescent_microscope - initialises a microscope model.
"""

import warnings
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class Fluorescent_microscope:
    """ Build an epi-illumination microscope model.

    The model deals with both generating images from bacteria models
    and displaying such images.

    Parameters
    ----------
    m: magnification of microscope (micrometers)
    NA: microscope numerical aperture
    em_wavelength: emitted wavelength captured by microscope (micrometers)
    ex_wavelength: wavelength of light emitted by microscope (micrometers)
    pixel_size: size of pixels in CCD of microscope (micrometers)

    Public methods
    --------------
    image_bacteria(self, bacteria): Produces an image of the bacteria given.
    display_image(self, image): Displays the image given.
    """

    def __init__(self, m, NA, ex_wavelength, em_wavelength, pixel_size):
        """Initialise constants."""
        self.m = m  # magnification
        self.NA = NA  # numberical aperture
        self.em_wavelength = em_wavelength  # emitted wavelength captured
        self.ex_wavelength = ex_wavelength  # wavelength emitted by microscope
        self.pixel_size = pixel_size  # assuming pizels are square
        self.image = []

        # Calculate rayleigh_criterion for microscope
        self.rayleigh_criterion = 0.61 * (self.em_wavelength / self.NA)

        # Check that image resolution is diffraciton limited
        if(self.rayleigh_criterion * self.m / 2 < self.pixel_size):
            warnings.warn("The pixels are smaller than the nyquist frequency of
                          diffraction of light")

    def image_bacteria(self, bacteria):
        """Returns an image of the bacteria.

        This method is used to generate synthetic images of single bacteria.

        Parameters
        ----------
        bacteria: instance of the bacteria_model.Fluorescent_bacteria() class.

        Returns
        -------
        An 2D np.array of int, corresponding to the pixel value of the image.

        """

        # Check that bacteria emitted wavelength is compatible with microscope
        if (bacteria.em_wavelength != self.em_wavelength or
                bacteria.ex_wavelength != self.ex_wavelength):
            raise ValueError(
                "Bacteria and microscope must have compatible wavelengths")

        # set padding level
        zero_padding = 30

        # populate image with correct number of pixels
        # x-direction is total length of bacteria*magnification / length of
        # pixel width
        x_pixels = int(2 * bacteria.r * self.m /
                       self.pixel_size) + zero_padding
        # y-direction is total height of bacteria*magnification / length of
        # pixel height
        y_pixels = int(self.m * (2 * bacteria.r + bacteria.l) /
                       self.pixel_size) + zero_padding

        self.image = 255 * np.ones((x_pixels, y_pixels))

        # Populate image with pixel values of 255 where there are samples
        for sample in bacteria.b_samples:
            location_x = int(
                self.m * (sample[0] + bacteria.r + bacteria.l / 2)
                / self.pixel_size) + int(zero_padding / 2)
            location_y = int(
                self.m * (sample[1] + bacteria.r) / self.pixel_size)
            + int(zero_padding / 2)
            self.image[location_y, location_x] = 0

        # Calculate sigma of blur to match energy of Airy Disk with 2D gaussian
        sigma_blur = self.rayleigh_criterion * self.m / 1.476

        return gaussian_filter(self.image, sigma=sigma_blur)

    def display_image(self, image):
        """Displays image.

        This method is used to display fluorescent synthetic images.

        Parameters
        ----------
        image: 2D np.array of ints
        """

        # Create red color map
        colors = [(1, 0, 0), (0, 0, 0)]
        cm = LinearSegmentedColormap.from_list('test', colors, N=255)

        # Display image
        plt.imshow(image, cmap=cm, origin="lower")
        plt.show()
