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
from tqdm import tqdm


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
            warnings.warn("The pixels are smaller than the nyquist frequency"
                          " of diffraction of light")

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

        # Calculate sigma of blur to fit 2D Guassian to Airy Disk 
        sigma_blur = self.rayleigh_criterion / 3

        # set padding level
        zero_padding = 10

        # populate image with correct number of pixels
        # x-direction is total width of bacteria*magnification / length of
        # pixel width
        x_pixels = round(2 * bacteria.r * self.m /
                       self.pixel_size) + zero_padding*2
        # y-direction is total height of bacteria*magnification / length of
        # pixel height
        y_pixels = round(self.m * (2 * bacteria.r + bacteria.l) /
                       self.pixel_size) + zero_padding*2

        self.image = np.zeros((y_pixels, x_pixels))

        for x, y, z in tqdm(np.array(bacteria.b_samples)):
            number_photons = np.random.poisson(255)
            for iter in range(number_photons):
                photon_x, photon_y = np.random.multivariate_normal([x,y], [[sigma_blur**2, 0], [0, sigma_blur**2]])
                location_x = round(self.m*(photon_x+bacteria.r+bacteria.l/2)/self.pixel_size) \
                          + zero_padding
                location_y = round(self.m*(photon_y+bacteria.r)/self.pixel_size) \
                          + zero_padding
                self.image[int(location_x), int(location_y)] += 1

        self.image = np.round(self.image*255/np.amax(self.image)) + np.random.poisson(10, (y_pixels, x_pixels))

        return self.image

    def image_bacteria_conv(self, bacteria):
        # Check that bacteria emitted wavelength is compatible with microscope
        if (bacteria.em_wavelength != self.em_wavelength or
                bacteria.ex_wavelength != self.ex_wavelength):
            raise ValueError(
                "Bacteria and microscope must have compatible wavelengths")

        # Calculate sigma of blur to fit 2D Guassian to Airy Disk 
        sigma_blur = self.m*self.rayleigh_criterion / (3*self.pixel_size)

        # set padding level
        zero_padding = 5

        # populate image with correct number of pixels
        # x-direction is total width of bacteria*magnification / length of
        # pixel width
        x_pixels = round(2 * bacteria.r * self.m /
                       self.pixel_size) + zero_padding*2
        # y-direction is total height of bacteria*magnification / length of
        # pixel height
        y_pixels = round(self.m * (2 * bacteria.r + bacteria.l) /
                       self.pixel_size) + zero_padding*2

        self.image = np.zeros((y_pixels, x_pixels))

        for x, y, z in tqdm(np.array(bacteria.b_samples)):
            location_x = round(self.m*(x+bacteria.r+bacteria.l/2)/self.pixel_size) \
                          + zero_padding
            location_y = round(self.m*(y+bacteria.r)/self.pixel_size) \
                          + zero_padding
            self.image[int(location_x), int(location_y)] += np.random.poisson(255)

        return gaussian_filter(self.image, sigma=sigma_blur)


    def display_image(self, image):
        """Displays image.

        This method is used to display fluorescent synthetic images.

        Parameters
        ----------
        image: 2D np.array of ints
        """

        # Create red color map
        colors = [(0, 0, 0), (1, 0, 0)]
        cm = LinearSegmentedColormap.from_list('test', colors, N=np.amax(image))

        # Display image
        plt.imshow(image, cmap=cm, origin="lower")
        plt.show()
