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
from matplotlib.path import Path
import matplotlib.patches as patches


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

        for x, y, z in np.array(bacteria.b_samples):
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

        for x, y, z in np.array(bacteria.b_samples):
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

class Fluorescent_microscope_spline:
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
        self.padding = 2
        self.height = 60

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
        sigma_blur = self.rayleigh_criterion / 2.9

        # set padding level
        zero_padding = 15

        # populate image with correct number of pixels
        # x-direction is total width of bacteria*magnification / length of
        # pixel width
        x_pixels = round((np.amax(bacteria.b_samples_y)-np.amin(bacteria.b_samples_y)) * self.m /
                       self.pixel_size) + zero_padding*2
        # y-direction is total height of bacteria*magnification / length of
        # pixel height
        y_pixels = round((np.amax(bacteria.b_samples_x)-np.amin(bacteria.b_samples_x)) * self.m /
                       self.pixel_size) + zero_padding*2

        self.image = np.zeros((int(y_pixels), int(x_pixels)))

        for x, y, z in np.array(bacteria.b_samples):
            number_photons = np.random.poisson(255)
            for iter in range(number_photons):
                photon_x, photon_y = np.random.multivariate_normal([x,y], [[sigma_blur**2, 0], [0, sigma_blur**2]])
                location_x = round(self.m*(photon_x-np.amin(bacteria.b_samples_x))/self.pixel_size) \
                          + zero_padding
                location_y = round(self.m*(photon_y-np.amin(bacteria.b_samples_y))/self.pixel_size) \
                          + zero_padding
                self.image[int(location_x), int(location_y)] += 1

        self.image = np.round(self.image*255/np.amax(self.image))
        self.image = self.image + np.random.poisson(10, (int(y_pixels), int(x_pixels)))

        return self.image

    def image_bacteria_conv(self, bacteria):
        # Check that bacteria emitted wavelength is compatible with microscope
        if (bacteria.em_wavelength != self.em_wavelength or
                bacteria.ex_wavelength != self.ex_wavelength):
            raise ValueError(
                "Bacteria and microscope must have compatible wavelengths")

        # Calculate sigma of blur to fit 2D Guassian to Airy Disk 
        sigma_blur = self.m*self.rayleigh_criterion / (2.9*self.pixel_size)

        # populate image with correct number of pixels
        # x-direction is total width of bacteria*magnification / length of
        # pixel width
        x_pixels = 26
        # y-direction is total height of bacteria*magnification / length of
        # pixel height
        y_pixels = round((bacteria.x_max-bacteria.x_min)* self.m /
                       self.pixel_size) + self.padding*2

        self.image = np.zeros((int(y_pixels), int(x_pixels)))

        for x, y, z in np.array(bacteria.b_samples):
        # chane b_samples x min por radius -> since its always gonna be smallest
            location_x = round(self.m*(x-bacteria.x_min)/self.pixel_size) \
                          + self.padding
            location_y = round(self.m*(y-bacteria.y_min)/self.pixel_size) \
                          + self.padding
            self.image[int(location_x), int(location_y)] += np.random.poisson(255)

        self.image = gaussian_filter(self.image, sigma=sigma_blur)
        self.image = np.round(self.image*np.random.poisson(400)/np.amax(self.image))
        self.image = self.image + np.random.poisson(200, (int(y_pixels), int(x_pixels)))

        self.pad = (self.height - self.image.shape[0]) / 2
        if (self.pad).is_integer():
            self.image = np.pad(self.image, ((int(self.pad), int(
                self.pad)), (0, 0)), mode='constant', constant_values=0)
        else:
            self.image = np.pad(self.image, ((
                int(self.pad - 0.5), int(self.pad + 0.5)), (0, 0)),
                mode='constant', constant_values=0)

        return self.image

    def image_ground_truth_pixels(self, bacteria):
        # populate image with correct number of pixels
        # x-direction is total width of bacteria*magnification / length of
        # pixel width
        x_pixels = 26
        # y-direction is total height of bacteria*magnification / length of
        # pixel height
        y_pixels = round((bacteria.x_max-bacteria.x_min)* self.m /
                       self.pixel_size) + self.padding*2

        self.image = np.zeros((int(y_pixels), int(x_pixels)))

        for x, y, z in np.array(bacteria.b_samples):
        # chane b_samples x min por radius -> since its always gonna be smallest
            location_x = round(self.m*(x-bacteria.x_min)/self.pixel_size) \
                          + self.padding
            location_y = round(self.m*(y-bacteria.y_min)/self.pixel_size) \
                          + self.padding
            self.image[int(location_x), int(location_y)] = 1

        self.pad = (self.height - self.image.shape[0]) / 2
        if (self.pad).is_integer():
            self.image = np.pad(self.image, ((int(self.pad), int(
                self.pad)), (0, 0)), mode='constant', constant_values=0)
        else:
            self.image = np.pad(self.image, ((
                int(self.pad - 0.5), int(self.pad + 0.5)), (0, 0)),
                mode='constant', constant_values=0)

        return self.image

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
        plt.title("Bacteria Image")
        plt.imshow(image, cmap=cm, origin="lower")
        plt.show()

    def _transform_vertices(self, verts, bacteria):
        verts = verts - bacteria.min[:-1] # move to get non-zero values
        verts = verts*self.m #magnification
        verts = verts / self.pixel_size #scaling by size of pixels
        verts = verts + self.padding # add padding
        verts[:, 0] = verts[:, 0] + int(self.pad)
        verts[:,[0, 1]] = verts[:,[1, 0]] #make horizontal
        return verts

    def display_image_with_boundary(self, image, bacteria):
        fig, ax = plt.subplots()
        # Create red color map
        colors = [(0, 0, 0), (1, 0, 0)]
        cm = LinearSegmentedColormap.from_list('test', colors, N=np.amax(image))

        # Display spline
        verts_spline = bacteria.spline[:, :-1] # not the z-axis
        verts_spline = self._transform_vertices(verts_spline, bacteria)
        ax.plot(verts_spline[:, 0], verts_spline[:, 1], 'y', label='spline')

        # Display boundary
        verts_boundary = bacteria.boundary[:, :-1]
        verts_boundary = self._transform_vertices(verts_boundary, bacteria)
        ax.plot(verts_boundary[:, 0], verts_boundary[:, 1], 'g', label='boundary')

        # Display image
        plt.title("Bacteria Image with cell Boundary")
        plt.imshow(image, cmap=cm, origin="lower")
        plt.legend(fontsize='x-small')
        plt.show()


