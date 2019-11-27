import warnings
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class Fluorescent_microscope:


    def __init__(self, m, NA, ex_wavelength, em_wavelength, pixel_size):
        #all measurments must be in micrometers
        self.m = m #magnification
        self.NA = NA #numberical aperture
        self.em_wavelength = em_wavelength #emitted wavelength capture by microscope
        self.ex_wavelength = ex_wavelength #excitation wavelength created by microscope
        self.pixel_size = pixel_size #assuming pizels are square
        self.image = []
        self.rayleigh_criterion = 0.61*(self.em_wavelength/self.NA)

        if(self.rayleigh_criterion*self.m/ 2 < self.pixel_size):
            warnings.warn("The pixels are smaller than the nyquist frequency of diffraction of light")

    def image_bacteria(self, bacteria):
        if (bacteria.em_wavelength != self.em_wavelength or
            bacteria.ex_wavelength != self.ex_wavelength):
            raise ValueError("Bacteria and microscope must have compatible wavelengths")

        zero_padding = 30

        #populate image with correct number of pixels 
        #x-direction is total length of bacteria*magnification / length of pixel width
        x_pixels = int(2*bacteria.r*self.m/self.pixel_size) + zero_padding
        #y-direction is total height of bacteria*magnification / length of pixel height
        y_pixels = int(self.m*(2*bacteria.r+bacteria.l)/self.pixel_size) + zero_padding

        self.image = 255*np.ones((x_pixels, y_pixels))

        for sample in bacteria.b_samples:
            location_x = int(self.m*(sample[0]+bacteria.r+bacteria.l/2)/self.pixel_size) + int(zero_padding/2)
            location_y = int(self.m*(sample[1]+bacteria.r)/self.pixel_size) + int(zero_padding/2)
            self.image[location_y, location_x] = 0

        sigma_blur = self.rayleigh_criterion*self.m/1.476

        return gaussian_filter(self.image, sigma=sigma_blur)

    def display_image(self, image):
        colors = [(1, 0, 0), (0, 0, 0)] # Experiment with this
        cm = LinearSegmentedColormap.from_list('test', colors, N=255)
        plt.imshow(image, cmap=cm, origin="lower")
        plt.show()
