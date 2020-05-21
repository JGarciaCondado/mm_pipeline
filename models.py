"""Model and build a rod-shaped uniformly distributed fluorescing bacteria.

 Used in the Mother Machien image-processing pipeline to generate synthetic
 data and model the behaviour of fluorescing bacteria.

 Classes
 -------
 Fluorescent_bacteria - initialises a rod-shaped bacteria.
 """

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import cauchy
import warnings
from matplotlib.colors import LinearSegmentedColormap

class SpherocylindricalBacteria:
    """ Build an epi-illumination microscope model.

    The model deals with both generating images from bacteria models
    and displaying such images.

    Parameters
    ----------
    r: radius of caps of rod-shaped bacteria (micrometers)
    l: length of cylinder of rod_shaped bacteria (micrometers)
    theta: (degrees)
    em_wavelength: emitted wavelength captured by microscope (micrometers)
    ex_wavelength: wavelength of light emitted by microscope (micrometers)
    n: number of samples to sample
    n_total: True if n denotes the total numer of accepted samples or false
             if it denotes the total number of sampling iterations

    Public methods
    --------------
    plot_3D(self): Produces a 3D plot of samples within bacteria.
    plot_2D(self): Produces a 2D plot of samples ignoring z-coordinates.
    """
    def __init__(self, l, r, R, theta, density, ex_wavelength, em_wavelength, dx=0.01):
        """Constructor that defines the parametes of the bacteria and creates samples
        within the bacteria.
        """

        # Check that length is greater than radius
        if(l < r):
            raise ValueError("The length must be bigger than the radius")
        elif(l > 2*abs(R)):
            raise ValueError("The length must be smaller than twice the curvature")
        elif(ex_wavelength < em_wavelength):
            raise ValueError("The excitation wavelength must be shorter than the emission wavelength")

        self.r = r
        self.l = l
        self.R = R
        self.theta = theta*np.pi/180 #Convert to 
        self.dx = dx
        self.density = density
        self.ex_wavelength = ex_wavelength
        self.em_wavelength = em_wavelength

        #Calculate volume and number of samples
        self.volume_cap = 2*np.pi*self.r**3/3
        self.volume_cylinder = (np.pi*self.r**2)*(self.R*(2*np.arcsin(self.l/(2*self.R))))
        self.volume = 2*self.volume_cap+self.volume_cylinder
        self.n = int(self.volume*self.density)
        self.n_caps = int(2*self.volume_cap*self.density)
        self.n_cylinder = self.n-self.n_caps

        #Calculate rotaion matrix
        self.rotation_matrix = np.array(((np.cos(self.theta), -np.sin(self.theta), 0.0), (np.sin(self.theta), np.cos(self.theta), 0.0), (0.0, 0.0, 1.0)))

        #Define the spline of the curved cylinder
        self.spline = np.array([[x, self._fn(x), 0.0] for x in np.arange(-self.l/2, self.l/2+dx, dx)]).dot(np.transpose(self.rotation_matrix))

        #Define boundary
        self.boundary = list(LineString(self.spline).buffer(self.r).exterior.coords)

        #Define the maximums of each dimension
        self.x_max, self.x_min = self.l/2+self.r, -self.l/2-self.r
        self.y_max, self.y_min = self.r, -self.l
        self.z_max, self.z_min = self.r, -self.r

        #Define the angles that the torus extends 
        self.phi_max, self.phi_min = np.arctan((self.l/2)/(self.R+self._fn(self.l/2))), np.arctan((-self.l/2)/(self.R+self._fn(-self.l/2)))


        # Arrays to store the bacteria samples
        self.samples = self._sample()
        self.samples_x, self.samples_y, self.samples_z = list(
            zip(*self.samples))

    def _fn(self, x):
        """ Function describing a circle with center at (0, -R) and
        a radius of R """
        return np.sign(self.R)*np.sqrt(self.R**2 - x**2) - self.R

    def _sample(self):
        """ Samples point within the bacteria cell boundary.

        This method is used when initializign a bacteria to generate samples
        that represent infinitesimally small light sources of fluorescence.
        """

        # Sample until certain number of iteration or a certain number of
        # samples obtaines
        i = 0
        samples = []
        while(i < self.n_caps):
            # sample using rejection sampling for cylinder
            x_sample = np.random.uniform(-self.r, self.r)
            y_sample = np.random.uniform(-self.r, self.r)
            z_sample = np.random.uniform(-self.r, self.r)
            if x_sample**2+y_sample**2+z_sample**2 < self.r**2:
                sample = np.array([x_sample,y_sample,z_sample])
                if x_sample < 0:
                    sample = np.array(((np.cos(self.phi_max), -np.sin(self.phi_max), 0.0), (np.sin(self.phi_max), np.cos(self.phi_max), 0.0), (0.0, 0.0, 1.0))).dot(sample)
                    sample = sample+np.array([-self.l/2, self._fn(-self.l/2), 0.0])
                else:
                    sample = np.array(((np.cos(-self.phi_max), -np.sin(-self.phi_max), 0.0), (np.sin(-self.phi_max), np.cos(-self.phi_max), 0.0), (0.0, 0.0, 1.0))).dot(sample)
                    sample = sample+np.array([self.l/2, self._fn(self.l/2), 0.0])
                rotated_sample = self.rotation_matrix.dot(sample)
                samples.append(rotated_sample)
                i +=1

        i = 0
        while(i < self.n_cylinder):
            #sample using outlined in
            phi = np.random.uniform(self.phi_min, self.phi_max)
            theta = np.random.uniform(0, 2*np.pi)
            p = self.r*np.sqrt(np.random.uniform())
            W = np.random.uniform()
            if W < (self.R+p*np.cos(theta))/(self.R+p):
                x = (self.R+p*np.cos(theta))*np.sin(phi)
                y = (self.R+p*np.cos(theta))*np.cos(phi)-self.R
                z = p*np.sin(theta)
                sample = np.array([x,y,z])
                rotated_sample = self.rotation_matrix.dot(sample)
                samples.append(rotated_sample)
                i += 1

        return samples


    def plot_3D(self):
        """ Plot samples in a 3D plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            self.samples_x,
            self.samples_y,
            self.samples_z,
            s=1,
            c='r',
            marker='o')

        ax.set_xlim([self.x_min, self.x_max])
        ax.set_ylim([self.y_min, self.y_max])
        ax.set_zlim([self.z_min, self.z_max])
        plt.title("Bacteria samples in 3D")
        plt.axis('scaled')
        plt.show()

    def plot_2D(self):
        """ Plot samples in a 2D plot ignoring z-coordinate."""
        plt.title("Bacteria samples in 2D")
        plt.scatter(self.samples_x, self.samples_y, s=1, label="samples")
        plt.plot(self.spline[:, 0], self.spline[:, 1], c='orange', label='spline')
        boundary_x, boundary_y = list(zip(*self.boundary))
        plt.plot(boundary_x, boundary_y, c='green', label='boundary')
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.legend()
        plt.axis('scaled')
        plt.show()

class Microscope:
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

        # Calculate rayleigh_criterion for microscope
        self.rayleigh_criterion = 0.61 * (self.em_wavelength / self.NA)

        # Check that image resolution is diffraciton limited
        if(self.rayleigh_criterion * self.m / 2 < self.pixel_size):
            warnings.warn("The pixels are smaller than the nyquist frequency"
                          " of diffraction of light")

    def image_bacteria(self, bacteria, centroid, shape, sigma = 0.0, photons=30, noise=200, gain = 1.0, padding=False):
        # Check that bacteria emitted wavelength is compatible with microscope
        if (bacteria.em_wavelength != self.em_wavelength or
                bacteria.ex_wavelength != self.ex_wavelength):
            raise ValueError(
                "Bacteria and microscope must have compatible wavelengths")

        # Preset value of sigma to fit 2D Guassian to theoretical Airy Disk from rayleigh criterion 
        if sigma == 0.0:
            sigma = self.m*self.rayleigh_criterion / (2.9*self.pixel_size)

        # Create array to store image
        self.image = np.zeros(shape)

        # Emission of photons
        total_photons = 0
        # TODO check all rounding
        for x, y, z in np.array(bacteria.samples):
        # chanige b_samples x min por radius -> since its always gonna be smallest
            location_x = round(self.m*x/self.pixel_size \
                              + centroid[1] )
            location_y = round(self.m*y/self.pixel_size \
                              + centroid[0] )
            #TODO change this so it check x and y not out of boundary as well not only negative
            if location_x > 0 and location_y > 0 and location_x < self.image.shape[0] and location_y < self.image.shape[1]:
                photons_emitted = np.random.poisson(photons)
                self.image[int(location_x), int(location_y)] += gain*photons_emitted
                total_photons += gain*photons_emitted

        # Apply convolution with gaussian filter
        self.image = gaussian_filter(self.image, sigma=sigma)
        # Normalize so the number of phtons is the number of total_photons emitters
        self.image = np.round(self.image*np.sum(self.image)/total_photons)
        # Add noise
        self.image = self.image + gain*np.random.poisson(noise, shape)

        #TODO fix padding functions -> give padding shape 
        #TODO test that padding shape is not smaller than image
        if padding:
            self.pad = (0.0 - self.image.shape[0]) / 2
            if (self.pad).is_integer():
                self.image = np.pad(self.image, ((int(self.pad), int(
                    self.pad)), (0, 0)), mode='constant', constant_values=0)
            else:
                self.image = np.pad(self.image, ((
                    int(self.pad - 0.5), int(self.pad + 0.5)), (0, 0)),
                    mode='constant', constant_values=0)

        return self.image

    def image_bacteria_ground_truth(self, bacteria, centroid, shape, padding=False):
        # Check that bacteria emitted wavelength is compatible with microscope
        if (bacteria.em_wavelength != self.em_wavelength or
                bacteria.ex_wavelength != self.ex_wavelength):
            raise ValueError(
                "Bacteria and microscope must have compatible wavelengths")

        # Create array to store image
        self.image = np.zeros(shape)

        # TODO check all rounding
        for x, y, z in np.array(bacteria.samples):
        # chanige b_samples x min por radius -> since its always gonna be smallest
            location_x = round(self.m*x/self.pixel_size \
                              + centroid[1] )
            location_y = round(self.m*y/self.pixel_size \
                              + centroid[0] )
            #TODO change this so it check x and y not out of boundary as well not only negative
            if location_x > 0 and location_y > 0 and location_x < self.image.shape[0] and location_y < self.image.shape[1]:
                self.image[int(location_x), int(location_y)] = 1.0


        #TODO fix padding functions -> give padding shape 
        #TODO test that padding shape is not smaller than image
        if padding:
            self.pad = (0.0 - self.image.shape[0]) / 2
            if (self.pad).is_integer():
                self.image = np.pad(self.image, ((int(self.pad), int(
                    self.pad)), (0, 0)), mode='constant', constant_values=0)
            else:
                self.image = np.pad(self.image, ((
                    int(self.pad - 0.5), int(self.pad + 0.5)), (0, 0)),
                    mode='constant', constant_values=0)

        return self.image
    def image_bacteria_sampling(self, bacteria, centroid, shape, sigma = 0.0, photons=30, noise=200, gain = 1.0, padding=False):
        # Check that bacteria emitted wavelength is compatible with microscope
        if (bacteria.em_wavelength != self.em_wavelength or
                bacteria.ex_wavelength != self.ex_wavelength):
            raise ValueError(
                "Bacteria and microscope must have compatible wavelengths")

        # Preset value of sigma to fit 2D Guassian to theoretical Airy Disk from rayleigh criterion 
        if sigma == 0.0:
            sigma = self.m*self.rayleigh_criterion / (2.9*self.pixel_size)

        # Create array to store image
        self.image = np.zeros(shape)

        # TODO check all rounding
        for x, y, z in np.array(bacteria.samples):
        # chanige b_samples x min por radius -> since its always gonna be smallest
            location_x = self.m*x/self.pixel_size \
                              + centroid[1]
            location_y = self.m*y/self.pixel_size \
                              + centroid[0]
            #TODO change this so it check x and y not out of boundary as well not only negative
            photons_emitted = np.random.normal([location_x, location_y], sigma, (np.random.poisson(photons), 2))
            for xp, yp in photons_emitted:
                if xp > -0.5 and yp > -0.5 and xp < self.image.shape[0]-0.5 and yp < self.image.shape[1] - 0.5:
                    self.image[int(round(xp)), int(round(yp))] += gain

        # Add noise
        self.image = self.image + np.random.poisson(noise, shape)

        return self.image

    def image_bacteria_cauchy(self, bacteria, centroid, shape, gamma = 1.0, photons=30, noise=150, gain = 1.0, padding=False):
        # Check that bacteria emitted wavelength is compatible with microscope
        if (bacteria.em_wavelength != self.em_wavelength or
                bacteria.ex_wavelength != self.ex_wavelength):
            raise ValueError(
                "Bacteria and microscope must have compatible wavelengths")

        # Create array to store image
        self.image = np.zeros(shape)

        # TODO check all rounding
        for x, y, z in np.array(bacteria.samples):
        # chanige b_samples x min por radius -> since its always gonna be smallest
            location_x = self.m*x/self.pixel_size \
                              + centroid[1]
            location_y = self.m*y/self.pixel_size \
                              + centroid[0]
            #TODO change this so it check x and y not out of boundary as well not only negative
            photons_emitted = cauchy.rvs(loc = [location_x, location_y], scale=gamma, size=(np.random.poisson(photons), 2))
            for xp, yp in photons_emitted:
                if xp > -0.5 and yp > -0.5 and xp < self.image.shape[0]-0.5 and yp < self.image.shape[1] - 0.5:
                    self.image[int(round(xp)), int(round(yp))] += gain

        # Add noise
        self.image = self.image + gain*np.random.poisson(noise, shape)

        return self.image

    def image_trench(self, bacterias, centroids, shape, sigma = 0.0, photons=30, noise=200, gain = 1.0):

        # Preset value of sigma to fit 2D Guassian to theoretical Airy Disk from rayleigh criterion 
        if sigma == 0.0:
            sigma = self.m*self.rayleigh_criterion / (2.9*self.pixel_size)

        # Create array to store image
        self.image = np.zeros(shape)

        # Emission of photons
        total_photons = 0
        for bacteria, centroid in zip(bacterias, centroids):
            # Check that bacteria emitted wavelength is compatible with microscope
            if (bacteria.em_wavelength != self.em_wavelength or
                    bacteria.ex_wavelength != self.ex_wavelength):
                raise ValueError(
                    "Bacteria and microscope must have compatible wavelengths")

            for x, y, z in np.array(bacteria.samples):
            # chanige b_samples x min por radius -> since its always gonna be smallest
                location_x = round(self.m*x/self.pixel_size) \
                                  + centroid[1]
                location_y = round(self.m*y/self.pixel_size) \
                                  + centroid[0]
                #TODO change this so it check x and y not out of boundary as well not only negative
                if location_x > 0 and location_y > 0 and location_x < self.image.shape[0] and location_y < self.image.shape[1]:
                    photons_emitted = np.random.poisson(photons)
                    self.image[int(location_x), int(location_y)] += gain*photons_emitted
                    total_photons += gain*photons_emitted

        # Apply convolution with gaussian filter
        self.image = gaussian_filter(self.image, sigma=sigma)
        # Normalize so the number of phtons is the number of total_photons emitters
        self.image = np.round(self.image*np.sum(self.image)/total_photons)
        # Add noise
        self.image = self.image + np.random.poisson(noise, shape)

        return self.image

    def image_MM_sampling(self, bacteria_array, centroid_array, shape, distance_between_trenches, sigma = 0.0, photons=30, noise=200, gain = 1.0):

        # Preset value of sigma to fit 2D Guassian to theoretical Airy Disk from rayleigh criterion 
        if sigma == 0.0:
            sigma = self.m*self.rayleigh_criterion / (2.9*self.pixel_size)

        # Create array to store image
        self.image = np.zeros(shape)

        # Emission of photons
        total_photons = 0
        trench_pos = 0

        # Centroids are relative to the trench
        for bacterias, centroids in zip(bacteria_array, centroid_array):
            trench_pos += distance_between_trenches*self.m/self.pixel_size
            if bacterias == []:
                continue
            for bacteria, centroid in zip(bacterias, centroids):
                # Check that bacteria emitted wavelength is compatible with microscope
                if (bacteria.em_wavelength != self.em_wavelength or
                        bacteria.ex_wavelength != self.ex_wavelength):
                    raise ValueError(
                        "Bacteria and microscope must have compatible wavelengths")

                for x, y, z in np.array(bacteria.samples):
                    location_x = self.m*x/self.pixel_size \
                                  + centroid[1]
                    location_y = self.m*y/self.pixel_size \
                                  + centroid[0] + trench_pos
                    #TODO change this so it check x and y not out of boundary as well not only negative
                    photons_emitted = np.random.normal([location_x, location_y], sigma, (np.random.poisson(photons), 2))
                    for xp, yp in photons_emitted:
                        if xp > -0.5 and yp > -0.5 and xp < self.image.shape[0]-0.5 and yp < self.image.shape[1] - 0.5:
                            self.image[int(round(xp)), int(round(yp))] += gain

        # Add noise
        self.image = self.image + np.random.poisson(noise, shape)

        return self.image

    def image_MM(self, bacteria_array, centroid_array, shape, distance_between_trenches, sigma = 0.0, photons=30, noise=200, gain = 1.0):

        # Preset value of sigma to fit 2D Guassian to theoretical Airy Disk from rayleigh criterion 
        if sigma == 0.0:
            sigma = self.m*self.rayleigh_criterion / (2.9*self.pixel_size)

        # Create array to store image
        self.image = np.zeros(shape)

        # Emission of photons
        total_photons = 0
        trench_pos = 0

        # Centroids are relative to the trench
        for bacterias, centroids in zip(bacteria_array, centroid_array):
            trench_pos += distance_between_trenches*self.m/self.pixel_size
            if bacterias == []:
                continue
            for bacteria, centroid in zip(bacterias, centroids):
                # Check that bacteria emitted wavelength is compatible with microscope
                if (bacteria.em_wavelength != self.em_wavelength or
                        bacteria.ex_wavelength != self.ex_wavelength):
                    raise ValueError(
                        "Bacteria and microscope must have compatible wavelengths")

                for x, y, z in np.array(bacteria.samples):
                # chanige b_samples x min por radius -> since its always gonna be smallest
                    location_x = round(self.m*x/self.pixel_size) \
                                      + centroid[1]
                    location_y = round(self.m*y/self.pixel_size) \
                                      + centroid[0] + trench_pos
                    #TODO change this so it check x and y not out of boundary as well not only negative
                    if location_x > 0 and location_y > 0 and location_x < self.image.shape[0] and location_y < self.image.shape[1]:
                        photons_emitted = np.random.poisson(photons)
                        self.image[int(location_x), int(location_y)] += gain*photons_emitted
                        total_photons += gain*photons_emitted

        # Apply convolution with gaussian filter
        self.image = gaussian_filter(self.image, sigma=sigma)
        # Normalize so the number of phtons is the number of total_photons emitters
        self.image = np.round(self.image*np.sum(self.image)/total_photons)
        # Add noise
        self.image = self.image + np.random.poisson(noise, shape)

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
        #TODO change title
#        plt.title("Bacteria Image")
        plt.imshow(image, cmap=cm, origin="lower")
        plt.show()

    def _transform_vertices(self, verts, bacteria, centroid):
        verts = verts*self.m #magnification
        verts = verts / self.pixel_size #scaling by size of pixels
        verts[:,[0, 1]] = verts[:,[1, 0]] #make horizontal
        verts = verts + centroid # move by centroid
        return verts

    def display_image_with_boundary(self, image, bacteria, centroid):
        #TODO extend this to work with trenches and MM
        fig, ax = plt.subplots()
        # Create red color map
        colors = [(0, 0, 0), (1, 0, 0)]
        cm = LinearSegmentedColormap.from_list('test', colors, N=np.amax(image))

        # Display spline
        verts_spline = bacteria.spline[:, :-1] # not the z-axis
        verts_spline = self._transform_vertices(verts_spline, bacteria, centroid)
        ax.plot(verts_spline[:, 0], verts_spline[:, 1], 'y', label='spline')

        # Display boundary
        verts_boundary = np.array(list(map(list, bacteria.boundary)))
        verts_boundary = self._transform_vertices(verts_boundary, bacteria, centroid)
        ax.plot(verts_boundary[:, 0], verts_boundary[:, 1], 'g', label='boundary')

        # Display image
        plt.imshow(image, cmap=cm, origin="lower")
        plt.title("Bacteria Image with cell Boundary")
        plt.legend(fontsize='x-small')
        plt.show()

