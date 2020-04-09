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
        elif(l > 2*R):
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
        return np.sqrt(self.R**2 - x**2) - self.R

    def _sample(self):
        """ Samples point within the bacteria cell boundary.

        This method is used when initializign a bacteria to generate samples
        that represent infinitesimally small light sources of fluorescence.
        """

        # Sample until certain number of iteration or a certain number of
        # samples obtaines
        i = 0
        samples = []
        while(i < self.n):
            # lengthscale is alawys gonna be bigger
            x_sample = np.random.uniform(self.x_min, self.x_max)
            y_sample = np.random.uniform(self.y_min, self.y_max)
            z_sample = np.random.uniform(self.z_min, self.z_max)
            sample = np.array([x_sample, y_sample, z_sample])
            phi = np.arctan(x_sample/(self.R+y_sample))
            if phi < self.phi_min:
                if np.linalg.norm(sample-np.array([-self.l/2, self._fn(-self.l/2), 0.0])) < self.r:
                    rotated_sample = self.rotation_matrix.dot(sample)
                    samples.append(rotated_sample)
                    i += 1
            elif phi > self.phi_max:
                if np.linalg.norm(sample-np.array([self.l/2, self._fn(self.l/2), 0.0])) < self.r:
                    rotated_sample = self.rotation_matrix.dot(sample)
                    samples.append(rotated_sample)
                    i += 1
            else:
                x_int = self.R*np.sin(phi)
                y_int = self.R*(np.cos(phi)-1)
                if np.linalg.norm(sample-np.array([x_int, y_int, 0.0])) < self.r:
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
