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

class Fluorescent_bacteria_rod:
    """ Build an epi-illumination microscope model.

    The model deals with both generating images from bacteria models
    and displaying such images.

    Parameters
    ----------
    r: radius of caps of rod-shaped bacteria (micrometers)
    l: length of cylinder of rod_shaped bacteria (micrometers)
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

    def __init__(self, r, l, ex_wavelength, em_wavelength, n, n_total=True):
        """Initialise constants."""

        # Check that length is greater than radius
        if(l < r):
            raise ValueError("The length must be bigger than the radius")

        self.r = r
        self.l = l
        self.n = n
        self.ex_wavelength = ex_wavelength
        self.em_wavelength = em_wavelength
        # Arrays to store the bacteria samples
        self.b_samples = []
        self.b_samples_x = []
        self.b_samples_y = []
        self.b_samples_z = []
        # Obtain samples within the bacteria
        self.sample(n_total)

    def sample(self, n_total):
        """ Samples point within the bacteria cell boundary.

        This method is used when initializign a bacteria to generate samples
        that represent infinitesimally small light sources of fluorescence.
        """

        # Sample until certain number of iteration or a certain number of
        # samples obtaines
        if (n_total):
            i = 0
            while(i < self.n):
                # lengthscale is alawys gonna be bigger
                sample = np.random.uniform(-self.r - self.l / 2,
                                           self.r + self.l / 2, (1, 3))[0]
                [x, y, z] = sample
                # first check if the sample is wihtin the cylindrical body
                if (x < self.l / 2 and x > -self.l / 2):
                    # check that it is within the radius
                    if(y**2 + z**2 < self.r**2):
                        self.b_samples.append(sample)
                        i += 1
                else:
                    # if not it could be in the cap
                    if((abs(x) - self.l / 2)**2 + y**2 + z**2 < self.r**2):
                        self.b_samples.append(sample)
                        i += 1

        else:
            # lengthscale is alawys gonna be bigger
            sampled_data = np.random.uniform(-self.r - self.l / 2,
                                             self.r + self.l / 2, (self.n, 3))

            for sample in sampled_data:
                [x, y, z] = sample
                # first check if the sample is wihtin the cylindrical body
                if (x < self.l / 2 and x > -self.l / 2):
                    # check that it is within the radius
                    if(y**2 + z**2 < self.r**2):
                        self.b_samples.append(sample)
                else:
                    # if not it could be in the cap
                    if((abs(x) - self.l / 2)**2 + y**2 + z**2 < self.r**2):
                        self.b_samples.append(sample)

        self.b_samples_x, self.b_samples_y, self.b_samples_z = list(
            zip(*self.b_samples))

    def plot_3D(self):
        """ Plot samples in a 3D plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            self.b_samples_x,
            self.b_samples_y,
            self.b_samples_z,
            s=1,
            c='r',
            marker='o')

        ax.set_xlim([-self.r - self.l / 2, self.r + self.l / 2])
        ax.set_ylim([-self.r - self.l / 2, self.r + self.l / 2])
        ax.set_zlim([-self.r - self.l / 2, self.r + self.l / 2])

        plt.show()

    def plot_2D(self):
        """ Plot samples in a 2D plot ignoring z-coordinate."""
        plt.scatter(self.b_samples_x, self.b_samples_y, s=1)
        plt.xlim(-self.r - self.l / 2, self.r + self.l / 2)
        plt.ylim(-self.r - self.l / 2, self.r + self.l / 2)
        plt.show()

class Fluorescent_bacteria_spline:
    """ Build an epi-illumination microscope model.

    The model deals with both generating images from bacteria models
    and displaying such images.

    Parameters
    ----------
    r: radius of caps of rod-shaped bacteria (micrometers)
    l: length of cylinder of rod_shaped bacteria (micrometers)
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

    def __init__(self, r, l, spline, ex_wavelength, em_wavelength, n, n_total=True):
        """Initialise constants."""

        # Check that length is greater than radius
        if(l < r):
            raise ValueError("The length must be bigger than the radius")

        self.r = r
        self.l = l
        self.n = n
        self.spline = spline
        self.ex_wavelength = ex_wavelength
        self.em_wavelength = em_wavelength
        # Arrays to store the bacteria samples
        self.b_samples = []
        self.b_samples_x = []
        self.b_samples_y = []
        self.b_samples_z = []
        # Obtain samples within the bacteria
        self.sample(n_total)

    def sample(self, n_total):
        """ Samples point within the bacteria cell boundary.

        This method is used when initializign a bacteria to generate samples
        that represent infinitesimally small light sources of fluorescence.
        """

        # Sample until certain number of iteration or a certain number of
        # samples obtaines
        if (n_total):
            i = 0
            [x_max, y_max, z_max] = np.amax(self.spline, axis=0) + self.r
            [x_min, y_min, z_min] = np.amin(self.spline, axis=0) - self.r

            while(i < self.n):
                # lengthscale is alawys gonna be bigger
                x_sample = np.random.uniform(x_min, x_max)
                y_sample = np.random.uniform(y_min, y_max)
                z_sample = np.random.uniform(y_min, y_max)
                sample = np.array([x_sample, y_sample, z_sample])
                for spline_value in self.spline:
                    if np.linalg.norm(sample-np.array(spline_value)) < self.r:
                        self.b_samples.append(sample)
                        i += 1
                        break
        else:
            #TODO copy above
            pass

        self.b_samples_x, self.b_samples_y, self.b_samples_z = list(
            zip(*self.b_samples))


#FIX so that limits are different but spacing between ticks the same
    def plot_3D(self):
        """ Plot samples in a 3D plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            self.b_samples_x,
            self.b_samples_y,
            self.b_samples_z,
            s=1,
            c='r',
            marker='o')

        ax.set_xlim([np.amin(self.b_samples), np.amax(self.b_samples)])
        ax.set_ylim([np.amin(self.b_samples), np.amax(self.b_samples)])
        ax.set_zlim([np.amin(self.b_samples), np.amax(self.b_samples)])

        plt.show()

    def plot_2D(self):
        """ Plot samples in a 2D plot ignoring z-coordinate."""
        plt.scatter(self.b_samples_x, self.b_samples_y, s=1)
        plt.xlim(np.amin(self.b_samples), np.amax(self.b_samples))
        plt.ylim(np.amin(self.b_samples), np.amax(self.b_samples))
        plt.show()

class Fluorescent_bacteria_spline_fn:
    """ Build an epi-illumination microscope model.

    The model deals with both generating images from bacteria models
    and displaying such images.

    Parameters
    ----------
    r: radius of caps of rod-shaped bacteria (micrometers)
    l: length of cylinder of rod_shaped bacteria (micrometers)
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

    def __init__(self, r, l, dx, fn, theta, ex_wavelength, em_wavelength, n, n_total=True):
        """Initialise constants."""

        # Check that length is greater than radius
        if(l < r):
            raise ValueError("The length must be bigger than the radius")

        self.r = r
        self.l = l
        self.n = n
        self.dx=dx
        self.fn = fn
        self.theta = theta*np.pi/180
        self.rotation_matrix = np.array(((np.cos(self.theta), -np.sin(self.theta), 0.0), (np.sin(self.theta), np.cos(self.theta), 0.0), (0.0, 0.0, 1.0)))
        # TODO make dxx = 0.01 constant
        self.spline = np.array([[x, fn(x), 0.0] for x in np.arange(0, self.l+dx, dx)]).dot(np.transpose(self.rotation_matrix))
        self.max = np.amax(self.spline, axis=0) + self.r
        [self.x_max, self.y_max, self.z_max] = self.max
        self.min = np.amin(self.spline, axis=0) - self.r
        [self.x_min, self.y_min, self.z_min] = self.min
        self.boundary = self._boundary().dot(np.transpose(self.rotation_matrix[:, :-1]))
        self.ex_wavelength = ex_wavelength
        self.em_wavelength = em_wavelength
        # Arrays to store the bacteria samples
        self.b_samples = []
        self.b_samples_x = []
        self.b_samples_y = []
        self.b_samples_z = []
        # Obtain samples within the bacteria
        self._sample(n_total)

    def _boundary(self):
        verts_left_boundary = np.array([(x, self.fn(x)+self.r) for x in np.arange(0.0, self.l+self.dx, self.dx)])
        verts_right_boundary = np.array([(x, self.fn(x)-self.r) for x in np.arange(self.l, -self.dx , -self.dx)])
        verts_right_bottom_circle_boundary = np.array([(x, np.sqrt(self.r**2-x**2)) for x in np.arange(-self.r, self.dx, self.dx)])
        verts_left_bottom_circle_boundary = np.array([(x,-np.sqrt(self.r**2-x**2)) for x in np.arange(0.0, -self.r, -self.dx)])
        verts_left_top_circle_boundary = np.array([(x+self.l, np.sqrt(self.r**2-x**2)) for x in np.arange(0.0, self.r, self.dx)])
        verts_right_top_circle_boundary = np.array([(x+self.l, -np.sqrt(self.r**2-x**2)) for x in np.arange(self.r, -self.dx, -self.dx)])
        verts_boundary = np.concatenate((verts_left_boundary,verts_left_top_circle_boundary, verts_right_top_circle_boundary, verts_right_boundary,
                        verts_left_bottom_circle_boundary, verts_right_bottom_circle_boundary))

        return verts_boundary

    def _sample(self, n_total):
        """ Samples point within the bacteria cell boundary.

        This method is used when initializign a bacteria to generate samples
        that represent infinitesimally small light sources of fluorescence.
        """

        # Sample until certain number of iteration or a certain number of
        # samples obtaines
        if (n_total):
            i = 0
            x_max, x_min = self.l+self.r, -self.r
            y_max, y_min = np.amax([self.fn(x)+self.r for x in np.arange(0.0, self.l+self.dx, self.dx)]), -self.r
            z_max, z_min = self.r, -self.r
            while(i < self.n):
                # lengthscale is alawys gonna be bigger
                x_sample = np.random.uniform(x_min, x_max)
                y_sample = np.random.uniform(y_min, y_max)
                z_sample = np.random.uniform(z_min, z_max)
                sample = np.array([x_sample, y_sample, z_sample])
                if x_sample < 0.0:
                    if np.linalg.norm(sample-np.array([0.0, 0.0, 0.0])) < self.r:
                        rotated_sample = self.rotation_matrix.dot(sample)
                        self.b_samples.append(rotated_sample)
                        i += 1
                elif x_sample > self.l:
                    if np.linalg.norm(sample-np.array([self.l, 0.0, 0.0])) < self.r:
                        rotated_sample = self.rotation_matrix.dot(sample)
                        self.b_samples.append(rotated_sample)
                        i += 1
                elif np.linalg.norm(sample-np.array([x_sample, self.fn(x_sample), 0.0])) < self.r:
                    rotated_sample = self.rotation_matrix.dot(sample)
                    self.b_samples.append(rotated_sample)
                    i += 1
        else:
            #TODO copy above
            pass

        self.b_samples_x, self.b_samples_y, self.b_samples_z = list(
            zip(*self.b_samples))


# TODO so that limits are different but spacing between ticks the same
    def plot_3D(self):
        """ Plot samples in a 3D plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            self.b_samples_x,
            self.b_samples_y,
            self.b_samples_z,
            s=1,
            c='r',
            marker='o')

        ax.set_xlim([np.amin(self.b_samples), np.amax(self.b_samples)])
        ax.set_ylim([np.amin(self.b_samples), np.amax(self.b_samples)])
        ax.set_zlim([np.amin(self.b_samples), np.amax(self.b_samples)])
        plt.title("Bacteria samples in 3D")
        plt.show()

    def plot_2D(self):
        """ Plot samples in a 2D plot ignoring z-coordinate."""
        plt.title("Bacteria samples in 2D")
#        plt.scatter(self.b_samples_x, self.b_samples_y, s=1, label="samples")
        plt.plot(self.spline[:, 0], self.spline[:, 1], c='orange', label='spline')
        plt.plot(self.boundary[:, 0], self.boundary[:, 1], c='green', label='boundary')
        plt.xlim(np.amin(self.b_samples_x), np.amax(self.b_samples_x))
        plt.ylim(np.amin(self.b_samples_y), np.amax(self.b_samples_y))
        plt.axis('equal')
        plt.legend(loc=4)
        plt.show()

class bacteria_spline:
    """ Build an epi-illumination microscope model.

    The model deals with both generating images from bacteria models
    and displaying such images.

    Parameters
    ----------
    r: radius of caps of rod-shaped bacteria (micrometers)
    l: length of cylinder of rod_shaped bacteria (micrometers)
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

    def __init__(self, r, l, dx, fn, theta, ex_wavelength, em_wavelength, n, n_total=True):
        """Initialise constants."""

        # Check that length is greater than radius
        if(l < r):
            raise ValueError("The length must be bigger than the radius")

        self.r = r
        self.l = l
        self.n = n
        self.fn = fn
        self.dx = dx
        self.theta = theta*np.pi/180
        self.rotation_matrix = np.array(((np.cos(self.theta), -np.sin(self.theta), 0.0), (np.sin(self.theta), np.cos(self.theta), 0.0), (0.0, 0.0, 1.0)))
        # TODO make dxx = 0.01 constant
        self.spline = np.array([[x, fn(x), 0.0] for x in np.arange(0, self.l+dx, dx)]).dot(np.transpose(self.rotation_matrix))
        self.max = np.amax(self.spline, axis=0) + self.r
        [self.x_max, self.y_max, self.z_max] = self.max
        self.min = np.amin(self.spline, axis=0) - self.r
        [self.x_min, self.y_min, self.z_min] = self.min
        self.boundary = self._boundary().dot(np.transpose(self.rotation_matrix[:, :-1]))
        self.ex_wavelength = ex_wavelength
        self.em_wavelength = em_wavelength

    def _boundary(self):
        verts_left_boundary = np.array([(x, self.fn(x)+self.r) for x in np.arange(0.0, self.l+self.dx, self.dx)])
        verts_right_boundary = np.array([(x, self.fn(x)-self.r) for x in np.arange(self.l, -self.dx , -self.dx)])
        verts_right_bottom_circle_boundary = np.array([(x, np.sqrt(self.r**2-x**2)) for x in np.arange(-self.r, self.dx, self.dx)])
        verts_left_bottom_circle_boundary = np.array([(x,-np.sqrt(self.r**2-x**2)) for x in np.arange(0.0, -self.r, -self.dx)])
        verts_left_top_circle_boundary = np.array([(x+self.l, np.sqrt(self.r**2-x**2)) for x in np.arange(0.0, self.r, self.dx)])
        verts_right_top_circle_boundary = np.array([(x+self.l, -np.sqrt(self.r**2-x**2)) for x in np.arange(self.r, -self.dx, -self.dx)])
        verts_boundary = np.concatenate((verts_left_boundary,verts_left_top_circle_boundary, verts_right_top_circle_boundary, verts_right_boundary,
                        verts_left_bottom_circle_boundary, verts_right_bottom_circle_boundary))

        return verts_boundary

class BacteriaBoundary:
    """ Build an epi-illumination microscope model.

    The model deals with both generating images from bacteria models
    and displaying such images.

    Parameters
    ----------
    r: radius of caps of rod-shaped bacteria (micrometers)
    l: length of cylinder of rod_shaped bacteria (micrometers)
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

    def __init__(self, boundary, density, ex_wavelength, em_wavelength):
        """Initialise constants."""

        self.boundary = boundary
        self.density = density
        self.ex_wavelength = ex_wavelength
        self.em_wavelength = em_wavelength
        # Arrays to store the bacteria samples
        self.b_samples = []
        self.b_samples_x = []
        self.b_samples_y = []
        self.b_samples_z = []
        # Obtain samples within the bacteria
        self._sample()


    def _sample(self):
        """ Samples point within the bacteria cell boundary.

        This method is used when initializign a bacteria to generate samples
        that represent infinitesimally small light sources of fluorescence.
        """

        # Sample until certain number of iteration or a certain number of
        # samples obtaines
        i = 0
        x_max, x_min = self.l+self.r, -self.r
        y_max, y_min = np.amax([self.fn(x)+self.r for x in np.arange(0.0, self.l+self.dx, self.dx)]), -self.r
        z_max, z_min = self.r, -self.r
        while(i < self.n):
            # lengthscale is alawys gonna be bigger
            x_sample = np.random.uniform(x_min, x_max)
            y_sample = np.random.uniform(y_min, y_max)
            z_sample = np.random.uniform(z_min, z_max)
            sample = np.array([x_sample, y_sample, z_sample])
            if x_sample < 0.0:
                if np.linalg.norm(sample-np.array([0.0, 0.0, 0.0])) < self.r:
                    rotated_sample = self.rotation_matrix.dot(sample)
                    self.b_samples.append(rotated_sample)
                    i += 1
            elif x_sample > self.l:
                if np.linalg.norm(sample-np.array([self.l, 0.0, 0.0])) < self.r:
                    rotated_sample = self.rotation_matrix.dot(sample)
                    self.b_samples.append(rotated_sample)
                    i += 1
            elif np.linalg.norm(sample-np.array([x_sample, self.fn(x_sample), 0.0])) < self.r:
                rotated_sample = self.rotation_matrix.dot(sample)
                self.b_samples.append(rotated_sample)
                i += 1

        self.b_samples_x, self.b_samples_y, self.b_samples_z = list(
            zip(*self.b_samples))


# TODO so that limits are different but spacing between ticks the same
    def plot_3D(self):
        """ Plot samples in a 3D plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            self.b_samples_x,
            self.b_samples_y,
            self.b_samples_z,
            s=1,
            c='r',
            marker='o')

        ax.set_xlim([np.amin(self.b_samples), np.amax(self.b_samples)])
        ax.set_ylim([np.amin(self.b_samples), np.amax(self.b_samples)])
        ax.set_zlim([np.amin(self.b_samples), np.amax(self.b_samples)])
        plt.title("Bacteria samples in 3D")
        plt.show()

    def plot_2D(self):
        """ Plot samples in a 2D plot ignoring z-coordinate."""
        plt.title("Bacteria samples in 2D")
#        plt.scatter(self.b_samples_x, self.b_samples_y, s=1, label="samples")
        plt.plot(self.spline[:, 0], self.spline[:, 1], c='orange', label='spline')
        plt.plot(self.boundary[:, 0], self.boundary[:, 1], c='green', label='boundary')
        plt.xlim(np.amin(self.b_samples_x), np.amax(self.b_samples_x))
        plt.ylim(np.amin(self.b_samples_y), np.amax(self.b_samples_y))
        plt.axis('equal')
        plt.legend(loc=4)
        plt.show()

class SpherocylindricalBacteria:
    def __init__(self, r, l, R, theta, ex_wavelength, em_wavelength, density):
        """Initialise constants."""

        # Check that length is greater than radius
        if(l < r):
            raise ValueError("The length must be bigger than the radius")

        self.r = r
        self.l = l
        self.R = R
        self.theta = theta*np.pi/180
        dx = 0.01
        self.dx = dx
        self.density = density
        self.volume_cap = 2*np.pi*self.r**3/3
        self.volume_cylinder = (np.pi*self.r**2)*(self.R*(2*np.arcsin(self.l/(2*self.R))))
        self.volume = 2*self.volume_cap+self.volume_cylinder
        # Make density a funciton of volume
        self.n = int(self.volume*self.density)
        self.rotation_matrix = np.array(((np.cos(self.theta), -np.sin(self.theta), 0.0), (np.sin(self.theta), np.cos(self.theta), 0.0), (0.0, 0.0, 1.0)))
        self.spline = np.array([[x, self.fn(x), 0.0] for x in np.arange(-self.l/2, self.l/2+dx, dx)]).dot(np.transpose(self.rotation_matrix))
        self.max = np.amax(self.spline, axis=0) + self.r
        [self.x_max, self.y_max, self.z_max] = self.max
        self.min = np.amin(self.spline, axis=0) - self.r
        [self.x_min, self.y_min, self.z_min] = self.min
        self.boundary = self._boundary().dot(np.transpose(self.rotation_matrix[:, :-1]))
        self.ex_wavelength = ex_wavelength
        self.em_wavelength = em_wavelength
        # Arrays to store the bacteria samples
        self.b_samples = []
        self.b_samples_x = []
        self.b_samples_y = []
        self.b_samples_z = []
        # Obtain samples within the bacteria
        self._sample()

    def fn(self, x):
        return np.sqrt(self.R**2 - x**2) - self.R

    def _boundary(self):
        # Fix caps
        verts_left_boundary = np.array([(x, self.fn(x)+self.r) for x in np.arange(-self.l/2, self.l/2+self.dx, self.dx)])
        verts_right_boundary = np.array([(x, self.fn(x)-self.r) for x in np.arange(self.l/2, -self.l/2-self.dx , -self.dx)])
        verts_right_bottom_circle_boundary = np.array([(x-self.l/2, np.sqrt(self.r**2-x**2)) for x in np.arange(-self.r, self.dx, self.dx)])
        verts_left_bottom_circle_boundary = np.array([(x-self.l/2,-np.sqrt(self.r**2-x**2)) for x in np.arange(0.0, -self.r, -self.dx)])
        verts_left_top_circle_boundary = np.array([(x+self.l/2, np.sqrt(self.r**2-x**2)) for x in np.arange(0.0, self.r, self.dx)])
        verts_right_top_circle_boundary = np.array([(x+self.l/2, -np.sqrt(self.r**2-x**2)) for x in np.arange(self.r, -self.dx, -self.dx)])
        verts_boundary = np.concatenate((verts_left_boundary,verts_left_top_circle_boundary, verts_right_top_circle_boundary, verts_right_boundary,
                        verts_left_bottom_circle_boundary, verts_right_bottom_circle_boundary))

        return verts_boundary

    def _sample(self):
        """ Samples point within the bacteria cell boundary.

        This method is used when initializign a bacteria to generate samples
        that represent infinitesimally small light sources of fluorescence.
        """

        # Sample until certain number of iteration or a certain number of
        # samples obtaines
        i = 0
        x_max, x_min = self.l/2+self.r, -self.l/2-self.r
        y_max, y_min = self.r, -self.l#self.r, self.fn(self.l/2)-self.r
        z_max, z_min = self.r, -self.r
        while(i < self.n):
            # lengthscale is alawys gonna be bigger
            x_sample = np.random.uniform(x_min, x_max)
            y_sample = np.random.uniform(y_min, y_max)
            z_sample = np.random.uniform(z_min, z_max)
            sample = np.array([x_sample, y_sample, z_sample])
            if x_sample < -self.l/2:
                if np.linalg.norm(sample-np.array([-self.l/2, self.fn(-self.l/2), 0.0])) < self.r:
                    rotated_sample = self.rotation_matrix.dot(sample)
                    self.b_samples.append(rotated_sample)
                    i += 1
            elif x_sample > self.l/2:
                if np.linalg.norm(sample-np.array([self.l/2, self.fn(self.l/2), 0.0])) < self.r:
                    rotated_sample = self.rotation_matrix.dot(sample)
                    self.b_samples.append(rotated_sample)
                    i += 1
#            elif np.linalg.norm(sample-np.array([x_sample, self.fn(x_sample), 0.0])) < self.r:
            else:
                phi = np.arctan(x_sample/(self.R+y_sample))
                x_int = self.R*np.sin(phi)
                y_int = self.R*(np.cos(phi)-1)
                if np.linalg.norm(sample-np.array([x_int, y_int, 0.0])) < self.r:
                    rotated_sample = self.rotation_matrix.dot(sample)
                    self.b_samples.append(rotated_sample)
                    i += 1

        self.b_samples_x, self.b_samples_y, self.b_samples_z = list(
            zip(*self.b_samples))


# TODO so that limits are different but spacing between ticks the same
    def plot_3D(self):
        """ Plot samples in a 3D plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            self.b_samples_x,
            self.b_samples_y,
            self.b_samples_z,
            s=1,
            c='r',
            marker='o')

        ax.set_xlim([np.amin(self.b_samples), np.amax(self.b_samples)])
        ax.set_ylim([np.amin(self.b_samples), np.amax(self.b_samples)])
        ax.set_zlim([np.amin(self.b_samples), np.amax(self.b_samples)])
        plt.title("Bacteria samples in 3D")
        plt.axis('scaled')
        plt.show()

    def plot_2D(self):
        """ Plot samples in a 2D plot ignoring z-coordinate."""
        plt.title("Bacteria samples in 2D")
        plt.scatter(self.b_samples_x, self.b_samples_y, s=1, label="samples")
        plt.plot(self.spline[:, 0], self.spline[:, 1], c='orange', label='spline')
        plt.plot(self.boundary[:, 0], self.boundary[:, 1], c='green', label='boundary')
        plt.xlim(np.amin(self.b_samples_x), np.amax(self.b_samples_x))
        plt.ylim(np.amin(self.b_samples_y), np.amax(self.b_samples_y))
        plt.legend(loc=4)
        plt.axis('scaled')
        plt.show()

