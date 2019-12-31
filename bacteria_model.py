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
from tqdm import tqdm

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
            pbar = tqdm(total=self.n)
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
                        pbar.update(1)
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
        self.fn = fn
        self.theta = theta*np.pi/180
        self.rotation_matrix = np.array(((np.cos(self.theta), -np.sin(self.theta), 0.0), (np.sin(self.theta), np.cos(self.theta), 0.0), (0.0, 0.0, 1.0)))
        self.spline = [[x, fn(x), 0.0] for x in np.arange(0, self.l+dx, dx)]
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
            pbar = tqdm(total=self.n)
            [x_max, y_max, z_max] = np.amax(self.spline, axis=0) + self.r
            [x_min, y_min, z_min] = np.amin(self.spline, axis=0) - self.r

            while(i < self.n):
                # lengthscale is alawys gonna be bigger
                x_sample = np.random.uniform(x_min, x_max)
                y_sample = np.random.uniform(y_min, y_max)
                z_sample = np.random.uniform(y_min, y_max)
                sample = np.array([x_sample, y_sample, z_sample])
                if x_sample < 0.0:
                    if np.linalg.norm(sample-np.array([0.0, 0.0, 0.0])) < self.r:
                        rotated_sample = self.rotation_matrix.dot(sample)
                        self.b_samples.append(rotated_sample)
                        i += 1
                        pbar.update(1)
                elif x_sample > self.l:
                    if np.linalg.norm(sample-np.array([self.l, 0.0, 0.0])) < self.r:
                        rotated_sample = self.rotation_matrix.dot(sample)
                        self.b_samples.append(rotated_sample)
                        i += 1
                        pbar.update(1)
                elif np.linalg.norm(sample-np.array([x_sample, self.fn(x_sample), 0.0])) < self.r:
                    rotated_sample = self.rotation_matrix.dot(sample)
                    self.b_samples.append(rotated_sample)
                    i += 1
                    pbar.update(1)
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
