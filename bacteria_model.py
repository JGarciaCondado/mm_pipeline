import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class Fluorescent_bacteria:

    def __init__(self, r, l, ex_wavelength, em_wavelength, n, n_total = True):
        if(l < r):
            raise ValueError("The length must be bigger than the radius")
        #all measurments must be in micrometers
        self.r = r
        self.l = l
        self.n = n
        self.ex_wavelength = ex_wavelength
        self.em_wavelength = em_wavelength
        self.b_samples = []
        self.b_samples_x = []
        self.b_samples_y = []
        self.b_samples_z = []
        self.sample(n_total)

    def sample(self, n_total):
        #Sample until certain number of iteration or a certain number of samples obtaines
        if (n_total):
            i = 0
            while(i<self.n):
                sample = np.random.uniform(-self.r-self.l/2, self.r+self.l/2, (1,3))[0] #lengthscale is alawys gonna be bigger
                [x, y, z] = sample
                #first check if the sample is wihtin the cylindrical body
                if (x < self.l/2 and x > -self.l/2):
                    #check that it is within the radius
                    if(y**2+z**2<self.r**2):
                        self.b_samples.append(sample)
                        i += 1
                else:
                    #if not it could be in the cap
                    if((abs(x)-self.l/2)**2+y**2+z**2 < self.r**2):
                        self.b_samples.append(sample)
                        i += 1

        else:
            sampled_data = np.random.uniform(-self.r-self.l/2, self.r+self.l/2, (self.n,3)) #lengthscale is alawys gonna be bigger

            for sample in sampled_data:
                [x, y, z] = sample
                #first check if the sample is wihtin the cylindrical body
                if (x < self.l/2 and x > -self.l/2):
                    #check that it is within the radius
                    if(y**2+z**2<self.r**2):
                        self.b_samples.append(sample)
                else:
                    #if not it could be in the cap
                    if((abs(x)-self.l/2)**2+y**2+z**2 < self.r**2):
                        self.b_samples.append(sample)

        self.b_samples_x, self.b_samples_y, self.b_samples_z = list(zip(*self.b_samples))

    def plot_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.b_samples_x, self.b_samples_y, self.b_samples_z, s=1, c='r', marker='o')

        ax.set_xlim([-self.r-self.l/2, self.r+self.l/2])
        ax.set_ylim([-self.r-self.l/2, self.r+self.l/2])
        ax.set_zlim([-self.r-self.l/2, self.r+self.l/2])

        plt.show()

    def plot_2D(self):
        plt.scatter(self.b_samples_x, self.b_samples_y, s=1)
        plt.xlim(-self.r-self.l/2, self.r+self.l/2)
        plt.ylim(-self.r-self.l/2, self.r+self.l/2)
        plt.show()
