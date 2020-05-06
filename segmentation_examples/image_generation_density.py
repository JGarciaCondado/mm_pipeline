import numpy as np
import sys
sys.path.append('../')
from models import SpherocylindricalBacteria, Microscope
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    n = 100 # number of images
    ex_wv = 0.8  # emmitted wavelength by microscope for excitation
    em_wv = 0.59  # emitted wavelength due to fluorescence
    pixel_size = 4.4  # pixel size
    NA = 0.95  # Numerical aperture
    magnification = 40  # magnification
    shape = (50, 26)

    #Create microscope image
    microscope = Microscope(magnification, NA, ex_wv, em_wv, pixel_size)

    #store images as array
    im_stack = []

    #Ground truth stack
    im_gt_stack = []

    #Density
    densities = [1100, 1300, 1500, 1700, 1900, 2100, 2300]
    photons = [10, 15, 20, 25, 30, 35, 40, 45, 50]

    for i in tqdm(range(n)):
        #all measurments are in micrometers
        r = np.random.uniform(0.4, 0.6)  # radius of cylinder caps in micrometers
        l = 0.0 # total length of cylindrical body excluding the caps
        while l < 1.0: #ensure cells are greater than one micrometer
            l = np.random.normal(2.57, 0.7)
        R = (2*np.random.randint(0,2)-1)*np.random.uniform(1, 50)*l
        theta = np.random.normal(0, 7.4) # in degrees
        sigma = 3.1 #in pixels
        centroid = [13, 25]
        centroid = centroid + np.random.normal(0,3,size=len(centroid))

        im_density = []
        im_density_gt = []
        for density in densities:
            bacteria = SpherocylindricalBacteria(l,r,R,theta,density,ex_wv,em_wv)
            im_photons = []
            im_photons_gt = []
            for photon_density in photons:
                im = microscope.image_bacteria(bacteria, centroid, shape, sigma, photons=photon_density)
                im_gt = microscope.image_bacteria_ground_truth(bacteria, centroid, shape)
                im_photons.append(im)
                im_photons_gt.append(im_gt)
            im_density.append(im_photons)
            im_density_gt.append(im_photons_gt)
        im_gt_stack.append(im_density_gt)
        im_stack.append(im_density)

    #TODO save as .npz file with column names image/gt/params/
    np.save("im_stack_pd.npy", im_stack)
    np.save("im_gt_stack_pd.npy", im_gt_stack)

if __name__ == "__main__":
    main()
