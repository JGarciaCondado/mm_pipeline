import numpy as np
from models import SpherocylindricalBacteria, Microscope
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    n = 20000 # number of images
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
    #store values as array
    params = []

    #Ground truth stack
    im_gt_stack = []

    for i in tqdm(range(n)):
        #all measurments are in micrometers
        r = np.random.uniform(0.4, 0.6)  # radius of cylinder caps in micrometers
        l = 0.0 # total length of cylindrical body excluding the caps
        while l < 1.0: #ensure cells are greater than one micrometer
            l = np.random.normal(2.57, 0.7)
        R = (2*np.random.randint(0,2)-1)*np.random.uniform(1, 50)*l
        theta = np.random.normal(0, 7.4) # in degrees
        sigma = 3.1 #in pixels
        density = np.random.normal(1700, 100)
        centroid = [13, 25]
        centroid = centroid + np.random.normal(0,3,size=len(centroid))
        bacteria = SpherocylindricalBacteria(l,r,R,theta,density,ex_wv,em_wv)

        im = microscope.image_bacteria(bacteria, centroid, shape, sigma)
        im_gt = microscope.image_bacteria_ground_truth(bacteria, centroid, shape)
        im_gt_stack.append(im_gt)
        im_stack.append(im)
        params.append([r,l,R,theta, centroid])

    np.save("Dataset/im_stack.npy", im_stack)
    np.save("Dataset/im_gt_stack.npy", im_gt_stack)
    np.save("Dataset/params.npy", params)

if __name__ == "__main__":
    main()
