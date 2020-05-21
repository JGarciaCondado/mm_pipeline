import numpy as np
import sys
sys.path.append('../')
from models import SpherocylindricalBacteria, Microscope
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    n = 10000 # number of images
    ex_wv = 0.490  # emmitted wavelength by microscope for excitation
    em_wv = 0.470  # emitted wavelength due to fluorescence
    pixel_size = 6.6  # pixel size
    NA = 1.35  # Numerical aperture
    magnification = 60  # magnification
    shape = (80, 30)

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
        while l < 1.0 or l>8.0: #ensure cells are greater than one micrometer
            l = np.random.normal(4.59, 1.51)
        R = (2*np.random.randint(0,2)-1)*np.random.uniform(1, 100)*l
        theta = np.random.normal(0, 5.96) # in degrees
        sigma = 4  #in pixels
        density = np.random.normal(1700, 100)
        cx = np.random.normal(0.66, 0.09)
        cy = np.random.normal(0.49, 0.05)
        centroid = [cx*30, cy*80]
        bacteria = SpherocylindricalBacteria(l,r,R,theta,density,ex_wv,em_wv)

        im = microscope.image_bacteria(bacteria, centroid, shape, sigma, gain=10)
        im_gt = microscope.image_bacteria_ground_truth(bacteria, centroid, shape)
        im_gt_stack.append(im_gt)
        im_stack.append(im)
        params.append([r,l,R,theta, centroid])

    #TODO save as .npz file with column names image/gt/params/
    np.save("data/synthetic_dataset/im_stack.npy", im_stack)
    np.save("data/synthetic_dataset/im_gt_stack.npy", im_gt_stack)
    np.save("data/synthetic_dataset/params.npy", params)

if __name__ == "__main__":
    main()
