import numpy as np
import sys
sys.path.append('../')

from contour import contour_real
from shapely.geometry import Polygon
from centerline.geometry import Centerline
from models import SpherocylindricalBacteria, Microscope
from scipy import ndimage, misc
from scipy.interpolate import splrep, sproot, splev
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

#TODO calculate distance from end points to min and max to determine whether its awat from r!
#Maybe create a centerline class?
#Has clean, debranch, theta, l, and R
def get_centerline(boundary):
    #What does centerline consit of?
    # Also maybe we only need coordinates i.e. do list geoms already
    centerline = Centerline(boundary)
    return centerline

def clean_centerline(centerline):
    end_points = []
    spline = []
    branch = []
    for line in list(centerline.geoms):
        coordinates = list(line.coords)
        for coor in coordinates:
            if coor not in branch:
                if coor not in spline:
                    if coor not in end_points:
                        end_points.append(coor)
                    else:
                        end_points.remove(coor)
                        spline.append(coor)
                else:
                    spline.remove(coor)
                    branch.append(coor)

    min_coor = (1000.0, 1000.0)
    max_coor = (0.0, 0.0)
    for branch_coor in branch:
        if branch_coor[1] < min_coor[1]:
            min_coor = branch_coor
        if branch_coor[1] > max_coor[1]:
            max_coor = branch_coor
    end_branches = [max_coor, min_coor]
    return [end_points, spline, branch, end_branches]


def debranch_centerline(centerline):
    [end_points, spline, branch, end_branches] = centerline
    [max_coor, min_coor] = end_branches
    if branch != []:
        #save all the removed for plotting purposes // maybe optional argument?
        removed = []
        for end_coor in end_points:
            distanceToBranch = 1000
            closest_branch = 0.0
            for branch_coor in branch:
                if distanceToBranch > np.sqrt((end_coor[0]-branch_coor[0])**2 + (end_coor[1]-branch_coor[1])**2):
                    distanceToBranch = np.sqrt((end_coor[0]-branch_coor[0])**2 + (end_coor[1]-branch_coor[1])**2)
                    close_branch = branch_coor
            if distanceToBranch < 4.0 or close_branch not in end_branches:
                remove_coor = []
                for spline_coor in spline:
                    if np.sqrt((end_coor[0]-spline_coor[0])**2 + (end_coor[1]-spline_coor[1])**2) < distanceToBranch:
                        remove_coor.append(spline_coor)
                        removed.append(spline_coor)
                for coor in remove_coor:
                    spline.remove(coor)
            # need to create then endpoints as branch // maybe pop the end_points?
            else:
                if end_coor[1] < min_coor[1]:
                    min_coor = end_coor
                if end_coor[1] > max_coor[1]:
                    max_coor = end_coor
    else:
        branch = end_points

    for branch_coor in branch:
        spline.append(branch_coor)

    return [spline, [max_coor, min_coor], removed]


def calculate_theta(centerline):
    [spline, [max_coor, min_coor], remove_coors] = centerline
    theta = np.arctan((max_coor[0]-min_coor[0])/(max_coor[1]-min_coor[1]))*180/np.pi
    return theta

def calculate_l(centerline, pixel_size, m):
    [spline, [max_coor, min_coor], remove_coor] = centerline
    l = (max_coor[1]-min_coor[1])*pixel_size/m
    return l

# max and min coor to the spline
def convert_centerline(centerline, theta, pixel_size, m, l):
    [spline, [max_coor, min_coor], remove_coor] = centerline
    theta = theta*np.pi/180
    rotation_matrix = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    for i, coor in enumerate(spline):
        new_coor = np.subtract(coor,min_coor)
        new_coor = rotation_matrix.dot(new_coor)
        new_coor = new_coor*pixel_size/m
        new_coor[1] = new_coor[1]-l/2
        new_coor = new_coor[::-1]
        spline[i] = new_coor
    return spline

def calculate_R(spline, l):
    def fn(x, R):
        return np.sign(R)*(np.sqrt(R**2 - x**2) - np.sqrt(R**2-(l**2)/4))

    x, y = zip(*spline)
    if np.mean(y) > 0:
        popt, pcov = curve_fit(fn, x, y, bounds=(l, 50*l))
    else:
        popt, pcov = curve_fit(fn, x, y, bounds=(-50*l, -l))

    return popt[0]

def horizontal_probability_density(im):
    #Treat zeros as NaNs for mean
    im = im.astype('float')
    im[im == 0.0] = np.nan
    counts = np.nanmean(im, axis=0)
    counts = counts - np.min(counts)
    counts = counts/np.sum(counts)
    return counts

def extract_density_photons(im, pixel_size, m, r, noise):
    #Treat zeros as NaNs for mean
    im = im.astype('float')
    im[im == 0.0] = np.nan
    counts = np.nanmean(im, axis=0)
    counts = counts - noise
    total_counts = np.sum(counts)
    volume = pixel_size/m*np.pi*r**2
    density = total_counts/volume
    return density

class MultiplePeaks(Exception): pass
class NoPeaksFound(Exception): pass

def fwhm(x, y, k=3):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    half_max = np.max(y)/2.0
    s = splrep(x, y - half_max, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros).")
    else:
        return abs(roots[1] - roots[0])

def calculate_r(cell, l, R, theta, shape, y_max, y_min, centroid):
    #TODO this constants have to be defined elswhere // maybe ask for a microscope class, density and bacteria as well
    #Number of samples can make FWHM fail
    ex_wv = 0.8  # emmitted wavelength by microscope for excitation
    em_wv = 0.59  # emitted wavelength due to fluorescence
    pixel_size = 4.4  # pixel size
    NA = 0.95  # Numerical aperture
    magnification = 40  # magnification

    microscope = Microscope(magnification, NA, ex_wv, em_wv, pixel_size)
    cell_rot = ndimage.rotate(cell, -theta, reshape=False, cval=230)
    counts = horizontal_probability_density(cell_rot[y_min:y_max, :])
    org = fwhm(range(len(counts)), counts)
    r_values = np.arange(0.4, 0.7, 0.01)
    prev_fwhm = 0.0
    r_final = 0.5
    for r in r_values:
        bacteria = SpherocylindricalBacteria(l, r, R, theta, 1700,  ex_wv, em_wv)
        image = microscope.image_bacteria(bacteria, centroid, shape)
        im_rot = ndimage.rotate(image, -theta, reshape=False, cval=0.0)
        counts = horizontal_probability_density(im_rot[y_min:y_max, :])
        current_fwhm = fwhm(range(len(counts)), counts)
        if current_fwhm > org:
            r_final = r - 0.01*((current_fwhm-org)/(current_fwhm-prev_fwhm))
            break
        prev_fwhm = current_fwhm
    return r_final

def kolmogrov_smirnov_test(dist1, dist2):
    #TODO check that dist1 and dist2 same size
    #Assumes dist1 and dist2 are p(x) at each 1 position
    F1 = np.cumsum(dist1)
    F2 = np.cumsum(dist2)
    abs_diff = np.abs(F1-F2)
    return np.max(abs_diff)

def grid_search_r_sigma(dist, l, R, theta, shape, y_max, y_min, centroid, microscope, ex_wv, em_wv):
    r_values = np.arange(0.4, 0.6, 0.01)
    sigma_values = np.arange(3, 5, 0.1)
    opt_r, opt_sigma = 0.0, 0.0
    test_val = 1
    for r in r_values:
        bacteria = SpherocylindricalBacteria(l, r, R, theta, 1700,  ex_wv, em_wv)
        for sigma in sigma_values:
            image = microscope.image_bacteria(bacteria, centroid, shape, sigma = sigma)
            im_rot = ndimage.rotate(image, -theta, reshape=False, cval=0.0)
            counts = horizontal_probability_density(im_rot[y_min:y_max, :])
            test_result = kolmogrov_smirnov_test(dist, counts)
            if test_result < test_val:
                test_val = test_result
                opt_r, opt_sigma = r, sigma

    return opt_r, opt_sigma


def extract_distribution(cell):
    m = 40
    pixel_size = 4.4
    rm_indices = np.where(cell==0.0)[0]
    cell = np.delete(cell, rm_indices, axis=0)
    shape = cell.shape
    length = shape[0]
    contour = contour_real(cell, 1.0)
    active_contour = contour.active_contour
    boundary = Polygon(active_contour).buffer(0)
    [minx, miny, maxx, maxy] = boundary.bounds
    centroid = (boundary.centroid.x, boundary.centroid.y)
    centerline = get_centerline(boundary)
    centerline = clean_centerline(centerline)
    centerline = debranch_centerline(centerline)
    [spline, [max_coor, min_coor], remove_coords] = centerline
    y_max, y_min = int(max_coor[1]), int(min_coor[1])
    theta = calculate_theta(centerline)
    l = calculate_l(centerline, pixel_size, m)
    centerline = convert_centerline(centerline, theta, pixel_size, m, l)
    R = calculate_R(centerline, l)
    if R > l*4:
        cell_rot = ndimage.rotate(cell, -theta, reshape=False, cval=0.0)
        dist = horizontal_probability_density(cell_rot[y_min:y_max, :])
        return dist
    else:
        return  []

def extract_params(cell, m, pixel_size):
    rm_indices = np.where(cell==0.0)[0]
    cell = np.delete(cell, rm_indices, axis=0)
    shape = cell.shape
    length = shape[0]
    contour = contour_real(cell, 1.0)
    active_contour = contour.active_contour
    boundary = Polygon(active_contour).buffer(0)
    [minx, miny, maxx, maxy] = boundary.bounds
    centroid = (boundary.centroid.x, boundary.centroid.y)
    centerline = get_centerline(boundary)
    centerline = clean_centerline(centerline)
    centerline = debranch_centerline(centerline)
    [spline, [max_coor, min_coor], remove_coords] = centerline
    y_max, y_min = int(max_coor[1]), int(min_coor[1])
    theta = calculate_theta(centerline)
    l = calculate_l(centerline, pixel_size, m)
    centerline = convert_centerline(centerline, theta, pixel_size, m, l)
    R = calculate_R(centerline, l)

    return l, R, theta


def extract_all_params(cell, m, pixel_size):
    shape = cell.shape
    length = shape[0]
    contour = contour_real(cell, 1.0)
    active_contour = contour.active_contour
    boundary = Polygon(active_contour).buffer(0)
    centroid = (boundary.centroid.x, boundary.centroid.y)
    [minx, miny, maxx, maxy] = boundary.bounds
    centroid = (boundary.centroid.x, boundary.centroid.y)
    centerline = get_centerline(boundary)
    centerline = clean_centerline(centerline)
    centerline = debranch_centerline(centerline)
    [spline, [max_coor, min_coor], remove_coords] = centerline
    y_max, y_min = int(max_coor[1]), int(min_coor[1])
    theta = calculate_theta(centerline)
    l = calculate_l(centerline, pixel_size, m)
    centerline = convert_centerline(centerline, theta, pixel_size, m, l)
    R = calculate_R(centerline, l)

    return l, R, theta, centroid
def extract_cell_info(filename):
    pixel_size = 4.4  # pixel size
    NA = 0.95  # Numerical aperture
    m = 40  # magnification
    cell = np.load(filename)
    rm_indices = np.where(cell==0.0)[0]
    cell = np.delete(cell, rm_indices, axis=0)
    shape = cell.shape
    length = shape[0]
    contour = contour_real(cell, 1.0)
    active_contour = contour.active_contour
    boundary = Polygon(active_contour).buffer(0)
    [minx, miny, maxx, maxy] = boundary.bounds
    centroid = (boundary.centroid.x, boundary.centroid.y)
    centerline = get_centerline(boundary)
    centerline = clean_centerline(centerline)
    centerline = debranch_centerline(centerline)
    [spline, [max_coor, min_coor], remove_coords] = centerline
    y_max, y_min = int(max_coor[1]), int(min_coor[1])
    theta = calculate_theta(centerline)
    l = calculate_l(centerline, pixel_size, m)
    centerline = convert_centerline(centerline, theta, pixel_size, m, l)
    R = calculate_R(centerline, l)
    if R > l*4:
        r = calculate_r(cell, l, R, theta, shape, y_max, y_min, centroid)
    else:
        r = None

    return length, centroid, theta, l, R, r
