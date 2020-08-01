import numpy as np
import sys
from shapely.geometry import Polygon, LineString, Point, LinearRing
from shapely.ops import nearest_points
sys.path.append('../')
from segmentation import segment_cell, boundary_from_pixelated_mask, smooth_boundary
from calibration.extraction import get_centerline, clean_centerline, debranch_centerline
from scipy import signal
from skimage.restoration import wiener

def gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def deblur_ch2(im, sigma, reg=30):
    min_val = np.min(im.shape)
    return wiener(im, gkern(min_val, sigma), reg)

def normalize_ch2(im, pixelated_mask):
    # Normalize with values within bacteria
    bacteria_px = np.argwhere(pixelated_mask==1.0)
    bac_x, bac_y = zip(*bacteria_px)
    min_val = np.min(im[bac_x, bac_y])
    max_val = np.max(im[bac_x, bac_y])
    im = (im-min_val)/(max_val-min_val)
    return im

def get_boundary(pixelated_mask, efd=5):
    # Obtain pixelated boundary
    boundary = boundary_from_pixelated_mask(pixelated_mask)
    # Smooth boundary
    smoothed_boundary = smooth_boundary(boundary, 5)
    return smoothed_boundary

def boundary2centerline(boundary):
    centerline = get_centerline(Polygon(boundary).buffer(0))
    centerline = clean_centerline(centerline)
    centerline = debranch_centerline(centerline)
    # Order spline to create line string that is straight
    spline = np.sort(np.array(centerline[0], dtype=([('xcoor', float), ('ycoor', float)])),order='ycoor')
    spline = [point for point in spline]
    spline = LineString(spline)
    return spline


def getExtrapoledLine(p1,p2, flip=False):
    'Creates a line extrapoled in p1->p2 direction'
    if flip:
        p1, p2 = p2, p1
    EXTRAPOL_RATIO = 200
    a = p1
    b = (p1[0]+EXTRAPOL_RATIO*(p2[0]-p1[0]), p1[1]+EXTRAPOL_RATIO*(p2[1]-p1[1]) )
    return LineString([a,b])


def extendcenterline(boundary, centerline):
    boundary = Polygon(boundary).buffer(0)
    boundary_ext = LinearRing(boundary.exterior.coords)#we only care about the boundary intersection
    l_coords = list(centerline.coords)
    up_line = getExtrapoledLine(*l_coords[-2:]) #we use the last two points
    down_line = getExtrapoledLine(*l_coords[:2], flip=True) #we use the last two points
    if boundary_ext.intersects(up_line) and boundary_ext.intersects(down_line):
        up_intersection_points = boundary_ext.intersection(up_line)
        down_intersection_points = boundary_ext.intersection(down_line)
        new_up_point_coords = list(up_intersection_points.coords)[0] #
        new_down_point_coords = list(down_intersection_points.coords)[0] #
        l_coords.insert(0, new_down_point_coords)
        l_coords.append(new_up_point_coords)
        new_extended_line = LineString(l_coords)
    else:
        raise Exception("Something went really wrong")
    return new_extended_line

def estimate_r(boundary, spline, pm):
    r = 0.0
    for point in boundary:
        r += pm*spline.distance(Point(point))/boundary.shape[0]
    return r

def volume_elements(pixelated_mask, spline, r, pm):

    # Volume of deepest section
    v_max = 2*r*pm**2

    # Obtain volume elements
    volume_elements = np.empty(pixelated_mask.shape)
    volume_elements[:] = np.nan
    for px in np.argwhere(pixelated_mask==1.0):
        pixel = Point((px[1], px[0]))
        # Calculate distance to pixel from centerline
        d2c = pm*spline.distance(pixel)
        # If bigger than radius ignore that pixel
        if d2c < r:
            v = 2*np.sqrt(r**2-d2c**2)*pm**2
            # Avoid very small volumes as well
            if v > v_max*0.1:
                volume_elements[px[0], px[1]] = 1/v
    return volume_elements

def fluorescence_density(im, volume_elements):
    return im*volume_elements

def quantiles(im, n_quantiles):
    # Remove NaN
    px_val = [value for value in im.flatten() if not np.isnan(value)]
    # Sort values
    px_val = np.sort(px_val)
    elements_per_quantile = np.floor(len(px_val)/n_quantiles)
    quantiles = [px_val[int(i*elements_per_quantile):int((i+1)*elements_per_quantile)] for i in range(n_quantiles)]
    return quantiles

def average_r_quantile(im, quantile, spline, r, pm):
    distance = []
    for index in np.argwhere((im>=quantile[0]) & (im<=quantile[-1])):
        coor = Point((index[1], index[0]))
        # Calculate distance to pixel from centerline
        d2c = pm*spline.distance(coor)/r
        distance.append(d2c)
    return np.mean(distance)

def average_s_quantile(im, quantile, spline):
    distance = []
    centroid = spline.centroid
    distance_center = spline.project(centroid)
    total_length = spline.length/2
    for index in np.argwhere((im>=quantile[0]) & (im<=quantile[-1])):
        coor = Point((index[1], index[0]))
        # Calculate distance to pixel from centerline
        dac = (spline.project(coor)-distance_center)/total_length
        distance.append(abs(dac))
    return np.mean(distance)

def r_quantile(im, quantile, spline, r, pm):
    distance = []
    for index in np.argwhere((im>=quantile[0]) & (im<=quantile[-1])):
        coor = Point((index[1], index[0]))
        # Calculate distance to pixel from centerline
        d2c = pm*spline.distance(coor)/r
        cp = nearest_points(spline, coor)[0]
        d2c = np.sign(coor.x-cp.x)*d2c
        distance.append(d2c)
    return distance

def s_quantile(im, quantile, spline):
    distance = []
    centroid = spline.centroid
    distance_center = spline.project(centroid)
    total_length = spline.length/2
    for index in np.argwhere((im>=quantile[0]) & (im<=quantile[-1])):
        coor = Point((index[1], index[0]))
        # Calculate distance to pixel from centerline
        dac = (spline.project(coor)-distance_center)/total_length
        distance.append(dac)
    return distance
