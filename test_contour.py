import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

r_b = 0.5
l_b = 4
magnification = 40
pixel_size = 4.4
padding = 5

def spline_fn_curvature(x, R=20, l=4):
    return np.sqrt(R**2 - (x-l/2)**2) - np.sqrt(R**2-l**2/4)

def transform_vertices(verts):
    theta = 10*np.pi/180
    rotation_matrix = np.array(((np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta))))
    verts = verts.dot(np.transpose(rotation_matrix))
    verts = verts + r_b # move to get non-zero values
    verts = verts*magnification #magnification
    verts = verts / pixel_size #scaling by size of pixels
    verts = verts + padding # add padding
    return verts

verts_spline = np.array([(spline_fn_curvature(x), x) for x in np.arange(0.0, l_b+0.01, 0.01)])
verts_spline_transformed = transform_vertices(verts_spline)

codes = [Path.MOVETO] + [Path.LINETO] * (len(verts_spline_transformed) - 1)

path = Path(verts_spline_transformed, codes)

fig, ax = plt.subplots()
patch = patches.PathPatch(path, fill=False, lw=1, ec = 'orange')
ax.add_patch(patch)

verts_left_boundary = np.array([(spline_fn_curvature(x)+r_b, x) for x in np.arange(0.0, l_b+0.01, 0.01)])
verts_right_boundary = np.array([(spline_fn_curvature(x)-r_b, x) for x in np.arange(l_b, -0.01 , -0.01)])
verts_right_bottom_circle_boundary = np.array([(np.sqrt(r_b**2-x**2), x) for x in np.arange(-r_b, 0.01, 0.01)])
verts_left_bottom_circle_boundary = np.array([(-np.sqrt(r_b**2-x**2), x) for x in np.arange(0.0, -r_b - 0.01,-0.01)])
verts_left_top_circle_boundary = np.array([(np.sqrt(r_b**2-x**2), x+l_b) for x in np.arange(0.0, r_b+0.01, 0.01)])
verts_right_top_circle_boundary = np.array([(-np.sqrt(r_b**2-x**2), x+l_b) for x in np.arange(r_b, -0.01, -0.01)])
verts_boundary = np.concatenate((verts_left_boundary,verts_left_top_circle_boundary, verts_right_top_circle_boundary, verts_right_boundary,
                                verts_left_bottom_circle_boundary, verts_right_bottom_circle_boundary))
verts_boundary = transform_vertices(verts_boundary)
codes = [Path.MOVETO] + [Path.LINETO] * (len(verts_boundary) - 1)

path = Path(verts_boundary, codes)
patch = patches.PathPatch(path, fill=False, lw=1, ec = 'orange')
ax.add_patch(patch)

im = imread('test_bacteria.tif')
colors = [(0, 0, 0), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list('test', colors, N=np.amax(im))
plt.imshow(im, cmap = cm, origin="lower")
plt.show()
