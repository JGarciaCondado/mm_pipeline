from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Make folders in directory above available
import sys
sys.path.append('../')

# Import require molyso function
from molyso.generic.otsu import threshold_otsu
from molyso.generic.signal import hamming_smooth, \
    simple_baseline_correction, find_extrema_and_prominence, vertical_mean
from molyso.mm.channel_detection import find_channels
from molyso.generic.rotation import find_rotation, \
    apply_rotate_and_cleanup

from contour import contour_real

# Load image as numpy array
im = imread('test_data_fl.tif')

# Create red color map for image show
colors = [(0, 0, 0), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list('test', colors, N=np.amax(im))

# Display original image
plt.title('Original Image')
plt.imshow(im, cmap=cm)
plt.show()

# Split image into bottom and top channel
[im_top, im_bottom] = np.vsplit(im, 2)

# Flip top image
im_top_flip = np.flip(im_top)

# Show bottom image
plt.title('Bottom image')
plt.imshow(im_bottom, cmap=cm)
plt.show()

# Apply rotation to both images
im_bottom_rot = apply_rotate_and_cleanup(
    im_bottom, find_rotation(im_bottom))[0]
im_top_flip_rot = apply_rotate_and_cleanup(
    im_top_flip, find_rotation(im_top_flip))[0]

# Find positions of channels and top and bottom of channel
positions, (upper, lower) = find_channels(im_bottom_rot)

# Crop off top of channel and show
im_bottom_rot_cr = im_bottom_rot[upper:lower, :]
plt.title('Rotated and cropped bottom image')
plt.imshow(im_bottom_rot_cr, cmap=cm)
plt.show()

# Find center of channels. The positions given by MOLYSO are not
# accurate enough for fluorescent images.
avg_positions = np.mean(positions, axis=1)

# Define image width / channel width
channel_width = 26

# Get first channel in the image
channel = im_bottom_rot_cr[:, int(avg_positions[0] - channel_width / 2):
                           int(avg_positions[0] + channel_width / 2)]
plt.title('First Channel')
plt.imshow(channel, cmap=cm)
plt.show()

# Padding
padding = 2

# Remove big black spaces at the top or bottom of channel
# Perform otsu binarization
binary_image = channel > (threshold_otsu(channel) * 1.0)
plt.title('Otsu binarization of channel')
plt.imshow(binary_image)
plt.show()

# Determine rows that contain only blanks from otsu image
rows, columns = np.where(binary_image == False)
unique, counts = np.unique(rows, return_counts=True)
blank_rows = unique[np.where(counts == channel_width)]

# Determine areas of black

# Finid initial position
init_pos = 0
for _pos in blank_rows:
    if init_pos == _pos:
        init_pos += 1
    else:
        break

# Find end position
end_pos = channel.shape[0] - 1
for _pos in np.flip(blank_rows):
    if end_pos == _pos:
        end_pos -= 1
    else:
        break

channel = channel[init_pos - padding:end_pos + padding]
plt.title('Cropped channel')
plt.imshow(channel, cmap=cm)
plt.show()

# Obtain vertical smoothed intensity profile of channel
profile = vertical_mean(channel)
profile = simple_baseline_correction(profile)
profile = hamming_smooth(profile, 10)

# Find extrema in the profile by sliding window approach
extrema = find_extrema_and_prominence(profile, 15)

# Define cells positions
positions = [_pos for _pos in extrema.minima if extrema.prominence[_pos] > 0]
positions = positions + [profile.size]
# Check certain requirements to show they are cells
cells = [[_last_pos, _pos] for _last_pos, _pos in zip(
    [0] + positions, positions) if _pos - _last_pos > 10]

# Final padding added to make standard size
default_height = 50

# Show cells with final padding
N_half = int(np.ceil(len(cells) / 2))
n_rows = 2
fig, ax = plt.subplots()
fig.suptitle("Cells")
for n, points in enumerate(cells):
    start, end = points
    pad = (default_height - end + start) / 2
    if (pad).is_integer():
        cell = np.pad(channel[start:end, :], ((int(pad), int(
            pad)), (0, 0)), mode='constant', constant_values=0)
    else:
        cell = np.pad(channel[start:end, :], ((
            int(pad - 0.5), int(pad + 0.5)), (0, 0)),
            mode='constant', constant_values=0)
    ax = plt.subplot2grid((n_rows, N_half), (n // N_half, n % N_half))
    ax.imshow(cell, cmap=cm)
plt.show()
contour = contour_real(channel[start:end, :], 1.0)
contour.show_contours()
