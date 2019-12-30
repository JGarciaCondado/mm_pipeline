from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import sys
sys.path.append('../')
from molyso.molyso.generic.rotation import find_rotation, apply_rotate_and_cleanup
from molyso.molyso.mm.channel_detection import find_channels
from molyso.molyso.mm.cell_detection import find_cells_in_channel
from molyso.molyso.debugging import DebugPlot
from molyso.molyso.generic.signal import hamming_smooth, simple_baseline_correction, find_extrema_and_prominence, vertical_mean
from molyso.molyso.generic.otsu import threshold_otsu

channel_width = 24

im = imread('test_data.tif')
[im_top, im_bottom] = np.vsplit(im, 2)
im_top_flip = np.flip(im_top)
im_bottom_rot = apply_rotate_and_cleanup(im_bottom, find_rotation(im_bottom))[0] #only interested in cropped image
im_top_flip_rot = apply_rotate_and_cleanup(im_top_flip, find_rotation(im_top_flip))[0] #only interested in cropped image
positions, (upper, lower) = find_channels(im_bottom_rot)
im_bottom_rot_cr = im_bottom_rot[upper:lower, :]


colors = [(0, 0, 0), (1, 0, 0)]
cm = LinearSegmentedColormap.from_list('test', colors, N=np.amax(im))

#h = np.sum(im_bottom_rot_cr, axis = 0)/im_bottom_rot_cr.shape[0]
#plt.plot(range(len(h)), h)
#plt.show()
#h_grad = np.gradient(h)
#plt.plot(range(len(h)), h_grad)
#plt.show()
#v = np.sum(im_bottom_rot_cr, axis = 1)/im_bottom_rot_cr.shape[1]
#plt.plot(range(len(v)), v)
#plt.show()
avg_positions = np.sum(positions, axis = 1)/2 #use numpy mean lol
for avg_position in avg_positions[:1]:
    channel = im_bottom_rot_cr[:, int(avg_position)-12:int(avg_position)+12]
    plt.imshow(channel, cmap=cm)
    plt.show()
#h_channel = np.sum(channel, axis = 0)/channel.shape[0]
#plt.plot(range(len(h_channel)), h_channel)
#plt.show()
#v_channel = np.sum(channel, axis = 1)/channel.shape[1]
#plt.plot(range(len(v_channel)), v_channel)
#plt.show()

#DebugPlot.force_active = True
#DebugPlot.post_figure = 'show'
#channel_inverse = channel*-1 + np.max(channel)
#cells = find_cells_in_channel(channel_inverse)
#print(cells)

profile = vertical_mean(channel)
profile = simple_baseline_correction(profile)
profile = hamming_smooth(profile, 10)
extrema = find_extrema_and_prominence(profile, 15)
positions = [_pos for _pos in extrema.minima if extrema.prominence[_pos] > 0]
positions = positions + [profile.size]
cells = [[_last_pos, _pos] for _last_pos, _pos in zip([0] + positions, positions)]
for start, end in cells:
    if end - start < 10:
        pass
    elif start < 2: #or end > profile.size-2:
        plt.imshow(channel[start:end, :], cmap=cm)
#        plt.imshow(channel[start:end, :]>(threshold_otsu(channel[start:end, :])*1.0))
        plt.show()
    else:
        plt.imshow(channel[start-2:end+2, :], cmap=cm)
        plt.show()
        binary_image = channel[start-2:end+2, :]>(threshold_otsu(channel[start-2:end+2, :])*1.0)
        rows, columns = np.where(binary_image == False)
        unique, counts = np.unique(rows, return_counts=True)
        blank_rows = unique[np.where(counts == channel_width)]
#TODO create a function that you give it channel width in pixels that you want image to create
        init_pos = 0
        end_pos = 0
        for _pos in blank_rows:
            if end_pos + 1 == _pos:
                end_pos = _pos
            else:
                init_pos = _pos
                end_pos = _pos
#TODO determine where to remove them
        if init_pos != 0:
            plt.imshow(channel[start-2:end+2-end_pos+init_pos, :], cmap=cm)
            plt.show()
        else:
            plt.imshow(channel[start+end_pos-2:end+2, :], cmap=cm)
            plt.show()
#for start, end in cells:
#    plt.imshow(channel[start:end, :])
#    plt.show()
