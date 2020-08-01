from tifffile import imread, imsave
import numpy as np

import re
import argparse
import os
import sys
sys.path.append('../')

# Import require molyso function
from molyso.generic.otsu import threshold_otsu
from molyso.generic.signal import hamming_smooth, \
    simple_baseline_correction, find_extrema_and_prominence, vertical_mean
# Tunable for number of channels must be changed to 200
from molyso.mm.channel_detection import find_channels
from molyso.generic.rotation import find_rotation, \
    apply_rotate_and_cleanup

import matplotlib.pyplot as plt

DEBUG = True

def debug(string):
    if DEBUG:
        print(string)

def remove_black_in_channel(ch, fl_ch, padding=2):
    # Remove big black spaces at the top or bottom of channel
    # Perform otsu binarization
    binary_image = ch > (threshold_otsu(ch) * 1.0)

    # Determine rows that contain only blanks from otsu image
    rows, columns = np.where(binary_image == False)
    unique, counts = np.unique(rows, return_counts=True)
    blank_rows = unique[np.where(counts == ch.shape[1])]

    # Determine areas of black

    # First check that at least the number of blak rows is bigger than 2
    if len(blank_rows) > 2:
        # Finid initial position
        init_pos = 0
        for _pos in blank_rows:
            if init_pos == _pos:
                init_pos += 1
            else:
                break

        # Find end position
        end_pos = ch.shape[0] - 1
        for _pos in np.flip(blank_rows):
            if end_pos == _pos:
                end_pos -= 1
            else:
                break

        if init_pos < padding or end_pos > ch.shape[0] - padding:
            return ch[init_pos:end_pos], fl_ch[init_pos:end_pos]
        else:
            return ch[init_pos - padding:end_pos + padding], fl_ch[init_pos-padding:end_pos+padding]
    else:
        return ch, fl_ch

def find_cells(ch):
    # Obtain vertical smoothed intensity profile of channel
    profile = vertical_mean(ch)
    profile = simple_baseline_correction(profile)
    profile = hamming_smooth(profile, 10)

    # Find extrema in the profile by sliding window approach
    extrema = find_extrema_and_prominence(profile, 5)

    # Define cells positions
    #Remove those with a minima that doesnt have a value smaller than 0
    positions = [_pos for _pos in extrema.minima if profile[_pos]< -50]
    positions = positions + [profile.size]

    # Check certain requirements to show they are cells
    # If smaller than 10 too small and if biggger than 70 too big
    cells = [[_last_pos, _pos] for _last_pos, _pos in zip(
             [0] + positions, positions)
             if _pos - _last_pos > 10 and _pos - _last_pos < 80]

    # Check that cells mean intensity value above some threshold
    cells = [[start, end] for start, end in cells if np.mean(ch[start:end, :]) > 2.0*10**3]

    return cells

def image_find_cells(im, im_fl, args, position, time, angle, ch_positions):

    debug("File: %s" % args.image)
    debug("Position: %s" % position)
    debug("Time: %s" % time)

    # Apply rotation to images
    im_rot = apply_rotate_and_cleanup(
                im, angle)[0]
    im_rot_fl = apply_rotate_and_cleanup(
                im_fl, angle)[0]

    # Define image width / channel width
    channel_width = 30

    # Number of channels
    n_channels = 0

    for center in ch_positions:
        n_channels += 1

        # Avoid indexing outside of bounds
        if center < channel_width / 2:
            center = channel_width / 2
        elif center > im_rot.shape[1] - channel_width / 2:
            center = im_rot.shape[1] - channel_width / 2

        #TODO make channel and cell extraction just deal with positions
        # crop all at the end at one go
        # Get channel in the image
        channel = im_rot[:, int(center - channel_width / 2):
                               int(center + channel_width / 2)]
        fl_channel = im_rot_fl[:, int(center - channel_width / 2):
                               int(center + channel_width / 2)]

        # Remove black ends of channel
        channel, fl_channel = remove_black_in_channel(channel, fl_channel)
        plt.imshow(channel)
        plt.show()

        # Find cells in channel
        cells = find_cells(channel)

        n_cells = 0
        if cells:
            # Show cells with final padding
            for start, end in cells:
                n_cells += 1
                cell_seg = channel[start:end, :]
                cell_fl = fl_channel[start:end, :]
                cell = np.stack([cell_seg, cell_fl], axis=0)
                np.save('%s/pos%s_time%s_channel_%s_cell_%s.npy' % (args.output_directory, position, time, n_channels, n_cells), cell)
        debug('Number of cells in channel %s: %s' % (n_channels, n_cells))
    debug('Number of channels found: %s' % (n_channels))

def find_ch_positions(im, angle):
    # Correct rotation
    im_rot = apply_rotate_and_cleanup(
                im, angle)[0]

    # Find positions of channels and top and bottom of channel
    positions, (upper, lower) = find_channels(im_rot)

    # Find center of channels. The positions given by MOLYSO are not
    # accurate enough for fluorescent images.
    avg_positions = np.mean(positions, axis=1)

    return avg_positions

def main(args):

    if not args.dir:
        # Load image as numpy array
        im_type = re.findall(r'\b\w+\b', args.image)[3]
        if im_type != 'c_raw':
            raise TypeError("Give a segmentation image")
        [position, time] = re.findall(r'[0-9]+', args.image)[-2:]
        im = imread("%s.tif" % args.image)
        # Get second channel of image
        fl_channel = args.image[:20] + 'y' + args.image[21:]
        im_fl = imread("%s.tif" % fl_channel)
        im_fl = np.flip(im_fl)
        # Flip image to have correct orientation
        im = np.flip(im)
        # Extract image roation
        angle = find_rotation(im)
        # Extract channel average_position
        ch_positions = find_ch_positions(im, angle)
        # Extract cells in image
        image_find_cells(im, im_fl,  args, position, time, angle, ch_positions)
    else:
        directory = args.image
        for f in sorted(os.listdir(directory)):
            if f[-4:] == ".tif" and f[11] == 'c':
                # Obtain image and flip
                im_seg = imread(directory + f)
                im_seg = np.flip(im_seg)
                # Get second channel of image
                fl_channel = f[:11] + 'y' + f[12:]
                im_fl = imread(directory + fl_channel)
                im_fl = np.flip(im_fl)
                # Obtain position and time
                [position, time] = re.findall(r'[0-9]+', f)[-2:]
                if time == '001':
                    # Extract image roation
                    angle = find_rotation(im_seg)
                    angle = 0
                    # Extract channel average_position
                    ch_positions = find_ch_positions(im_seg, angle)+4
                # Extract cells in image
                image_find_cells(im_seg, im_fl,  args, position, time, angle, ch_positions)

            else:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cells form MM fluorescence images")
    parser.add_argument("image", help="image from which to extract cells")
    parser.add_argument("output_directory", help="directory to save tiffs to")
    parser.add_argument("-d", "--dir",  help="treat image as directory", action="store_true")
    parser.add_argument("-D", "--debug", help="turn debugging on", action="store_true")
    args = parser.parse_args()

    # Set debug
    DEBUG = args.debug

    main(args)

