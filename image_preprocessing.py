from tifffile import imread, imsave
import numpy as np

# Make folders in directory above available
import sys
sys.path.append('../')

import argparse
import os

# Import require molyso function
from molyso.molyso.generic.otsu import threshold_otsu
from molyso.molyso.generic.signal import hamming_smooth, \
    simple_baseline_correction, find_extrema_and_prominence, vertical_mean
# Tunable for number of channels must be changed to 200
from molyso.molyso.mm.channel_detection import find_channels
from molyso.molyso.generic.rotation import find_rotation, \
    apply_rotate_and_cleanup

DEBUG = True

def debug(string):
    if DEBUG:
        print(string)

def remove_black_in_channel(ch, padding=2):
    # Remove big black spaces at the top or bottom of channel
    # Perform otsu binarization
    binary_image = ch > (threshold_otsu(ch) * 1.0)

    # Determine rows that contain only blanks from otsu image
    rows, columns = np.where(binary_image == False)
    unique, counts = np.unique(rows, return_counts=True)
    blank_rows = unique[np.where(counts == ch.shape[1])]

    # Determine areas of black

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
        return ch[init_pos:end_pos]
    else:
        return ch[init_pos - padding:end_pos + padding]

def find_cells(ch):
    # Obtain vertical smoothed intensity profile of channel
    profile = vertical_mean(ch)
    profile = simple_baseline_correction(profile)
    profile = hamming_smooth(profile, 10)

    # Find extrema in the profile by sliding window approach
    extrema = find_extrema_and_prominence(profile, 15)

    # Define cells positions
    positions = [_pos for _pos in extrema.minima if extrema.prominence[_pos] > 0]
    positions = positions + [profile.size]
    # Check certain requirements to show they are cells
    # If smaller than 10 too small and if biggger than 50 too big
    return [[_last_pos, _pos] for _last_pos, _pos in zip(
        [0] + positions, positions) if _pos - _last_pos > 10 and _pos - _last_pos < 50]

def image_find_cells(im, args, position):

    debug("File: %s" % args.image)
    debug("Position: %s" % position)

    # Apply rotation to both images
    im_rot = apply_rotate_and_cleanup(
                im, find_rotation(im))[0]

    # Find positions of channels and top and bottom of channel
    positions, (upper, lower) = find_channels(im_rot)

    # Crop off top of channel and show
    im_rot_cr = im_rot[upper:lower, :]

    # Find center of channels. The positions given by MOLYSO are not
    # accurate enough for fluorescent images.
    avg_positions = np.mean(positions, axis=1)

    # Define image width / channel width
    channel_width = 26

    # Number of channels
    n_channels = 0

    for avg_position in avg_positions:
        n_channels += 1

        # Avoid indexing outside of bounds
        if avg_position < channel_width / 2:
            avg_position = channel_width / 2
        elif avg_position > im_rot_cr.shape[0] - channel_width / 2:
            avg_position = im_rot_cr.shape[0] - channel_width / 2

        # Get channel in the image
        channel = im_rot_cr[:, int(avg_position - channel_width / 2):
                               int(avg_position + channel_width / 2)]

        # Remove black ends of channel
        channel = remove_black_in_channel(channel)

        # Find cells in channel
        cells = find_cells(channel)

        # Final padding added to make standard size
        default_height = 70

        n_cells = 0

        # Show cells with final padding
        for start, end in cells:
            pad = (default_height - end + start) / 2
            if (pad).is_integer():
                cell = np.pad(channel[start:end, :], ((int(pad), int(
                    pad)), (0, 0)), mode='constant', constant_values=0)
            else:
                cell = np.pad(channel[start:end, :], ((
                    int(pad - 0.5), int(pad + 0.5)), (0, 0)),
                    mode='constant', constant_values=0)
            n_cells += 1
            imsave('%s/%s_%s_channel_%s_cell_%s.tif' % (args.output_directory, args.image, position, n_channels, n_cells), cell)
        debug('Number of cells in channel %s: %s' % (n_channels, n_cells))

    debug('Number of channels found: %s' % (n_channels))


def extract_image(im, args):
    # Split image into bottom and top channel
    [im_top, im_bottom] = np.vsplit(im, 2)

    # Flip top image
    im_top_flip = np.flip(im_top)

    # Find cells in both
    image_find_cells(im_bottom, args, "bottom")
    image_find_cells(im_top_flip, args, "top")


def main(args):

    if not args.dir:
        # Load image as numpy array
        im = imread("%s.tif" % args.image)
        # Extract cells in image
        extract_image(im, args)
    else:
        directory = args.image
        for f in os.listdir(directory):
            if f[-4:] == ".tif":
                im = imread(directory + f)
                args.image = f[:-4]
                # Extract cells in image
                extract_image(im, args)
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

