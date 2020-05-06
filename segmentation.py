import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import efd

def boundary_from_pixelated_mask(pixelated_mask):
    rows, columns = np.where(pixelated_mask == True)
    prev_r = 0
    coordinates_l = []
    coordinates_r = []
    for i, r in enumerate(rows):
        if prev_r != r:
            coordinates_l.append([columns[i]-0.5, r-0.5])
            coordinates_l.append([columns[i]-0.5, r+0.5])
            coordinates_r.append([columns[i-1]+0.5, prev_r-0.5])
            coordinates_r.append([columns[i-1]+0.5, prev_r+0.5])
            prev_r = r
    del coordinates_r[0:2] # pop initial useless value
    coordinates_r.append([columns[-1]+0.5, rows[-1]-0.5]) # add last value missing
    coordinates_r.append([columns[-1]+0.5, rows[-1]+0.5]) # add last value missing
    last_columns = np.where(rows == rows[-1])
    for i in np.arange(0.5, columns[-1]-columns[last_columns[0][0]], 1):
        coordinates_r.append([columns[-1]-i, rows[-1]+0.5])
    coordinates_r.reverse() # revers to add to l
    first_columns = np.where(rows == rows[0])
    for i in np.arange(0.5, columns[first_columns[0][-1]] - columns[0] , 1):
        coordinates_r.append([columns[first_columns[0][-1]]-i, rows[0]-0.5])
    coordinates = coordinates_l + coordinates_r
    coordinates.append(coordinates[0]) # duplicate start point
    return np.array(coordinates)

def smooth_boundary(boundary, descriptors):
    locus = efd.calculate_dc_coefficients(boundary)
    coeffs = efd.elliptic_fourier_descriptors(boundary, order=descriptors)
    contour = efd.reconstruct_contour(coeffs, locus=locus, num_points=100)
    return contour

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def segment_cell(cell, model, pad_flag = True):
    # Normalize
    cell = (cell-np.min(cell))/(np.max(cell)-np.min(cell))

    if pad_flag:
        # Pad cell
        height = 50
        pad = (height - cell.shape[0]) / 2
        if pad.is_integer():
            cell = np.pad(cell, ((int(pad), int(
                pad)), (0, 0)), mode='constant', constant_values=0)
        else:
            cell = np.pad(cell, ((
                int(pad - 0.5), int(pad + 0.5)), (0, 0)),
                mode='constant', constant_values=0)

    # Add extra channel
    cell = cell[...,tf.newaxis]

    # Predict
    prediction = model.predict(cell[tf.newaxis, ...])
    pixelated_mask = create_mask(prediction)[:,:,0].numpy()

    # Remove padding 
    if pad_flag:
        if pad.is_integer():
            pixelated_mask = pixelated_mask[int(pad):-int(pad),:]
        else:
            pixelated_mask = pixelated_mask[int(pad-0.5):-int(pad+0.5), :]

    return pixelated_mask

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.show()

def display_boundary(cell, boundaries):
  plt.figure(figsize=(15, 15))

  title = ['Pixelated Boundary', 'Smoothed Boundary']
  colors = ['r', 'g']

  plt.imshow(cell)
  for i in range(len(boundaries)):
    plt.plot(boundaries[i][:,0], boundaries[i][:,1], colors[i], label=title[i])
  plt.legend()
  plt.show()
