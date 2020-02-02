from molyso.generic.otsu import threshold_otsu
from matplotlib.colors import LinearSegmentedColormap
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import numpy as np
import efd

class contour:
    def __init__(self, im, bias, bacteria, m, pixel_size, padding, f_order=10):
        self.im = im
        self.bias = bias #otsu bias
        self.m = m
        self.pixel_size = pixel_size
        self.padding = padding
        self.f_order = f_order
        self.colors = [(0, 0, 0), (1, 0, 0)]
        self.cm = LinearSegmentedColormap.from_list('test', self.colors, N=np.amax(self.im))

        self.boundary = self._transform_vertices(bacteria.boundary[:, :-1], bacteria)
        self.pixelated_contour = self._px_contour()
        self.smoothed_contour = self.smooth()
        self.smoothed_eliptical = self.smooth_eliptical()
        self.active_contour = active_contour(self.im, self.smoothed_contour, alpha=2.0, beta=20.0, w_line=-0.4, max_px_move=0.05)

    def _transform_vertices(self, verts,  bacteria):
        verts = verts - bacteria.min[:-1] # move to get non-zero values
        verts = verts*self.m #magnification
        verts = verts / self.pixel_size #scaling by size of pixels
        verts = verts + self.padding # add padding
        verts[:,[0, 1]] = verts[:,[1, 0]] #make horizontal
        return verts

    def _px_contour(self):
        binary_image = self.im > (threshold_otsu(self.im)*self.bias)
        rows, columns = np.where(binary_image == True)
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

    def smooth(self):
        complex_boundary = np.array([x+1j*y for x, y in self.pixelated_contour])
        dft = np.fft.fft(complex_boundary)
        dft[1+self.f_order:-self.f_order] = 0
        smoothed_boundary = np.fft.ifft(dft)
        smoothed_boundary = np.stack((smoothed_boundary.real, smoothed_boundary.imag),-1)
        return smoothed_boundary

    def smooth_eliptical(self):
        locus = efd.calculate_dc_coefficients(self.pixelated_contour)
        coeffs = efd.elliptic_fourier_descriptors(self.pixelated_contour, order=self.f_order)
        contour = efd.reconstruct_contour(coeffs, locus=locus, num_points=100)
        return contour

    def re_smooth(self, f):
        #TODO make this call the original smoothing functions
        locus = efd.calculate_dc_coefficients(self.pixelated_contour)
        coeffs = efd.elliptic_fourier_descriptors(self.pixelated_contour, order=f)
        contour = efd.reconstruct_contour(coeffs, locus=locus, num_points=100)
        self.smoothed_eliptical = contour
        complex_boundary = np.array([x+1j*y for x, y in self.pixelated_contour])
        dft = np.fft.fft(complex_boundary)
        dft[1+f:-f] = 0
        smoothed_boundary = np.fft.ifft(dft)
        smoothed_boundary = np.stack((smoothed_boundary.real, smoothed_boundary.imag),-1)
        self.smoothed_contour = smoothed_boundary

    def show_pixelated_contour(self):
        fig, ax = plt.subplots()
        ax.plot(self.pixelated_contour[:, 0], self.pixelated_contour[:, 1], 'y', lw=1, label="Pixelated boundary")
        ax.plot(self.boundary[:,0], self.boundary[:, 1], 'g', lw=1, label="Ground Truth")
        plt.imshow(self.im, origin='lower', cmap=self.cm)
        plt.title("Pixelated Boundary")
        plt.legend(fontsize="x-small")
        plt.show()

    def show_smoothed_contour(self):
        fig, ax = plt.subplots()
        ax.plot(self.smoothed_contour[:, 0], self.smoothed_contour[:, 1], 'y', lw=1, label="Smoothed boundary")
        ax.plot(self.boundary[:,0], self.boundary[:, 1], 'g', lw=1, label="Ground Truth")
        plt.imshow(self.im, origin='lower', cmap=self.cm)
        plt.title("Smoothed Boundary")
        plt.legend(fontsize="x-small")
        plt.show()

    def show_active_contour(self):
        fig, ax = plt.subplots()
        ax.plot(self.active_contour[:, 0], self.active_contour[:, 1], 'y', lw=1, label="Active Contour")
        ax.plot(self.boundary[:,0], self.boundary[:, 1], 'g', lw=1, label="Ground Truth")
        plt.imshow(self.im, origin='lower', cmap=self.cm)
        plt.title("Active Contour")
        plt.legend(fontsize="x-small")
        plt.show()

    def show_contours(self):
        fig, ax = plt.subplots()
        ax.plot(self.pixelated_contour[:, 0], self.pixelated_contour[:, 1], 'y', lw=2, label="Pixelated boundary")
        ax.plot(self.smoothed_contour[:, 0], self.smoothed_contour[:, 1], 'r', lw=2, label="Smoothed boundary")
        ax.plot(self.active_contour[:, 0], self.active_contour[:, 1], 'b', lw=2, label="Active Contour")
        ax.plot(self.boundary[:,0], self.boundary[:, 1], 'g', lw=2, label="Ground Truth")
        plt.imshow(self.im, origin='lower', cmap=self.cm)
        plt.title("Contours")
        plt.legend(fontsize="x-small")
        plt.show()

class contour_real:
    def __init__(self, im, bias):
        self.im = im
        self.bias = bias #otsu bias
        self.colors = [(0, 0, 0), (1, 0, 0)]
        self.cm = LinearSegmentedColormap.from_list('test', self.colors, N=np.amax(self.im))
        self.pixelated_contour = self._px_contour()
        self.smoothed_contour = self.smooth()
        self.active_contour = active_contour(self.im, self.smoothed_contour, alpha=2.0, beta=20.0, w_line=-0.4, max_px_move=0.05)

    def _px_contour(self):
        binary_image = self.im > (threshold_otsu(self.im)*self.bias)
        rows, columns = np.where(binary_image == True)
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

    def smooth(self):
        locus = efd.calculate_dc_coefficients(self.pixelated_contour)
        coeffs = efd.elliptic_fourier_descriptors(self.pixelated_contour, order=10)
        contour = efd.reconstruct_contour(coeffs, locus=locus, num_points=100)
        return contour

    def show_pixelated_contour(self):
        fig, ax = plt.subplots()
        ax.plot(self.pixelated_contour[:, 0], self.pixelated_contour[:, 1], 'y', lw=1, label="Pixelated boundary")
        plt.imshow(self.im, origin='lower', cmap=self.cm)
        plt.title("Pixelated Boundary")
        plt.legend(fontsize="x-small")
        plt.show()

    def show_smoothed_contour(self):
        fig, ax = plt.subplots()
        ax.plot(self.smoothed_contour[:, 0], self.smoothed_contour[:, 1], 'y', lw=1, label="Smoothed boundary")
        plt.imshow(self.im, origin='lower', cmap=self.cm)
        plt.title("Smoothed Boundary")
        plt.legend(fontsize="x-small")
        plt.show()

    def show_active_contour(self):
        fig, ax = plt.subplots()
        ax.plot(self.active_contour[:, 0], self.active_contour[:, 1], 'y', lw=1, label="Active Contour")
        plt.imshow(self.im, origin='lower', cmap=self.cm)
        plt.title("Active Contour")
        plt.legend(fontsize="x-small")
        plt.show()

    def show_contours(self):
        fig, ax = plt.subplots()
        ax.plot(self.pixelated_contour[:, 0], self.pixelated_contour[:, 1], 'y', lw=2, label="Pixelated boundary")
        ax.plot(self.smoothed_contour[:, 0], self.smoothed_contour[:, 1], 'r', lw=2, label="Smoothed boundary")
        ax.plot(self.active_contour[:, 0], self.active_contour[:, 1], 'b', lw=2, label="Active Contour")
        plt.imshow(self.im, origin='lower', cmap=self.cm)
        plt.title("Contours")
        plt.legend(fontsize="x-small")
        plt.show()
