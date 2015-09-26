from __future__ import division, print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll

from .somoclu_wrap import train as wrap_train


class Somoclu(object):

    def __init__(self, n_columns, n_rows, data=None, initialcodebook=None,
                 kerneltype=0, maptype="planar", gridtype="square",
                 compactsupport=False):
        self._kernel_type = kerneltype
        self._map_type = maptype
        self._grid_type = gridtype
        self._compact_support = compactsupport
        self._check_parameters()
        self.nVectors = 0
        self.nDimensions = 0
        self.data = None
        self.bmus = None
        if data is not None:
            self.update_data(data)
        self._n_columns, self._n_rows = n_columns, n_rows
        self.umatrix = np.zeros(n_columns * n_rows, dtype=np.float32)
        self.codebook = initialcodebook

    def load_bmus(self, filename):
        self.bmus = np.loadtxt(filename, comments='%')
        if self.nVectors != 0 and len(self.bmus) != self.nVectors:
            raise Exception("The number of best matching units does not match"
                            "the number of data instances")
        else:
            self.nVectors = len(self.bmus)
        if max(self.bmus[:, 1]) > self._n_columns - 1 or \
                max(self.bmus[:, 2]) > self._n_rows - 1:
            raise Exception("The dimensions of the best matching units do not"
                            "match that of the map")

    def load_umatrix(self, filename):
        self.umatrix = np.loadtxt(filename, comments='%')
        if self.umatrix.shape != (self._n_columns, self._n_rows):
            raise Exception("The dimensions of the U-matrix do not "
                            "match that of the map")

    def load_codebook(self, filename):
        self.codebook = np.loadtxt(filename, comments='%')
        if self.nDimensions == 0:
            self.nDimensions = self.codebook.shape[1]
        if self.codebook.shape != (self._n_rows*self._n_columns,
                                   self.nDimensions):
            raise Exception("The dimensions of the codebook do not "
                            "match that of the map")
        self.codebook.shape = (self._n_rows, self._n_columns, self.nDimensions)

    def train(self, nEpoch=10, radius0=0, radiusN=1, radiusCooling="linear",
              scale0=0.1, scaleN=0.01, scaleCooling="linear"):
        self._check_cooling_parameters(radiusCooling, scaleCooling)
        if self.data is None:
            raise Exception("No data was provided!")
        self._init_codebook()
        self.umatrix.shape = (self._n_rows*self._n_columns, )
        wrap_train(np.ravel(self.data), nEpoch, self._n_columns, self._n_rows,
                   self.nDimensions, self.nVectors, radius0, radiusN,
                   radiusCooling, scale0, scaleN, scaleCooling,
                   self._kernel_type, self._map_type, self._grid_type,
                   self._compact_support, self.codebook, self.bmus,
                   self.umatrix)
        self.umatrix.shape = (self._n_rows, self._n_columns)
        self.bmus.shape = (self.nVectors, 2)
        self.codebook.shape = (self._n_rows, self._n_columns, self.nDimensions)

    def update_data(self, data):
        oldNDimensions = self.nDimensions
        if data.dtype != np.float32:
            print("Warning: data was not float32. A 32-bit copy was made")
            self.data = np.float32(data)
        else:
            self.data = data
        self.nVectors, self.nDimensions = data.shape
        if self.nDimensions != oldNDimensions and oldNDimensions != 0:
            raise Exception("The dimension of the new data does not match!")
        self.bmus = np.zeros(self.nVectors*2, dtype=np.intc)

    def view_component_planes(self, dimensions=None, figsize=None,
                              colormap=cm.Spectral_r, colorbar=False,
                              bestmatches=False, bestmatchcolors=None,
                              labels=None, filename=None):
        if figsize is None:
            figsize = (5*float(self._n_columns/self._n_rows), 5)
        if dimensions is None:
            dimensions = range(self.nDimensions)
        for i in dimensions:
            self._view_matrix(self.codebook[:, :, i], figsize, colormap,
                              colorbar, bestmatches, bestmatchcolors, labels,
                              filename)

    def view_umatrix(self, figsize=None, colormap=cm.Spectral_r,
                     colorbar=False, bestmatches=False, bestmatchcolors=None,
                     labels=None, filename=None):
        if figsize is None:
            figsize = (6*float(self._n_columns/self._n_rows), 6)
        self._view_matrix(self.umatrix, figsize, colormap, colorbar,
                          bestmatches, bestmatchcolors, labels, filename)

    def _view_matrix(self, matrix, figsize, colormap, colorbar, bestmatches,
                     bestmatchcolors, labels, filename):
        fig = plt.figure(figsize=figsize)
        if self._grid_type == "hexagonal":
            offsets = self._hexplot(matrix, fig, colormap)
            bmu_coords = np.zeros(self.bmus.shape)
            for i, (x, y) in enumerate(self.bmus):
                bmu_coords[i] = offsets[y*self._n_columns + x]
        else:
            plt.imshow(matrix, aspect='auto')
            plt.set_cmap(colormap)
            bmu_coords = self.bmus

        if colorbar:
            m = cm.ScalarMappable(cmap=colormap)
            m.set_array(matrix)
            plt.colorbar(m, orientation='horizontal', shrink=0.5)

        if bestmatches:
            if bestmatchcolors is None:
                colors = "white"
            else:
                colors = bestmatchcolors
            plt.scatter(bmu_coords[:, 0], bmu_coords[:, 1], c=colors)

        if labels is not None:
            for label, x, y in zip(labels, bmu_coords[:, 0], bmu_coords[:, 1]):
                if label is not None:
                    plt.annotate(label, xy=(x, y), xytext=(10, -5),
                                 textcoords='offset points', ha='left',
                                 va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.3',
                                           fc='white', alpha=0.8))
        plt.axis('off')
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
        return plt

    def _hexplot(self, matrix, fig, colormap):
        umatrix_min = matrix.min()
        umatrix_max = matrix.max()
        cmap = plt.get_cmap(colormap)
        s = 1.1
        offsets = np.zeros((self._n_columns*self._n_rows, 2))
        facecolors = []
        for y in range(self._n_rows):
            for x in range(self._n_columns):
                if y % 2 == 0:
                    offsets[y*self._n_columns + x] = [x+0.5, 2*y]
                    facecolors.append(cmap((matrix[y, x]-umatrix_min) /
                                           (umatrix_max)*255))
                else:
                    offsets[y*self._n_columns + x] = [x, 2*y]
                    facecolors.append(cmap((matrix[y, x]-umatrix_min) /
                                           (umatrix_max)*255))
        polygon = np.zeros((6, 2), float)
        polygon[:, 0] = s * np.array([0.5, 0.5, 0.0, -0.5, -0.5, 0.0])
        polygon[:, 1] = s * np.array([-np.sqrt(3)/6, np.sqrt(3)/6,
                                      np.sqrt(3)/2+np.sqrt(3)/6,
                                      np.sqrt(3)/6, -np.sqrt(3)/6,
                                      -np.sqrt(3)/2-np.sqrt(3)/6])
        polygons = np.expand_dims(polygon, 0) + np.expand_dims(offsets, 1)
        ax = fig.gca()
        collection = mcoll.PolyCollection(
            polygons,
            offsets=offsets,
            facecolors=facecolors,
            edgecolors=facecolors,
            linewidths=1.0,
            offset_position="data")
        ax.add_collection(collection, autolim=False)
        corners = ((-0.5, -0.5), (self._n_columns + 0.5, 2*self._n_rows + 0.5))
        ax.update_datalim(corners)
        ax.autoscale_view(tight=True)
        return offsets

    def _check_cooling_parameters(self, radiusCooling, scaleCooling):
        if radiusCooling != "linear" and radiusCooling != "exponential":
            raise Exception("Invalid parameter for radiusCooling: " +
                            radiusCooling)
        if scaleCooling != "linear" and scaleCooling != "exponential":
            raise Exception("Invalid parameter for scaleCooling: " +
                            scaleCooling)

    def _check_parameters(self):
        if self._map_type != "planar" and self._map_type != "toroid":
            raise Exception("Invalid parameter for _map_type: " +
                            self._map_type)
        if self._grid_type != "square" and self._grid_type != "hexagonal":
            raise Exception("Invalid parameter for _grid_type: " +
                            self._grid_type)
        if self._kernel_type != 0 and self._kernel_type != 1:
            raise Exception("Invalid parameter for kernelTye: " +
                            self._kernel_type)

    def _init_codebook(self):
        codebook_size = self._n_columns * self._n_rows * self.nDimensions
        if self.codebook is None:
            self.codebook = np.zeros(codebook_size, dtype=np.float32)
            self.codebook[0:2] = [1000, 2000]
        elif self.codebook.size != codebook_size:
            raise Exception("Invalid size for initial codebook")
        else:
            if self.codebook.dtype != np.float32:
                print("Warning: initialCodebook was not float32. A 32-bit "
                      "copy was made")
                self.codebook = np.float32(self.codebook)
        self.codebook.shape = (codebook_size, )
