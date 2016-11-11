# -*- coding: utf-8 -*-
"""
The module contains the Somoclu class that trains and visualizes
self-organizing maps and emergent self-organizing maps.

Created on Sun July 26 15:07:47 2015

@author: Peter Wittek
"""
from __future__ import division, print_function
import sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
from scipy.spatial.distance import cdist
try:
    import seaborn as sns
    from sklearn.metrics.pairwise import pairwise_distances
    have_heatmap = True
except ImportError:
    have_heatmap = False

try:
    from .somoclu_wrap import train as wrap_train
except ImportError:
    print("Warning: training function cannot be imported. Only pre-trained "
          "maps can be analyzed.")
    if sys.platform.startswith('win'):
        print("If you installed Somoclu with pip on Windows, this typically "
              "means missing DLLs. Please refer to the documentation and to "
              "this issue:")
        print("https://github.com/peterwittek/somoclu/issues/28")
    elif sys.platform.startswith('darwin'):
        print("If you installed Somoclu with pip on OS X, this typically "
              "means missing libiomp. Please refer to the documentation and to"
              " this issue:")
        print("https://github.com/peterwittek/somoclu/issues/28")


class Somoclu(object):
    """Class for training and visualizing a self-organizing map.

    :param n_columns: The number of columns in the map.
    :type n_columns: int.
    :param n_rows: The number of rows in the map.
    :type n_rows: int.
    :param data: Optional parameter to provide training data. It is not
                 necessary if the map is otherwise trained outside Python,
                 e.g., on a GPU cluster.
    :type data: 2D numpy.array of float32.
    :param initialcodebook: Optional parameter to start the training with a
                            given codebook.
    :type initialcodebook: 2D numpy.array of float32.
    :param kerneltype: Optional parameter to specify which kernel to use:

                           * 0: dense CPU kernel (default)
                           * 1: dense GPU kernel (if compiled with it)
    :type kerneltype: int.
    :param maptype: Optional parameter to specify the map topology:

                           * "planar": Planar map (default)
                           * "toroid": Toroid map
    :type maptype: str.
    :param gridtype: Optional parameter to specify the grid form of the nodes:

                           * "rectangular": rectangular neurons (default)
                           * "hexagonal": hexagonal neurons
    :type gridtype: str.
    :param compactsupport: Optional parameter to cut off map updates beyond the
                           training radius with the Gaussian neighborhood.
                           Default: True.
    :type compactsupport: bool.
    :param neighborhood: Optional parameter to specify the neighborhood:

                           * "gaussian": Gaussian neighborhood (default)
                           * "bubble": bubble neighborhood function
    :type neighborhood: str.
    :param std_coeff: Optional parameter to set the coefficient in the Gaussian
                      neighborhood function exp(-||x-y||^2/(2*(coeff*radius)^2))
                      Default: 0.5
    :type std_coeff: float.
    :param initialization: Optional parameter to specify the initalization:

                           * "random": random weights in the codebook
                           * "pca": codebook is initialized from the first
                             subspace spanned by the first two eigenvectors of
                             the correlation matrix
    :type initialization: str.
    """

    def __init__(self, n_columns, n_rows, data=None, initialcodebook=None,
                 kerneltype=0, maptype="planar", gridtype="rectangular",
                 compactsupport=True, neighborhood="gaussian", std_coeff=0.5,
                 initialization=None):
        """Constructor for the class.
        """
        self._n_columns, self._n_rows = n_columns, n_rows
        self._kernel_type = kerneltype
        self._map_type = maptype
        self._grid_type = gridtype
        self._compact_support = compactsupport
        self._neighborhood = neighborhood
        self._std_coeff = std_coeff
        self._check_parameters()
        self.activation_map = None
        if initialcodebook is not None and initialization is not None:
            raise Exception("An initial codebook is given but initilization"
                            " is also requested")
        self.bmus = None
        self.umatrix = np.zeros(n_columns * n_rows, dtype=np.float32)
        self.codebook = initialcodebook
        if initialization is None or initialization == "random":
            self._initialization = "random"
        elif initialization == "pca":
            self._initialization = "pca"
        else:
            raise Exception("Unknown initialization method")
        self.n_vectors = 0
        self.n_dim = 0
        self.clusters = None
        self._data = None
        if data is not None:
            self.update_data(data)

    def load_bmus(self, filename):
        """Load the best matching units from a file to the Somoclu object.

        :param filename: The name of the file.
        :type filename: str.
        """
        self.bmus = np.loadtxt(filename, comments='%', usecols=(1, 2))
        if self.n_vectors != 0 and len(self.bmus) != self.n_vectors:
            raise Exception("The number of best matching units does not match "
                            "the number of data instances")
        else:
            self.n_vectors = len(self.bmus)
        tmp = self.bmus[:, 0].copy()
        self.bmus[:, 0] = self.bmus[:, 1].copy()
        self.bmus[:, 1] = tmp
        if max(self.bmus[:, 0]) > self._n_columns - 1 or \
                max(self.bmus[:, 1]) > self._n_rows - 1:
            raise Exception("The dimensions of the best matching units do not "
                            "match that of the map")

    def load_umatrix(self, filename):
        """Load the umatrix from a file to the Somoclu object.

        :param filename: The name of the file.
        :type filename: str.
        """

        self.umatrix = np.loadtxt(filename, comments='%')
        if self.umatrix.shape != (self._n_rows, self._n_columns):
            raise Exception("The dimensions of the U-matrix do not "
                            "match that of the map")

    def load_codebook(self, filename):
        """Load the codebook from a file to the Somoclu object.

        :param filename: The name of the file.
        :type filename: str.
        """
        self.codebook = np.loadtxt(filename, comments='%')
        if self.n_dim == 0:
            self.n_dim = self.codebook.shape[1]
        if self.codebook.shape != (self._n_rows*self._n_columns,
                                   self.n_dim):
            raise Exception("The dimensions of the codebook do not "
                            "match that of the map")
        self.codebook.shape = (self._n_rows, self._n_columns, self.n_dim)

    def train(self, epochs=10, radius0=0, radiusN=1, radiuscooling="linear",
              scale0=0.1, scaleN=0.01, scalecooling="linear"):
        """Train the map on the current data in the Somoclu object.

        :param epochs: The number of epochs to train the map for.
        :type epochs: int.
        :param radius0: The initial radius on the map where the update happens
                        around a best matching unit. Default value of 0 will
                        trigger a value of min(n_columns, n_rows)/2.
        :type radius0: int.
        :param radiusN: The radius on the map where the update happens around a
                        best matching unit in the final epoch. Default: 1.
        :type radiusN: int.
        :param radiuscooling: The cooling strategy between radius0 and radiusN:

                                   * "linear": Linear interpolation (default)
                                   * "exponential": Exponential decay
        :param scale0: The initial learning scale. Default value: 0.1.
        :type scale0: float.
        :param scaleN: The learning scale in the final epoch. Default: 0.01.
        :type scaleN: float.
        :param scalecooling: The cooling strategy between scale0 and scaleN:

                                   * "linear": Linear interpolation (default)
                                   * "exponential": Exponential decay
        :type scalecooling: str.
        """
        _check_cooling_parameters(radiuscooling, scalecooling)
        if self._data is None:
            raise Exception("No data was provided!")
        self._init_codebook()
        self.umatrix.shape = (self._n_rows*self._n_columns, )
        self.bmus.shape = (self.n_vectors*2, )
        wrap_train(np.ravel(self._data), epochs, self._n_columns, self._n_rows,
                   self.n_dim, self.n_vectors, radius0, radiusN,
                   radiuscooling, scale0, scaleN, scalecooling,
                   self._kernel_type, self._map_type, self._grid_type,
                   self._compact_support, self._neighborhood == "gaussian",
                   self._std_coeff, self.codebook, self.bmus, self.umatrix)
        self.umatrix.shape = (self._n_rows, self._n_columns)
        self.bmus.shape = (self.n_vectors, 2)
        self.codebook.shape = (self._n_rows, self._n_columns, self.n_dim)

    def update_data(self, data):
        """Change the data set in the Somoclu object. It is useful when the
        data is updated and the training should continue on the new data.

        :param data: The training data.
        :type data: 2D numpy.array of float32.
        """
        oldn_dim = self.n_dim
        if data.dtype != np.float32:
            print("Warning: data was not float32. A 32-bit copy was made")
            self._data = np.float32(data)
        else:
            self._data = data
        self.n_vectors, self.n_dim = data.shape
        if self.n_dim != oldn_dim and oldn_dim != 0:
            raise Exception("The dimension of the new data does not match!")
        self.bmus = np.zeros(self.n_vectors*2, dtype=np.intc)

    def view_component_planes(self, dimensions=None, figsize=None,
                              colormap=cm.Spectral_r, colorbar=False,
                              bestmatches=False, bestmatchcolors=None,
                              labels=None, zoom=None, filename=None):
        """Observe the component planes in the codebook of the SOM.

        :param dimensions: Optional parameter to specify along which dimension
                           or dimensions should the plotting happen. By
                           default, each dimension is plotted in a sequence of
                           plots.
        :type dimension: int or list of int.
        :param figsize: Optional parameter to specify the size of the figure.
        :type figsize: (int, int)
        :param colormap: Optional parameter to specify the color map to be
                         used.
        :type colormap: matplotlib.colors.Colormap
        :param colorbar: Optional parameter to include a colormap as legend.
        :type colorbar: bool.
        :param bestmatches: Optional parameter to plot best matching units.
        :type bestmatches: bool.
        :param bestmatchcolors: Optional parameter to specify the color of each
                                best matching unit.
        :type bestmatchcolors: list of int.
        :param labels: Optional parameter to specify the label of each point.
        :type labels: list of str.
        :param zoom: Optional parameter to zoom into a region on the map. The
                     first two coordinates of the tuple are the row limits, the
                     second tuple contains the column limits.
        :type zoom: ((int, int), (int, int))
        :param filename: If specified, the plot will not be shown but saved to
                         this file.
        :type filename: str.
        """
        if self.codebook is None:
            raise Exception("The codebook is not available. Either train a map"
                            " or load a codebook from a file")
        if dimensions is None:
            dimensions = range(self.n_dim)
        for i in dimensions:
            plt = self._view_matrix(self.codebook[:, :, i], figsize, colormap,
                                    colorbar, bestmatches, bestmatchcolors,
                                    labels, zoom, filename)
        return plt

    def view_umatrix(self, figsize=None, colormap=cm.Spectral_r,
                     colorbar=False, bestmatches=False, bestmatchcolors=None,
                     labels=None, zoom=None, filename=None):
        """Plot the U-matrix of the trained map.

        :param figsize: Optional parameter to specify the size of the figure.
        :type figsize: (int, int)
        :param colormap: Optional parameter to specify the color map to be
                         used.
        :type colormap: matplotlib.colors.Colormap
        :param colorbar: Optional parameter to include a colormap as legend.
        :type colorbar: bool.
        :param bestmatches: Optional parameter to plot best matching units.
        :type bestmatches: bool.
        :param bestmatchcolors: Optional parameter to specify the color of each
                                best matching unit.
        :type bestmatchcolors: list of int.
        :param labels: Optional parameter to specify the label of each point.
        :type labels: list of str.
        :param zoom: Optional parameter to zoom into a region on the map. The
                     first two coordinates of the tuple are the row limits, the
                     second tuple contains the column limits.
        :type zoom: ((int, int), (int, int))
        :param filename: If specified, the plot will not be shown but saved to
                         this file.
        :type filename: str.
        """
        if self.umatrix is None:
            raise Exception("The U-matrix is not available. Either train a map"
                            " or load a U-matrix from a file")
        return self._view_matrix(self.umatrix, figsize, colormap, colorbar,
                                 bestmatches, bestmatchcolors, labels, zoom,
                                 filename)

    def view_activation_map(self, data_vector=None, data_index=None,
                            activation_map=None, figsize=None,
                            colormap=cm.Spectral_r, colorbar=False,
                            bestmatches=False, bestmatchcolors=None,
                            labels=None, zoom=None, filename=None):
        """Plot the activation map of a given data instance or a new data
        vector

        :param data_vector: Optional parameter for a new vector
        :type data_vector: numpy.array
        :param data_index: Optional parameter for the index of the data instance
        :type data_index: int.
        :param activation_map: Optional parameter to pass the an activation map
        :type activation_map: numpy.array
        :param figsize: Optional parameter to specify the size of the figure.
        :type figsize: (int, int)
        :param colormap: Optional parameter to specify the color map to be
                         used.
        :type colormap: matplotlib.colors.Colormap
        :param colorbar: Optional parameter to include a colormap as legend.
        :type colorbar: bool.
        :param bestmatches: Optional parameter to plot best matching units.
        :type bestmatches: bool.
        :param bestmatchcolors: Optional parameter to specify the color of each
                                best matching unit.
        :type bestmatchcolors: list of int.
        :param labels: Optional parameter to specify the label of each point.
        :type labels: list of str.
        :param zoom: Optional parameter to zoom into a region on the map. The
                     first two coordinates of the tuple are the row limits, the
                     second tuple contains the column limits.
        :type zoom: ((int, int), (int, int))
        :param filename: If specified, the plot will not be shown but saved to
                         this file.
        :type filename: str.
        """
        if data_vector is None and data_index is None:
            raise Exception("Either specify a vector to see its activation "
                            "or give an index of the training data instances")
        if data_vector is not None and data_index is not None:
            raise Exception("You cannot specify both a data vector and the "
                            "index of a training data instance")
        if data_vector is not None and activation_map is not None:
            raise Exception("You cannot pass a previously computated"
                            "activation map with a data vector")
        if data_vector is not None:
            try:
                d1, _ = data_vector.shape
                w = data_vector.copy()
            except ValueError:
                d1, _ = data_vector.shape
                w = data_vector.reshape(1, d1)
            if w.shape[1] == 1:
                w = w.T
            matrix = cdist(self.codebook.reshape((self.codebook.shape[0] *
                                                  self.codebook.shape[1],
                                                  self.codebook.shape[2])),
                           w, 'euclidean').T
            matrix.shape = (self.codebook.shape[0], self.codebook.shape[1])
        else:
            if activation_map is None and self.activation_map is None:
                self.get_surface_state()
            if activation_map is None:
                activation_map = self.activation_map
            matrix = activation_map[data_index].reshape((self.codebook.shape[0],
                                                         self.codebook.shape[1]))
        return self._view_matrix(matrix, figsize, colormap, colorbar,
                                 bestmatches, bestmatchcolors, labels, zoom,
                                 filename)

    def _view_matrix(self, matrix, figsize, colormap, colorbar, bestmatches,
                     bestmatchcolors, labels, zoom, filename):
        """Internal function to plot a map with best matching units and labels.
        """
        if zoom is None:
            zoom = ((0, self._n_rows), (0, self._n_columns))
        if figsize is None:
            figsize = (8, 8/float(zoom[1][1]/zoom[0][1]))
        fig = plt.figure(figsize=figsize)
        if self._grid_type == "hexagonal":
            offsets = _hexplot(matrix[zoom[0][0]:zoom[0][1],
                                      zoom[1][0]:zoom[1][1]], fig, colormap)
            filtered_bmus = self._filter_array(self.bmus, zoom)
            filtered_bmus[:, 0] = filtered_bmus[:, 0] - zoom[1][0]
            filtered_bmus[:, 1] = filtered_bmus[:, 1] - zoom[0][0]
            bmu_coords = np.zeros(filtered_bmus.shape)
            for i, (row, col) in enumerate(filtered_bmus):
                bmu_coords[i] = offsets[col*zoom[1][1] + row]
        else:
            plt.imshow(matrix[zoom[0][0]:zoom[0][1], zoom[1][0]:zoom[1][1]],
                       aspect='auto')
            plt.set_cmap(colormap)
            bmu_coords = self._filter_array(self.bmus, zoom)
            bmu_coords[:, 0] = bmu_coords[:, 0] - zoom[1][0]
            bmu_coords[:, 1] = bmu_coords[:, 1] - zoom[0][0]
        if colorbar:
            cmap = cm.ScalarMappable(cmap=colormap)
            cmap.set_array(matrix)
            plt.colorbar(cmap, orientation='horizontal', shrink=0.5)

        if bestmatches:
            if bestmatchcolors is None:
                if self.clusters is None:
                    colors = "white"
                else:
                    colors = []
                    for bm in self.bmus:
                        colors.append(self.clusters[bm[1], bm[0]])
                    colors = self._filter_array(colors, zoom)
            else:
                colors = self._filter_array(bestmatchcolors, zoom)
            plt.scatter(bmu_coords[:, 0], bmu_coords[:, 1], c=colors)

        if labels is not None:
            for label, col, row in zip(self._filter_array(labels, zoom),
                                       bmu_coords[:, 0], bmu_coords[:, 1]):
                if label is not None:
                    plt.annotate(label, xy=(col, row), xytext=(10, -5),
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

    def _filter_array(self, a, zoom):
        filtered_array = []
        for index, bmu in enumerate(self.bmus):
            if bmu[0] >= zoom[1][0] and bmu[0] < zoom[1][1] and \
                    bmu[1] >= zoom[0][0] and bmu[1] < zoom[0][1]:
                filtered_array.append(a[index])
        return np.array(filtered_array)

    def _check_parameters(self):
        """Internal function to verify the basic parameters of the SOM.
        """
        if self._map_type != "planar" and self._map_type != "toroid":
            raise Exception("Invalid parameter for _map_type: " +
                            self._map_type)
        if self._grid_type != "rectangular" and self._grid_type != "hexagonal":
            raise Exception("Invalid parameter for _grid_type: " +
                            self._grid_type)
        if self._neighborhood != "gaussian" and self._neighborhood != "bubble":
            raise Exception("Invalid parameter for neighborhood: " +
                            self._neighborhood)
        if self._kernel_type != 0 and self._kernel_type != 1:
            raise Exception("Invalid parameter for kernelTye: " +
                            self._kernel_type)

    def _pca_init(self):
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, svd_solver="randomized")
        except:
            from sklearn.decomposition import RandomizedPCA
            pca = RandomizedPCA(n_components=2)
        coord = np.zeros((self._n_columns*self._n_rows, 2))
        for i in range(self._n_columns*self._n_rows):
            coord[i, 0] = int(i / self._n_columns)
            coord[i, 1] = int(i % self._n_columns)
        coord = coord/[self._n_rows-1, self._n_columns-1]
        coord = (coord - .5)*2
        me = np.mean(self._data, 0)
        self.codebook = np.tile(me, (self._n_columns*self._n_rows, 1))
        pca.fit(self._data - me)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.linalg.norm(eigvec, axis=1)
        eigvec = ((eigvec.T/norms)*eigval).T
        for j in range(self._n_columns*self._n_rows):
            for i in range(eigvec.shape[0]):
                self.codebook[j, :] = self.codebook[j, :] + \
                                      coord[j, i] * eigvec[i, :]

    def _init_codebook(self):
        """Internal function to set the codebook or to indicate it to the C++
        code that it should be randomly initialized.
        """
        codebook_size = self._n_columns * self._n_rows * self.n_dim
        if self.codebook is None:
            if self._initialization == "random":
                self.codebook = np.zeros(codebook_size, dtype=np.float32)
                self.codebook[0:2] = [1000, 2000]
            else:
                self._pca_init()
        elif self.codebook.size != codebook_size:
            raise Exception("Invalid size for initial codebook")
        else:
            if self.codebook.dtype != np.float32:
                print("Warning: initialcodebook was not float32. A 32-bit "
                      "copy was made")
                self.codebook = np.float32(self.codebook)
        self.codebook.shape = (codebook_size, )

    def cluster(self, algorithm=None):
        """Cluster the codebook. The clusters of the data instances can be
        assigned based on the BMUs. The method populates the class variable
        Somoclu.clusters. If viewing methods are called after clustering, but
        without colors for best matching units, colors will be automatically
        assigned based on cluster membership.

        :param algorithm: Optional parameter to specify a scikit-learn
                          clustering algorithm. The default is K-means with
                          eight clusters.
        :type filename: sklearn.base.ClusterMixin.
        """

        import sklearn.base
        if algorithm is None:
            import sklearn.cluster
            algorithm = sklearn.cluster.KMeans()
        elif not isinstance(algorithm, sklearn.base.ClusterMixin):
            raise Exception("Cannot use algorithm of type " + type(algorithm))
        original_shape = self.codebook.shape
        self.codebook.shape = (self._n_columns*self._n_rows, self.n_dim)
        linear_clusters = algorithm.fit_predict(self.codebook)
        self.codebook.shape = original_shape
        self.clusters = np.zeros((self._n_rows, self._n_columns), dtype=int)
        for i, c in enumerate(linear_clusters):
            self.clusters[i // self._n_columns, i % self._n_columns] = c

    def get_surface_state(self, data=None):
        """Return the Euclidean distance between codebook and data.

        :param data: Optional parameter to specify data, otherwise the
                     data used previously to train the SOM is used.
        :type data: 2D numpy.array of float32.

        :returns: The the dot product of the codebook and the data.
        :rtype: 2D numpy.array
        """

        if data is None:
            d = self._data
        else:
            d = data
        am = cdist(self.codebook.reshape((self.codebook.shape[0] *
                                          self.codebook.shape[1],
                                          self.codebook.shape[2])),
                   d, 'euclidean').T
        if data is None:
            self.activation_map = am
        return am

    def get_bmus(self, activation_map):
        """Returns Best Matching Units indexes of the activation map.

        :param activation_map: Activation map computed with self.get_surface_state()
        :type activation_map: 2D numpy.array

        :returns: The bmus indexes corresponding to this activation map
                  (same as self.bmus for the training samples).
        :rtype: 2D numpy.array
        """

        Y, X = np.unravel_index(activation_map.argmin(axis=1),
                                (self._n_rows, self._n_columns))
        return np.vstack((X,Y)).T

    def view_similarity_matrix(self, data=None, labels=None, figsize=None,
                               filename=None):
        """Plot the similarity map according to the activation map

        :param data: Optional parameter for data points to calculate the
                     similarity with
        :type data: numpy.array
        :param figsize: Optional parameter to specify the size of the figure.
        :type figsize: (int, int)
        :param labels: Optional parameter to specify the label of each point.
        :type labels: list of str.
        :param filename: If specified, the plot will not be shown but saved to
                         this file.
        :type filename: str.
        """

        if not have_heatmap:
            raise Exception("Import dependencies missing for viewing "
                            "similarity matrix. You must have seaborn and "
                            "scikit-learn")
        if data is None and self.activation_map is None:
            self.get_surface_state()
        if data is None:
            X = self.activation_map
        else:
            X = data
        # Calculate the pairwise correlations as a metric for similarity
        corrmat = 1-pairwise_distances(X, metric="correlation")

        # Set up the matplotlib figure
        if figsize is None:
            figsize = (12, 9)
        f, ax = plt.subplots(figsize=figsize)

        # Y axis has inverted labels (seaborn default, no idea why)
        yticklabels = np.atleast_2d(labels)
        yticklabels = np.fliplr(yticklabels)[0]

        # Draw the heatmap using seaborn
        sns.heatmap(corrmat, vmax=1, vmin=-1, square=True, xticklabels=labels,
                    yticklabels=labels, cmap="RdBu_r", center=0)
        f.tight_layout()

        # This sets the ticks to a readable angle
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        # This sets the labels for the two axes
        ax.set_yticklabels(yticklabels, ha='right', va='center', size=8)
        ax.set_xticklabels(labels, ha='center', va='top', size=8)

        # Save and close the figure
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()
        return plt


def _check_cooling_parameters(radiuscooling, scalecooling):
    """Helper function to verify the cooling parameters of the training.
    """
    if radiuscooling != "linear" and radiuscooling != "exponential":
        raise Exception("Invalid parameter for radiuscooling: " +
                        radiuscooling)
    if scalecooling != "linear" and scalecooling != "exponential":
        raise Exception("Invalid parameter for scalecooling: " +
                        scalecooling)


def _hexplot(matrix, fig, colormap):
    """Internal function to plot a hexagonal map.
    """
    umatrix_min = matrix.min()
    umatrix_max = matrix.max()
    n_rows, n_columns = matrix.shape
    cmap = plt.get_cmap(colormap)
    offsets = np.zeros((n_columns*n_rows, 2))
    facecolors = []
    for row in range(n_rows):
        for col in range(n_columns):
            if row % 2 == 0:
                offsets[row*n_columns + col] = [col+0.5, 2*n_rows-2*row]
                facecolors.append(cmap((matrix[row, col]-umatrix_min) /
                                       (umatrix_max)*255))
            else:
                offsets[row*n_columns + col] = [col, 2*n_rows-2*row]
                facecolors.append(cmap((matrix[row, col]-umatrix_min) /
                                       (umatrix_max)*255))
    polygon = np.zeros((6, 2), float)
    polygon[:, 0] = 1.1 * np.array([0.5, 0.5, 0.0, -0.5, -0.5, 0.0])
    polygon[:, 1] = 1.1 * np.array([-np.sqrt(3)/6, np.sqrt(3)/6,
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
    corners = ((-0.5, -0.5), (n_columns + 0.5, 2*n_rows + 0.5))
    ax.update_datalim(corners)
    ax.autoscale_view(tight=True)
    return offsets
