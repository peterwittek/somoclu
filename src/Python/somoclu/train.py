from __future__ import division, print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pylab import matshow
import matplotlib.collections as mcoll
import matplotlib.transforms as mtrans

from .somoclu_wrap import train as wrap_train


class Somoclu(object):

    def __init__(self, nSomX, nSomY, data=None, initialCodebook=None):
        if data is not None:
                if data.dtype != np.float32:
                    print("Warning: data was not float32. A 32-bit copy was "
                          "made")
                    self.data = np.float32(data)
                else:
                    self.data = data
                self.nVectors, self.nDimensions = data.shape
        else:
            self.nVectors = 0
            self.nDimensions = 0
        self.nSomX, self.nSomY = nSomX, nSomY
        self.gridType = "square"
        self.bmus = np.zeros(self.nVectors*2, dtype=np.intc)
        self.umatrix = np.zeros(nSomX * nSomY, dtype=np.float32)
        if initialCodebook is None:
            self.codebook = np.zeros(self.nSomY*self.nSomX*self.nDimensions,
                                     dtype=np.float32)
            self.codebook[0:2] = [1000, 2000]
        elif initialCodebook.size != self.nSomX*self.nSomY*self.nDimensions:
            raise Exception("Invalid size for initial codebook")
        else:
            if initialCodebook.dtype != np.float32:
                print("Warning: initialCodebook was not float32. A 32-bit "
                      "copy was made")
                self.codebook = np.float32(initialCodebook)
            else:
                self.codebook = initialCodebook

    def load_bmus(self, filename):
        self.bmus = np.loadtxt(filename, comments='%')
        if self.nVectors != 0 and len(self.bmus) != self.nVectors:
            raise Exception("The number of best matching units does not match"
                            "the number of data instances")
        else:
            self.nVectors = len(self.bmus)
        if max(self.bmus[:, 1]) > self.nSomX - 1 or \
                max(self.bmus[:, 2]) > self.nSomY - 1:
            raise Exception("The dimensions of the best matching units do not"
                            "match that of the map")

    def load_umatrix(self, filename):
        self.umatrix = np.loadtxt(filename, comments='%')
        if self.umatrix.shape != (self.nSomX, self.nSomY):
            raise Exception("The dimensions of the U-matrix do not "
                            "match that of the map")

    def load_codebook(self, filename):
        self.codebook = np.loadtxt(filename, comments='%')
        if self.nDimensions == 0:
            self.nDimensions = self.codebook.shape[1]
        if self.codebook.shape != (self.nSomY*self.nSomX, self.nDimensions):
            raise Exception("The dimensions of the codebook do not "
                            "match that of the map")
        self.codebook.shape = (self.nSomY, self.nSomX, self.nDimensions)

    def train(self, nEpoch=10, radius0=0, radiusN=1, radiusCooling="linear",
              scale0=0.1, scaleN=0.01, scaleCooling="linear",
              kernelType=0, mapType="planar", gridType="square",
              compact_support=False):
        self._check_parameters(radiusCooling, scaleCooling, kernelType,
                               mapType, gridType)
        wrap_train(np.ravel(self.data), nEpoch, self.nSomX, self.nSomY,
                   self.nDimensions, self.nVectors, radius0, radiusN,
                   radiusCooling, scale0, scaleN, scaleCooling,
                   kernelType, mapType, gridType, compact_support,
                   self.codebook, self.bmus,
                   self.umatrix)
        self.umatrix.shape = (self.nSomY, self.nSomX)
        self.bmus.shape = (self.nVectors, 2)
        self.codebook.shape = (self.nSomY, self.nSomX, self.nDimensions)

    def view_component_planes(self, dimensions=None):
        if dimensions is None:
            dimensions = range(self.nDimensions)
        for i in dimensions:
            matshow(self.codebook[:, :, i], cmap=cm.Spectral_r)
        plt.clf()

    def view_umatrix(self, colormap='spectral', colorbar=False,
                     bestmatches=False, bestmatchcolors=None, labels=None):
        if self.gridType == "hexagonal":
            return self._view_umatrixhex(colormap, colorbar, bestmatches,
                                         bestmatchcolors, labels)
        plt.figure(figsize=(12, 7))
        kw = {'origin': 'lower', 'aspect': 'equal'}
        imgplot = plt.imshow(self.umatrix, **kw)
        imgplot.set_cmap(colormap)
        if colorbar:
            plt.colorbar(orientation="horizontal", shrink=0.5)
        if bestmatches:
            if bestmatchcolors is None:
                colors = "white"
            else:
                colors = bestmatchcolors
            plt.scatter(self.bmus[:, 0], self.bmus[:, 1], c=colors)
        if labels is not None:
            for label, x, y in zip(labels, self.bmus[:, 0],
                                   self.bmus[:, 1]):
                if label is not None:
                    plt.annotate(label, xy=(x, y), xytext=(10, -5),
                                 textcoords='offset points', ha='left',
                                 va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.3',
                                           fc='white', alpha=0.8))
        plt.axis('off')
        plt.show()
        plt.clf()

    def _view_umatrixhex(self, colormap='spectral', colorbar=False,
                         bestmatches=False, bestmatchcolors=None, labels=None):

        fig = plt.figure()
        fig.figsize = (12, 7)
        ax = fig.gca()

        nx, ny = self.nSomX, self.nSomY

        x = np.array(range(self.nSomX), float)
        y = np.array(range(self.nSomY), float)
        xmin = np.amin(x)
        xmax = np.amax(x)
        ymin = np.amin(y)
        ymax = np.amax(y)
        # to avoid issues with singular data, expand the min/max pairs
        xmin, xmax = mtrans.nonsingular(xmin, xmax, expander=0.1)
        ymin, ymax = mtrans.nonsingular(ymin, ymax, expander=0.1)

        # In the x-direction, the hexagons exactly cover the region from
        # xmin to xmax. Need some padding to avoid roundoff errors.
        padding = 1.e-9 * (xmax - xmin)
        xmin -= padding
        xmax += padding
        sx = (xmax - xmin) / nx
        sy = (ymax - ymin) / ny

        x = (x - xmin) / sx
        y = (y - ymin) / sy
        ix1 = np.round(x).astype(int)
        iy1 = np.round(y).astype(int)
        ix2 = np.floor(x).astype(int)
        iy2 = np.floor(y).astype(int)

        nx1 = nx + 1
        ny1 = ny + 1
        nx2 = nx
        ny2 = ny
        n = nx1 * ny1 + nx2 * ny2

        d1 = (x - ix1) ** 2 + 3.0 * (y - iy1) ** 2
        d2 = (x - ix2 - 0.5) ** 2 + 3.0 * (y - iy2 - 0.5) ** 2
        bdist = (d1 < d2)
        accum = np.zeros(n)
        # Create appropriate views into "accum" array.
        lattice1 = accum[:nx1 * ny1]
        lattice2 = accum[nx1 * ny1:]
        lattice1.shape = (nx1, ny1)
        lattice2.shape = (nx2, ny2)

        for i in range(len(x)):
            if bdist[i]:
                if ((ix1[i] >= 0) and (ix1[i] < nx1) and
                        (iy1[i] >= 0) and (iy1[i] < ny1)):
                    lattice1[ix1[i], iy1[i]] += 1
            else:
                if ((ix2[i] >= 0) and (ix2[i] < nx2) and
                        (iy2[i] >= 0) and (iy2[i] < ny2)):
                    lattice2[ix2[i], iy2[i]] += 1
        accum = np.hstack((lattice1.astype(float).ravel(),
                           lattice2.astype(float).ravel()))
        good_idxs = ~np.isnan(accum)

        offsets = np.zeros((n, 2), float)
        offsets[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
        offsets[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
        offsets[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
        offsets[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
        offsets[:, 0] *= sx
        offsets[:, 1] *= sy
        offsets[:, 0] += xmin
        offsets[:, 1] += ymin
        # remove accumulation bins with no data
        offsets = offsets[good_idxs, :]
        polygon = np.zeros((6, 2), float)
        polygon[:, 0] = sx * np.array([0.5, 0.5, 0.0, -0.5, -0.5, 0.0])
        polygon[:, 1] = sy * np.array([-0.5, 0.5, 1.0, 0.5, -0.5, -1.0])
        polygons = np.expand_dims(polygon, 0) + np.expand_dims(offsets, 1)
        umatrix_min = self.umatrix.min()
        umatrix_max = self.umatrix.max()
        cmap = plt.get_cmap(colormap)
        facecolors = [cmap((i-umatrix_min)/(umatrix_max)*255)
                      for i in self.umatrix.reshape(self.umatrix.size)]
        collection = mcoll.PolyCollection(
            polygons,
            offsets=offsets,
            facecolors=facecolors,
            edgecolors='white',
            linewidths=0.2,
            transOffset=mtrans.IdentityTransform(),
            offset_position="data")
        corners = ((xmin, ymin), (xmax, ymax))
        ax.update_datalim(corners)
        ax.autoscale_view(tight=True)
        ax.add_collection(collection, autolim=False)
        plt.axis('off')
        plt.show()
        plt.clf()

    def _check_parameters(self, radiusCooling, scaleCooling, kernelType,
                          mapType, gridType):
        if radiusCooling != "linear" and radiusCooling != "exponential":
            raise Exception("Invalid parameter for radiusCooling: " +
                            radiusCooling)
        else:
            self.radiusCooling = radiusCooling
        if scaleCooling != "linear" and scaleCooling != "exponential":
            raise Exception("Invalid parameter for scaleCooling: " +
                            scaleCooling)
        else:
            self.scaleCooling = scaleCooling
        if mapType != "planar" and mapType != "toroid":
            raise Exception("Invalid parameter for mapType: " + mapType)
        else:
            self.mapType = mapType
        if gridType != "square" and gridType != "hexagonal":
            raise Exception("Invalid parameter for gridType: " + gridType)
        else:
            self.gridType = gridType
        if kernelType != 0 and kernelType != 1:
            raise Exception("Invalid parameter for kernelTye: " +
                            kernelType)
        else:
            self.kernelType = kernelType
