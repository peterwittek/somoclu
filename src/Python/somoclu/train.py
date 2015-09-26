from __future__ import division, print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pylab import matshow
import matplotlib.collections as mcoll
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from .somoclu_wrap import train as wrap_train


class Somoclu(object):

    def __init__(self, nSomX, nSomY, data=None, initialCodebook=None,
                 kernelType=0, mapType="planar", gridType="square",
                 compact_support=False):
        self.kernelType = kernelType
        self.mapType = mapType
        self.gridType = gridType
        self.compact_support = compact_support
        self._check_parameters()
        self.nVectors = 0
        self.nDimensions = 0
        self.data = None
        self.bmus = None
        if data is not None:
            self.update_data(data)
        self.nSomX, self.nSomY = nSomX, nSomY
        self.gridType = "square"
        self.umatrix = np.zeros(nSomX * nSomY, dtype=np.float32)
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
              scale0=0.1, scaleN=0.01, scaleCooling="linear"):
        self._check_cooling_parameters(radiusCooling, scaleCooling)
        if self.data is None:
            raise Exception("No data was provided!")
        self._init_codebook()
        self.umatrix.shape = (self.nSomY*self.nSomX, )
        wrap_train(np.ravel(self.data), nEpoch, self.nSomX, self.nSomY,
                   self.nDimensions, self.nVectors, radius0, radiusN,
                   radiusCooling, scale0, scaleN, scaleCooling,
                   self.kernelType, self.mapType, self.gridType,
                   self.compact_support, self.codebook, self.bmus,
                   self.umatrix)
        self.umatrix.shape = (self.nSomY, self.nSomX)
        self.bmus.shape = (self.nVectors, 2)
        self.codebook.shape = (self.nSomY, self.nSomX, self.nDimensions)

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

        umatrix_min = self.umatrix.min()
        umatrix_max = self.umatrix.max()
        cmap = plt.get_cmap(colormap)
        nx, ny = self.nSomX, self.nSomY
        sx = 1.1
        sy = 1.1
        offsets = np.zeros((nx*ny, 2))
        facecolors = []
        for y in range(ny):
            for x in range(nx):
                if y % 2 == 0:
                    offsets[y*nx + x] = [x+0.5, 2*y]
                    facecolors.append(cmap((self.umatrix[y, x]-umatrix_min) /
                                           (umatrix_max)*255))
                else:
                    offsets[y*nx + x] = [x, 2*y]
                    facecolors.append(cmap((self.umatrix[y, x]-umatrix_min) /
                                           (umatrix_max)*255))
        polygon = np.zeros((6, 2), float)
        polygon[:, 0] = sx * np.array([0.5, 0.5, 0.0, -0.5, -0.5, 0.0])
        polygon[:, 1] = sy * np.array([-np.sqrt(3)/6, np.sqrt(3)/6,
                                       np.sqrt(3)/2+np.sqrt(3)/6,
                                       np.sqrt(3)/6, -np.sqrt(3)/6,
                                       -np.sqrt(3)/2-np.sqrt(3)/6])
        polygons = np.expand_dims(polygon, 0) + np.expand_dims(offsets, 1)

        fig = plt.figure()
        fig.figsize = (12, 12)
        ax = fig.gca()
        collection = mcoll.PolyCollection(
            polygons,
            offsets=offsets,
            facecolors=facecolors,
            edgecolors=facecolors,
            linewidths=1.0,
            offset_position="data")
        ax.add_collection(collection, autolim=False)
        if bestmatches:
            if bestmatchcolors is None:
                colors = "white"
            else:
                colors = bestmatchcolors
            patches = []
            for x, y in self.bmus:
                coords = offsets[y*nx + x]
                patches += [Circle((coords[0], coords[1]), 0.3)]
            p = PatchCollection(patches, facecolors=colors, edgecolors=colors)
            ax.add_collection(p)
        corners = ((-0.5, -0.5), (nx + 0.5, 2*ny + 0.5))
        ax.update_datalim(corners)
        ax.autoscale_view(tight=True)
        plt.axis('off')
        plt.show()
        plt.clf()

    def _check_cooling_parameters(self, radiusCooling, scaleCooling):
        if radiusCooling != "linear" and radiusCooling != "exponential":
            raise Exception("Invalid parameter for radiusCooling: " +
                            radiusCooling)
        if scaleCooling != "linear" and scaleCooling != "exponential":
            raise Exception("Invalid parameter for scaleCooling: " +
                            scaleCooling)

    def _check_parameters(self):
        if self.mapType != "planar" and self.mapType != "toroid":
            raise Exception("Invalid parameter for mapType: " + self.mapType)
        if self.gridType != "square" and self.gridType != "hexagonal":
            raise Exception("Invalid parameter for gridType: " + self.gridType)
        if self.kernelType != 0 and self.kernelType != 1:
            raise Exception("Invalid parameter for kernelTye: " +
                            self.kernelType)

    def _init_codebook(self):
        if self.codebook is None:
            self.codebook = np.zeros(self.nSomY*self.nSomX*self.nDimensions,
                                     dtype=np.float32)
            self.codebook[0:2] = [1000, 2000]
        elif self.codebook.size != self.nSomX*self.nSomY*self.nDimensions:
            raise Exception("Invalid size for initial codebook")
        else:
            if self.codebook.dtype != np.float32:
                print("Warning: initialCodebook was not float32. A 32-bit "
                      "copy was made")
                self.codebook = np.float32(self.codebook)
        self.codebook.shape = (self.nSomY*self.nSomX*self.nDimensions, )
