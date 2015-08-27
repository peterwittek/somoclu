from __future__ import division, print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pylab import matshow
from matplotlib.colors import LogNorm

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
        check_parameters(radiusCooling, scaleCooling, kernelType, mapType,
                         gridType)
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
        # matshow(self.umatrix, cmap=cm.Spectral_r)
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


def check_parameters(radiusCooling, scaleCooling, kernelType, mapType,
                     gridType):
    if radiusCooling != "linear" and radiusCooling != "exponential":
        raise Exception("Invalid parameter for radiusCooling: "+radiusCooling)
    if scaleCooling != "linear" and scaleCooling != "exponential":
        raise Exception("Invalid parameter for scaleCooling: "+scaleCooling)
    if mapType != "planar" and mapType != "toroid":
        raise Exception("Invalid parameter for mapType: "+mapType)
    if gridType != "square" and gridType != "hexagonal":
        raise Exception("Invalid parameter for gridType: "+gridType)
    if kernelType != 0 and kernelType != 1:
        raise Exception("Invalid parameter for kernelTye: "+kernelType)
