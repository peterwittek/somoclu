from __future__ import division, print_function
import numpy as np
import matplotlib.cm as cm
from matplotlib.pylab import matshow

from somoclu_wrap import trainWrapper


class Somoclu(object):


    def __init__(self, data, nSomX, nSomY, initialCodebook=None):
        if data.dtype != np.float32:
            print("Warning: data was not float32. A 32-bit copy was made")
            self.data = np.float32(data)
        else:
            self.data = data
        self.nVectors, self.nDimensions = data.shape
        self.nSomX, self.nSomY = nSomX, nSomY
        self.globalBmus = np.zeros(self.nVectors*2, dtype=np.intc)
        self.uMatrix = np.zeros(nSomX * nSomY, dtype=np.float32)
        if initialCodebook is None:
            self.codebook = np.zeros(self.nSomY*self.nSomX*self.nDimensions,
                                     dtype=np.float32)
            self.codebook[0:2] = [1000, 2000]
        elif initialCodebook.size != self.nSomX*self.nSomY*self.nDimensions:
            raise Exception("Invalid size for initial codebook")
        else:
            if initialCodebook.dtype != np.float32:
                print("Warning: initialCodebook was not float32. A 32-bit copy was made")
                self.codebook = np.float32(initialCodebook)
            else:
                self.codebook = initialCodebook


    def train(self, nEpoch=10, radius0=0, radiusN=1, radiusCooling="linear",
              scale0=0.1, scaleN=0.01, scaleCooling="linear",
              kernelType=0, mapType="planar"):
        trainWrapper(np.ravel(self.data), nEpoch, self.nSomX, self.nSomY,
                     self.nDimensions, self.nVectors, radius0, radiusN,
                    radiusCooling, scale0, scaleN, scaleCooling,
                    kernelType, mapType, self.codebook, self.globalBmus,
                    self.uMatrix)
        self.uMatrix.shape = (self.nSomY, self.nSomX)
        self.globalBmus.shape = (self.nVectors, 2)
        self.codebook.shape = (self.nSomY, self.nSomX, self.nDimensions)

    def view_component_planes(self, dimensions=None):
        if dimensions is None:
            dimensions = range(self.nDimensions)
        for i in dimensions:
            matshow(self.codebook[:,:,i], cmap=cm.Spectral_r)

    def view_U_matrix(self):
        matshow(self.uMatrix, cmap=cm.Spectral_r)
