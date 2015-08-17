from __future__ import division, print_function
import numpy as np
import matplotlib.cm as cm
from matplotlib.pylab import matshow

from somoclu_wrap import trainWrapper


class Somoclu(object):


    def __init__(self, data, nSomX, nSomY, initialMap = None):
        if data.dtype != np.float32:
            print("Warning: data was not float32. A 32-bit copy was made")
            self.data = np.float32(data)
        else:
            self.data = data
        self.nVectors, self.nDimensions = data.shape
        self.nSomX, self.nSomY = nSomX, nSomY
        self.codebook = np.zeros(nSomY * nSomX * self.nDimensions, dtype=np.float32)
        self.globalBmus = np.zeros(self.nVectors*2, dtype=np.intc)
        self.uMatrix = np.zeros(nSomX * nSomY, dtype=np.float32)
        self.initialMap = initialMap


    def train(self, nEpoch=10, radius0=0, radiusN=1, radiusCooling="linear",
              scale0=0.1, scaleN=0.01, scaleCooling="linear",
              kernelType=0, mapType="planar"):
        snapshots = 0
        initialCodebookFilename = ""
        trainWrapper(np.ravel(self.data), nEpoch, self.nSomX, self.nSomY,
                     self.nDimensions, self.nVectors, radius0, radiusN,
                    radiusCooling, scale0, scaleN, scaleCooling, snapshots,
                    kernelType, mapType, initialCodebookFilename,
                    self.codebook, self.globalBmus, self.uMatrix)
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
