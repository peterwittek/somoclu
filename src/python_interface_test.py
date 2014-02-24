#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import somoclu
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Somoclu python example")
    parser.add_argument('-i', '--input_file', help='Input data file path',
                        required=True)
    args = parser.parse_args()
    data = np.loadtxt(args.input_file)
    print(data)
    data = np.float32(data)
    rank = 0
    sparseData = None
    nSomX = 200
    nSomY = 200
    nVectors = data.shape[0]
    nDimensions = data.shape[1]
    data_length = nSomX * nSomY * nDimensions
    data1D = np.ndarray.flatten(data, order='C')
    codebook = np.float32(np.random.rand(data_length) - 0.5)
    nProcs = 1
    nVectorsPerRank = int(np.ceil(nVectors / (1.0 * nProcs)))
    nEpoch = 0
    currentEpoch = 0
    snapshots = 0
    enableCalculatingUMatrix = snapshots > 0
    radius0 = 0
    radiusN = 0
    radiusCooling = "linear"
    scale0 = 0.0
    scaleN = 0.01
    scaleCooling = "linear"
    kernelType = 0
    mapType = "planar"
    globalBmus = np.zeros(nVectorsPerRank *
                          np.ceil(nVectors/nVectorsPerRank) * 2,
                          dtype=np.int32)
    uMatrix = np.array((1, 0), dtype=np.float32)
    res = somoclu.trainWrapper(rank, data1D, sparseData,
                               codebook, globalBmus, uMatrix,
                               nEpoch, currentEpoch, enableCalculatingUMatrix,
                               nSomX, nSomY, nDimensions, nVectors,
                               nVectorsPerRank, radius0, radiusN,
                               radiusCooling, scale0, scaleN,
                               scaleCooling, kernelType, mapType)
