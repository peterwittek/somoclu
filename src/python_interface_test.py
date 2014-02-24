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
    nSomX = 50
    nSomY = 50
    nVectors = data.shape[0]
    nDimensions = data.shape[1]
    data1D = np.ndarray.flatten(data, order='C')
    nEpoch = 10
    radius0 = 0
    radiusN = 0
    radiusCooling = "linear"
    scale0 = 0
    scaleN = 0.01
    scaleCooling = "linear"
    kernelType = 0
    mapType = "planar"
    snapshots = 0
    initialCodebookFilename = ''
    res = somoclu.trainWrapper(data1D, nEpoch, nSomX, nSomY,
                               nDimensions, nVectors,
                               radius0, radiusN,
                               radiusCooling, scale0, scaleN,
                               scaleCooling, snapshots,
                               kernelType, mapType, initialCodebookFilename)
