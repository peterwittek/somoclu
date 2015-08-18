data = importdata('../../data/rgbs.txt');
nSomX = 50;
nSomY = 50;
dataSize = size(data);
nVectors = dataSize(1);
nDimensions = dataSize(2);
nEpoch = 10;
radius0 = 0;
radiusN = 0;
radiusCooling = 'linear';
scale0 = 0;
scaleN = 0.01;
scaleCooling = 'linear';
kernelType = 1;
mapType = 'planar';
[codebook, globalBmus, uMatrix] = MexSomoclu(data, nEpoch, nSomX, nSomY, ...
radius0, radiusN, ...
radiusCooling, scale0, scaleN, ...
scaleCooling, ...
kernelType, mapType);
