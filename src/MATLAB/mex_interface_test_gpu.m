D = importdata('../../data/rgbs.txt');
msize = [50 50];
sMap  = som_randinit(D, 'msize', msize);
nEpoch = 100;
radius0 = 0;
radiusN = 0;
radiusCooling = 'linear';
scale0 = 0;
scaleN = 0.01;
scaleCooling = 'linear';
kernelType = 1;
mapType = 'planar';
gridType = 'square';
compactSupport = false;
[sMap, sTrain, globalBmus, uMatrix] = somoclu_train(sMap, D, 'msize', msize, 'radius0', radius0, ...
    'radiusN', radiusN, 'radiusCooling', radiusCooling, ...
    'scale0', scale0, 'scaleN', scaleN, 'scaleCooling', scaleCooling, ...
    'kernelType', kernelType, 'mapType', mapType, ...
    'gridType', gridType, 'compactSupport', compactSupport, ...
    'nEpoch', nEpoch);
    