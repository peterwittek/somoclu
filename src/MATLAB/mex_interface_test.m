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
kernelType = 0;
mapType = 'planar';
gridType = 'rectangular';
compactSupport = false;
neighborhood = 'gaussian';
stdCoeff = 0.5;
verbose = 0;
[sMap, sTrain, globalBmus, uMatrix] = somoclu_train(sMap, D, 'msize', msize, 'radius0', radius0, ...
    'radiusN', radiusN, 'radiusCooling', radiusCooling, ...
    'scale0', scale0, 'scaleN', scaleN, 'scaleCooling', scaleCooling, ...
    'kernelType', kernelType, 'mapType', mapType, ...
    'gridType', gridType, 'compactSupport', compactSupport, ...
    'neighborhood', neighborhood, 'stdCoeff', stdCoeff, 'nEpoch', nEpoch, ...
    'verbose', verbose);
    
