%SOM_DEMO1 Basic properties and behaviour of the Self-Organizing Map.

% Adapted from SOM Toolbox 2.0, February 11th, 2000 by Juha Vesanto
% http://www.cis.hut.fi/projects/somtoolbox/

% Version 1.0beta juuso 071197
% Version 2.0beta juuso 030200 

clf reset;
figure(gcf)
echo on



clc
%    ==========================================================
%    SOM_DEMO1 - BEHAVIOUR AND PROPERTIES OF SOM
%    ==========================================================

%    som_make        - Create, initialize and train a SOM.
%     som_randinit   - Create and initialize a SOM.
%     som_lininit    - Create and initialize a SOM.
%     som_seqtrain   - Train a SOM.
%     som_batchtrain - Train a SOM.
%    som_bmus        - Find best-matching units (BMUs).
%    som_quality     - Measure quality of SOM.

%    SELF-ORGANIZING MAP (SOM):

%    A self-organized map (SOM) is a "map" of the training data, 
%    dense where there is a lot of data and thin where the data 
%    density is low. 

%    The map constitutes of neurons located on a regular map grid. 
%    The lattice of the grid can be either hexagonal or rectangular.

subplot(1,2,1)
som_cplane('hexa',[10 15],'none')
title('Hexagonal SOM grid')

subplot(1,2,2)
som_cplane('rect',[10 15],'none')
title('Rectangular SOM grid')

%    Each neuron (hexagon on the left, rectangle on the right) has an
%    associated prototype vector. After training, neighboring neurons
%    have similar prototype vectors.

%    The SOM can be used for data visualization, clustering (or 
%    classification), estimation and a variety of other purposes.

pause % Strike any key to continue...

clf
clc
%    INITIALIZE AND TRAIN THE SELF-ORGANIZING MAP
%    ============================================

%    Here are 300 data points sampled from the unit square:

D = rand(300,2);

%    The map will be a 2-dimensional grid of size 10 x 10.

msize = [10 10];

%    SOM_RANDINIT and SOM_LININIT can be used to initialize the
%    prototype vectors in the map. The map size is actually an
%    optional argument. If omitted, it is determined automatically
%    based on the amount of data vectors and the principal
%    eigenvectors of the data set. Below, the random initialization
%    algorithm is used.

sMap  = som_randinit(D, 'msize', msize);

%    Actually, each map unit can be thought as having two sets
%    of coordinates: 
%      (1) in the input space:  the prototype vectors
%      (2) in the output space: the position on the map
%    In the two spaces, the map looks like this: 

subplot(1,3,1) 
som_grid(sMap)
axis([0 11 0 11]), view(0,-90), title('Map in output space')

subplot(1,3,2) 
plot(D(:,1),D(:,2),'+r'), hold on
som_grid(sMap,'Coord',sMap.codebook)
title('Map in input space')

%    The black dots show positions of map units, and the gray lines
%    show connections between neighboring map units.  Since the map
%    was initialized randomly, the positions in in the input space are
%    completely disorganized. The red crosses are training data.

pause % Strike any key to train the SOM...

%    During training, the map organizes and folds to the training
%    data. Here, the sequential training algorithm is used:

% sMap  = som_seqtrain(sMap,D,'radius',[5 1],'trainlen',10);
sMap  = somoclu_train(sMap, D, 'msize', msize, 'radius',[5 1], ...
    'nEpoch', 100);
% for GPU kernel:
% sMap  = somoclu_train(sMap, D, 'msize', msize, 'radius',[5 1], ...
%     'nEpoch', 100, 'kernelType', 1);

subplot(1,3,3)
som_grid(sMap,'Coord',sMap.codebook)
hold on, plot(D(:,1),D(:,2),'+r')
title('Trained map')


