%SOMOCLU_SOM_DEMO1 Basic properties and behaviour of the Self-Organizing Map.

% Adapted from SOM Toolbox 2.0
% http://www.cis.hut.fi/projects/somtoolbox/

clf reset;
figure(gcf)
echo on
clc
%    ==========================================================
%    SOMOCLU_SOM_DEMO1 - BEHAVIOUR AND PROPERTIES OF SOM
%    ==========================================================

%     som_randinit   - Create and initialize a SOM.
%     som_lininit    - Create and initialize a SOM.
%     somoclu_train   - Train a SOM.

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


pause % Strike any key to continue with 3D data...

clf

clc
%    TRAINING DATA: THE UNIT CUBE
%    ============================

%    Above, the map dimension was equal to input space dimension: both
%    were 2-dimensional. Typically, the input space dimension is much
%    higher than the 2-dimensional map. In this case the map cannot
%    follow perfectly the data set any more but must find a balance
%    between two goals:

%      - data representation accuracy
%      - data set topology representation accuracy    

%    Here are 500 data points sampled from the unit cube:

D = rand(500,3);

subplot(1,3,1), plot3(D(:,1),D(:,2),D(:,3),'+r')
view(3), axis on, rotate3d on
title('Data')

%    The ROTATE3D command enables you to rotate the picture by
%    dragging the pointer above the picture with the leftmost mouse
%    button pressed down.

pause % Strike any key to train the SOM...



sMap  = som_randinit(D);
[sMap, sTrain, globalBmus, uMatrix] = somoclu_train(sMap, D, 'nEpoch', 100);
U=som_umat(sMap)
%    Here, the linear initialization is done again, so that 
%    the results can be compared.

sMap0 = som_lininit(D); 

subplot(1,3,2)
som_grid(sMap0,'Coord',sMap0.codebook,...
	 'Markersize',2,'Linecolor','k','Surf',sMap0.codebook(:,3)) 
axis([0 1 0 1 0 1]), view(-120,-25), title('After initialization')

subplot(1,3,3)
som_grid(sMap,'Coord',sMap.codebook,...
	 'Markersize',2,'Linecolor','k','Surf',sMap.codebook(:,3)) 
axis([0 1 0 1 0 1]), view(3), title('After training'), hold on

%    Here you can see that the 2-dimensional map has folded into the
%    3-dimensional space in order to be able to capture the whole data
%    space. 

pause % Strike any key to evaluate the quality of maps...


clc
%    BEST-MATCHING UNITS (BMU)
%    =========================

%    Before going to the quality, an important concept needs to be
%    introduced: the Best-Matching Unit (BMU). The BMU of a data
%    vector is the unit on the map whose model vector best resembles
%    the data vector. In practise the similarity is measured as the
%    minimum distance between data vector and each model vector on the
%    map. The BMUs can be calculated using function SOM_BMUS. This
%    function gives the index of the unit.

%    Here the BMU is searched for the origin point (from the
%    trained map):

bmu = som_bmus(sMap,[0 0 0]);
%    Here the corresponding unit is shown in the figure. You can
%    rotate the figure to see better where the BMU is.

co = sMap.codebook(bmu,:);
text(co(1),co(2),co(3),'BMU','Fontsize',20)
plot3([0 co(1)],[0 co(2)],[0 co(3)],'ro-')

pause 

%    VISUALIZATION OF CLUSTERS: DISTANCE MATRICES
%    ===============================================

%    Distance matrices are typically used to show the cluster
%    structure of the SOM. They show distances between neighboring
%    units, and are thus closely related to single linkage clustering
%    techniques. The most widely used distance matrix technique is
%    the U-matrix. 

%    Here, the U-matrix of the map is shown (using all three
%    components in the distance calculation):

colormap(1-gray)
som_show(sMap,'umat','all');
pause % Strike any key to continue...
%    VISUALIZATION OF COMPONENTS: COMPONENT PLANES
%    ================================================

%    The component planes visualizations shows what kind of values the
%    prototype vectors of the map units have for different vector
%    components.

%    Here is the U-matrix and the three component planes of the map.

som_show(sMap)
% Please see demos that installed with som-toolbox for more 
% visualization examples...

% Or You can save the results with functions in Databionic ESOM Tools for further
% analysis using Databionic ESOM Tools
saveumx('demo', uMatrix);
savebm('demo', globalBmus);
savewts('demo', sMap.codebook, msize(1,1), msize(1,2));