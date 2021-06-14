******************
Function Reference
******************

.. py:class:: Somoclu(n_columns, n_rows, initialcodebook=None, kerneltype=0, maptype='planar', gridtype='rectangular', compactsupport=False, neighborhood='gaussian', vect_distance='euclidean', std_coeff=0.5, initialization=None)

   Class for training and visualizing a self-organizing map.

    Attributes:
        codebook     The codebook of the self-organizing map.
        bmus         The BMUs corresponding to the data points.

   :param n_columns: The number of columns in the map.
   :type n_columns: int.
   :param n_rows: The number of rows in the map.
   :type n_rows: int.
   :param initialcodebook: Optional parameter to start the training with a
                           given codebook.
   :type initialcodebook: 2D numpy.array of float32.
   :param kerneltype: Optional parameter to specify which kernel to use:

                          * 0: dense CPU kernel (default)
                          * 1: dense GPU kernel (if compiled with it)
   :type kerneltype: int.
   :param maptype: Optional parameter to specify the map topology:

                          * "planar": Planar map (default)
                          * "toroid": Toroid map
   :type maptype: str.
   :param gridtype: Optional parameter to specify the grid form of the nodes:

                          * "rectangular": rectangular neurons (default)
                          * "hexagonal": hexagonal neurons
   :type gridtype: str.
   :param compactsupport: Optional parameter to cut off map updates beyond the
                          training radius with the Gaussian neighborhood.
                          Default: True.
   :type compactsupport: bool.
   :param neighborhood: Optional parameter to specify the neighborhood:

                          * "gaussian": Gaussian neighborhood (default)
                          * "bubble": bubble neighborhood function
   :param vect_distance: Optional parameter to specify the vector distance:

                          * "euclidean": Euclidean distance (default)
                          * "norm-inf": Infinity-norm distance (max among components)
                          * "norm-p": p-norm (p-th root of summed differences raised to the power of p), works only if kerneltype is 0
   :type neighborhood: str.
   :param std_coeff: Optional parameter to set the coefficient in the Gaussian
                     neighborhood function exp(-||x-y||^2/(2*(coeff*radius)^2))
                     Default: 0.5
   :type std_coeff: float.
   :param initialization: Optional parameter to specify the initalization:

                          * "random": random weights in the codebook
                          * "pca": codebook is initialized from the first
                            subspace spanned by the first two eigenvectors of
                            the correlation matrix
   :type initialization: str.
   :param verbose: Optional parameter to specify verbosity (0, 1, or 2).
   :type verbose: int.


   .. py:method:: Somoclu.cluster(algorithm=None)

      Cluster the codebook. The clusters of the data instances can be
      assigned based on the BMUs. The method populates the class variable
      Somoclu.clusters. If viewing methods are called after clustering, but
      without colors for best matching units, colors will be automatically
      assigned based on cluster membership.

      :param algorithm: Optional parameter to specify a scikit-learn
                        clustering algorithm. The default is K-means with
                        eight clusters.
      :type filename: sklearn.base.ClusterMixin.

   .. py:method:: Somoclu.get_surface_state(data=None))

      Return the dot product of the codebook and the data.

      :param data: Optional parameter to specify data, otherwise the
                   data used previously to train the SOM is used.
      :type data: 2D numpy.array of float32.

      :returns: The the dot product of the codebook and the data.
      :rtype: 2D numpy.array


   .. py:method:: Somoclu.get_bmus(activation_map)

      Return Best Matching Unit indexes of the activation map.

      :param activation_map: Activation map computed with self.get_surface_state()
      :type activation_map: 2D numpy.array

      :returns: The bmus indexes corresponding to this activation map
                (same as self.bmus for the training samples).
      :rtype: 2D numpy.array


   .. py:method:: Somoclu.load_bmus(filename)

      Load the best matching units from a file to the Somoclu object.

      :param filename: The name of the file.
      :type filename: str.


   .. py:method:: Somoclu.load_codebook(filename)

      Load the codebook from a file to the Somoclu object.

      :param filename: The name of the file.
      :type filename: str.


   .. py:method:: Somoclu.load_umatrix(filename)

      Load the umatrix from a file to the Somoclu object.

      :param filename: The name of the file.
      :type filename: str.

   .. py:method:: Somoclu.train(data=None, epochs=10, radius0=0, radiusN=1, radiuscooling='linear', scale0=0.1, scaleN=0.01, scalecooling='linear')

      Train the map on the current data in the Somoclu object.

      :param data: Training data..
      :type data: 2D numpy.array of float32.
      :param epochs: The number of epochs to train the map for.
      :type epochs: int.
      :param radius0: The initial radius on the map where the update happens
                      around a best matching unit. Default value of 0 will
                      trigger a value of min(n_columns, n_rows)/2.
      :type radius0: float.
      :param radiusN: The radius on the map where the update happens around a
                      best matching unit in the final epoch. Default: 1.
      :type radiusN: float.
      :param radiuscooling: The cooling strategy between radius0 and radiusN:

                                 * "linear": Linear interpolation (default)
                                 * "exponential": Exponential decay
      :param scale0: The initial learning scale. Default value: 0.1.
      :type scale0: float.
      :param scaleN: The learning scale in the final epoch. Default: 0.01.
      :type scaleN: float.
      :param scalecooling: The cooling strategy between scale0 and scaleN:

                                 * "linear": Linear interpolation (default)
                                 * "exponential": Exponential decay
      :type scalecooling: str.

   .. py:method:: Somoclu.view_activation_map(data_vector=None, data_index=None, activation_map=None, figsize=None, colormap=cm.Spectral_r, colorbar=False, bestmatches=False, bestmatchcolors=None, labels=None, zoom=None, filename=None)

      Plot the activation map of a given data instance or a new data
      vector

      :param data_vector: Optional parameter for a new vector
      :type data_vector: numpy.array
      :param data_index: Optional parameter for the index of the data instance
      :type data_index: int.
      :param activation_map: Optional parameter to pass the an activation map
      :type activation_map: numpy.array
      :param figsize: Optional parameter to specify the size of the figure.
      :type figsize: (int, int)
      :param colormap: Optional parameter to specify the color map to be
                       used.
      :type colormap: matplotlib.colors.Colormap
      :param colorbar: Optional parameter to include a colormap as legend.
      :type colorbar: bool.
      :param bestmatches: Optional parameter to plot best matching units.
      :type bestmatches: bool.
      :param bestmatchcolors: Optional parameter to specify the color of each
                              best matching unit.
      :type bestmatchcolors: list of int.
      :param labels: Optional parameter to specify the label of each point.
      :type labels: list of str.
      :param zoom: Optional parameter to zoom into a region on the map. The
                   first two coordinates of the tuple are the row limits, the
                   second tuple contains the column limits.
      :type zoom: ((int, int), (int, int))
      :param filename: If specified, the plot will not be shown but saved to
                       this file.
      :type filename: str.

   .. py:method:: Somoclu.view_component_planes(dimensions=None, figsize=None, colormap=cm.Spectral_r, colorbar=False, bestmatches=False, bestmatchcolors=None, labels=None, zoom=None, filename=None)

      Observe the component planes in the codebook of the SOM.

      :param dimensions: Optional parameter to specify along which dimension
                         or dimensions should the plotting happen. By
                         default, each dimension is plotted in a sequence of
                         plots.
      :type dimension: int or list of int.
      :param figsize: Optional parameter to specify the size of the figure.
      :type figsize: (int, int)
      :param colormap: Optional parameter to specify the color map to be
                       used.
      :type colormap: matplotlib.colors.Colormap
      :param colorbar: Optional parameter to include a colormap as legend.
      :type colorbar: bool.
      :param bestmatches: Optional parameter to plot best matching units.
      :type bestmatches: bool.
      :param bestmatchcolors: Optional parameter to specify the color of each
                              best matching unit.
      :type bestmatchcolors: list of int.
      :param labels: Optional parameter to specify the label of each point.
      :type labels: list of str.
      :param zoom: Optional parameter to zoom into a region on the map. The
                   first two coordinates of the tuple are the row limits, the
                   second tuple contains the column limits.
      :type zoom: ((int, int), (int, int))
      :param filename: If specified, the plot will not be shown but saved to
                       this file.
      :type filename: str.

   .. py:method:: Somoclu.view_similarity_matrix(data=None, labels=None, figsize=None, filename=None)

      Plot the similarity map according to the activation map

      :param data: Optional parameter for data points to calculate the
                   similarity with
      :type data: numpy.array
      :param figsize: Optional parameter to specify the size of the figure.
      :type figsize: (int, int)
      :param labels: Optional parameter to specify the label of each point.
      :type labels: list of str.
      :param filename: If specified, the plot will not be shown but saved to
                       this file.
      :type filename: str.

   .. py:method:: Somoclu.view_umatrix(figsize=None, colormap=<Mock name=cm.Spectral_r, colorbar=False, bestmatches=False, bestmatchcolors=None, labels=None, zoom=None, filename=None)

      Plot the U-matrix of the trained map.

      :param figsize: Optional parameter to specify the size of the figure.
      :type figsize: (int, int)
      :param colormap: Optional parameter to specify the color map to be
                       used.
      :type colormap: matplotlib.colors.Colormap
      :param colorbar: Optional parameter to include a colormap as legend.
      :type colorbar: bool.
      :param bestmatches: Optional parameter to plot best matching units.
      :type bestmatches: bool.
      :param bestmatchcolors: Optional parameter to specify the color of each
                              best matching unit.
      :type bestmatchcolors: list of int.
      :param labels: Optional parameter to specify the label of each point.
      :type labels: list of str.
      :param zoom: Optional parameter to zoom into a region on the map. The
                   first two coordinates of the tuple are the row limits, the
                   second tuple contains the column limits.
      :type zoom: ((int, int), (int, int))
      :param filename: If specified, the plot will not be shown but saved to
                       this file.
      :type filename: str.
