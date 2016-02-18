******************
Function Reference
******************

.. py:class:: Somoclu(n_columns, n_rows, data=None, initialcodebook=None, kerneltype=0, maptype='planar', gridtype='rectangular', compactsupport=False, neighborhood='gaussian', initialization=None)

   Class for training and visualizing a self-organizing map.
   
   :param n_columns: The number of columns in the map.
   :type n_columns: int.
   :param n_rows: The number of rows in the map.
   :type n_rows: int.
   :param data: Optional parameter to provide training data. It is not
                necessary if the map is otherwise trained outside Python,
                e.g., on a GPU cluster.
   :type data: 2D numpy.array of float32.
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
                          Default: False.
   :type compactsupport: bool.
   :param neighborhood: Optional parameter to specify the neighborhood:

                          * "gaussian": Gaussian neighborhood (default)
                          * "bubble": bubble neighborhood function
   :type neighborhood: str.
   :param initialization: Optional parameter to specify the initalization:

                          * "random": random weights in the codebook
                          * "pca": codebook is initialized from the first
                            subspace spanned by the first two eigenvectors of
                            the correlation matrix
   :type initialization: str.
   
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
      
   .. py:method:: Somoclu.train(epochs=10, radius0=0, radiusN=1, radiuscooling='linear', scale0=0.1, scaleN=0.01, scalecooling='linear')
   
      Train the map on the current data in the Somoclu object.
      
      :param epochs: The number of epochs to train the map for.
      :type epochs: int.
      :param radius0: The initial radius on the map where the update happens
                      around a best matching unit. Default value of 0 will
                      trigger a value of min(n_columns, n_rows)/2.
      :type radius0: int.
      :param radiusN: The radius on the map where the update happens around a
                      best matching unit in the final epoch. Default: 1.
      :type radiusN: int.
      :param radiuscooling: The cooling strategy between radius0 and radiusN:

                                 * "linear": Linear interpolation (default)
                                 * "exponential": Exponential decay
      :param scale0: The initial learning scale. Default value: 0.1.
      :type scale0: int.
      :param scaleN: The learning scale in the final epoch. Default: 0.01.
      :type scaleN: int.
      :param scalecooling: The cooling strategy between scale0 and scaleN:

                                 * "linear": Linear interpolation (default)
                                 * "exponential": Exponential decay
      :type scalecooling: str.
      
   
   .. py:method:: Somoclu.update_data(data)
   
      Change the data set in the Somoclu object. It is useful when the
      data is updated and the training should continue on the new data.
      
      :param data: The training data.
      :type data: 2D numpy.array of float32.
      
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

