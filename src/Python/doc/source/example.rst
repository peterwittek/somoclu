*******
Example
*******

The following code generates 150 points in three different classes and trains a
small self-organizing map on them. The component planes are displayed next, then
the U-matrix.

::

    import somoclu
    import numpy as np
    import matplotlib.pyplot as plt
    c1 = np.random.rand(50, 2)/5
    c2 = (0.2, 0.5) + np.random.rand(50, 2)/5
    c3 = (0.4, 0.1) + np.random.rand(50, 2)/5
    data = np.float32(np.concatenate((c1, c2, c3)))
    colors = [ "red" ] * 50
    colors.extend(["green"] * 50)
    colors.extend(["blue"] * 50)
    plt.scatter(data[:,0], data[:,1], c=colors)
    labels = range(150)
    n_columns, n_rows = 50, 30
    som = somoclu.Somoclu(n_columns, n_rows, data=data, maptype="planar", 
                          gridtype="rectangular")
    som.train(epochs=10)    
    som.view_component_planes()
    som.view_umatrix(bestmatches=True, bestmatchcolors=colors, labels=labels)
