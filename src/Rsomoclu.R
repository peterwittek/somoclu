require(Rcpp)
# Rsomoclu <- Module("Rsomoclu",getDynLib("Rsomoclu.so"))
# dyn.unload("Rsomoclu.so")
dyn.load("Rsomoclu.so")
Rsomoclu.train <-
   function(input_data, nEpoch, 
            nSomX, nSomY,
#            nDimensions, nVectors,
           radius0, radiusN,
           radiusCooling, scale0, scaleN,
           scaleCooling, snapshots,
           kernelType, mapType,
           initialCodebookFilename
           )
  {
#     codebook
#     globalBmus
#     uMatrix
#     codebook_size <- nSomY * nSomX * nDimensions
#     globalBmus_size <- nVectors * ceiling(nVectors/nVectors)*2
#     uMatrix_size <- nSomX * nSomY
    res <- .Call("Rtrain", input_data, nEpoch,
                 nSomX, nSomY, radius0, radiusN,
                 radiusCooling, scale0, scaleN,
                 scaleCooling, snapshots, kernelType,
                 mapType, initialCodebookFilename)
    res
  }
