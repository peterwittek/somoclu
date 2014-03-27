#require(Rcpp)

Rsomoclu.train <-
  function(input_data, nEpoch, 
           nSomX, nSomY,
           radius0, radiusN,
           radiusCooling, scale0, scaleN,
           scaleCooling, snapshots,
           kernelType, mapType,
           initialCodebookFilename
  )
  {
    if(!is.loaded("Rtrain", PACKAGE = "Rsomoclu")){
      dyn.load("Rsomoclu.so")
    }
    res <- .Call("Rtrain", input_data, nEpoch,
                 nSomX, nSomY, radius0, radiusN,
                 radiusCooling, scale0, scaleN,
                 scaleCooling, snapshots, kernelType,
                 mapType, initialCodebookFilename)
    res
  }
