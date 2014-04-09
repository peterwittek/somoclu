#require(Rcpp)
.onLoad <- function(lib, pkg){
    library.dynam("Rsomoclu", pkg, lib)
}
Rsomoclu.train <-
  function(input_data, nEpoch, 
           nSomX, nSomY,
           radius0, radiusN,
           radiusCooling, scale0, scaleN,
           scaleCooling, snapshots,
           kernelType, mapType
  )
  {
    res <- .Call("Rtrain", input_data, nEpoch,
                 nSomX, nSomY, radius0, radiusN,
                 radiusCooling, scale0, scaleN,
                 scaleCooling, snapshots, kernelType,
                mapType)
    res
  }
