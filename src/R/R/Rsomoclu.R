#require(Rcpp)
.onLoad <- function(lib, pkg){
    library.dynam("Rsomoclu", pkg, lib)
}
Rsomoclu.train <-
  function(input_data, nEpoch, 
           nSomX, nSomY,
           radius0, radiusN,
           radiusCooling, scale0, scaleN,
           scaleCooling,
           kernelType=0, mapType="planar", gridType="rectangular", 
           compactSupport=FALSE, neighborhood="gaussian", codebook=NULL) {
    if (is.null(codebook)) {
      codebook <- matrix(data = 0, nrow = nSomX * nSomY, ncol = dim(input_data)[2])
      codebook[1, 1] <- 1000
      codebook[2, 1] <- 2000
    }
    res <- .Call("Rtrain", input_data, nEpoch,
                 nSomX, nSomY, radius0, radiusN,
                 radiusCooling, scale0, scaleN,
                 scaleCooling, kernelType, mapType,
                 gridType, compactSupport, neighborhood,
                 codebook)
    res
}
