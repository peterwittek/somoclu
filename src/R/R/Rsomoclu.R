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
           kernelType, mapType, codebook=NULL) {
    if (is.null(codebook)) {
      codebook <- numeric(nSomX*nSomY*dim(input_data)[2])
      codebook[1] <- 1000
      codebook[1] <- 2000
    }
    res <- .Call("Rtrain", input_data, nEpoch,
                 nSomX, nSomY, radius0, radiusN,
                 radiusCooling, scale0, scaleN,
                 scaleCooling, kernelType, mapType, codebook)
    res
}
