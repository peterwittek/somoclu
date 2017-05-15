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
           compactSupport=TRUE, neighborhood="gaussian", stdCoeff=0.5, 
           codebook=NULL) {
    if (is.null(codebook)) {
      codebook <- matrix(data = 0, nrow = nSomX * nSomY, ncol = dim(input_data)[2])
      codebook[1, 1] <- 1000
      codebook[2, 1] <- 2000
    }
    res <- .Call("Rtrain", input_data, nEpoch,
                 nSomX, nSomY, radius0, radiusN,
                 radiusCooling, scale0, scaleN,
                 scaleCooling, kernelType, mapType,
                 gridType, compactSupport, neighborhood, stdCoeff,
                 codebook)
    res
  }

Rsomoclu.kohonen <- function (input_data, result, n.hood = NULL, toroidal = FALSE) 
{
  mapping <- map(som(result$codebook), newdata = input_data)
  nSomX = nrow(result$uMatrix)
  nSomY = ncol(result$uMatrix)
  grid = somgrid(nSomX, nSomY)
  if (missing(n.hood)) {
    n.hood <- switch(grid$topo, hexagonal = "circular", 
                     rectangular = "square")
  }
  else {
    n.hood <- match.arg(n.hood, c("circular", "square"))
  }
  grid$n.hood <- n.hood
  sommap = structure(list(data = list(input_data), grid = grid, 
                          codes = list(result$codebook), changes = NULL, unit.classif = mapping$unit.classif, 
                          distances = mapping$distances, toroidal = toroidal, user.weights = 1, distance.weights=1,
                          whatmap=1,  maxNA.fraction = 0L, method = "som", dist.fcts="sumofsquares"), class = "kohonen")
  sommap
}

