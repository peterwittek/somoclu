library('Rsomoclu')
data_file <- system.file("data", "rgbs.txt.gz", package = 'Rsomoclu')
input_data <- read.table(data_file)
input_data <- data.matrix(input_data)
nSomX <- 50
nSomY <- 50
nEpoch <- 10
radius0 <- 0
radiusN <- 0
radiusCooling <- "linear"
scale0 <- 0
scaleN <- 0.01
scaleCooling <- "linear"
kernelType <- 0
mapType <- "planar"
gridType <- "rectangular"
compactSupport <- FALSE
codebook <- NULL
neighborhood <- "gaussian"
res <- Rsomoclu.train(input_data, nEpoch, nSomX, nSomY,
                      radius0, radiusN,
                      radiusCooling, scale0, scaleN,
                      scaleCooling,
                      kernelType, mapType, gridType, 
                      compactSupport, neighborhood, codebook)
res$codebook
res$globalBmus
res$uMatrix
library('kohonen')
sommap = Rsomoclu.kohonen(input_data, res)
## Show 'codebook'
plot(sommap, type="codes", main = c("Codes X", "Codes Y"))
## Show 'component planes'
plot(sommap, type = "property", property = sommap$codes[,1],
     main = colnames(sommap$codes)[1])
## Show 'U-Matrix'
plot(sommap, type="dist.neighbours")
