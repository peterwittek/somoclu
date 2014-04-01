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
snapshots <- 0
# initialCodebookFilename <- ''
res <- Rsomoclu.train(input_data, nEpoch, nSomX, nSomY,
                      radius0, radiusN,
                      radiusCooling, scale0, scaleN,
                      scaleCooling, snapshots,
                      kernelType, mapType)
#                initialCodebookFilename)
res$codebook
res$globalBmus
res$uMatrix
