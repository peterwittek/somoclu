source("Rsomoclu.R", chdir<-T)
input_data <- read.table("../data/rgbs.txt")
input_data <- data.matrix(input_data)
nSomX <- 50
nSomY <- 50
# nVectors <- nrow(input_data)
# nDimensions <- ncol(input_data)
#input_data1D <- np.ndarray.flatten(input_data, order<-'C')
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
initialCodebookFilename <- ''
res <- Rsomoclu.train(input_data, nEpoch, nSomX, nSomY,
               radius0, radiusN,
               radiusCooling, scale0, scaleN,
               scaleCooling, snapshots,
               kernelType, mapType,
               initialCodebookFilename)
res$codebook
res$globalBmus
res$uMatrix