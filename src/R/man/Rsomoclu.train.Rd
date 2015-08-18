\name{Rsomoclu.train}
\alias{Rsomoclu.train}
\title{Train function for somoclu}
\description{
  A function call to somoclu to train the Self Organizing Map.
}
\usage{
Rsomoclu.train(input_data, nEpoch, 
           nSomX, nSomY,
           radius0, radiusN,
           radiusCooling, scale0, scaleN,
           scaleCooling, snapshots,
           kernelType, mapType
           )
}
\arguments{
  \item{input_data}{input data, matrix format}
  \item{nEpoch}{Maximum number of epochs}
  \item{nSomX}{Number of columns in map (size of SOM in direction x)}
  \item{nSomY}{Number of rows in map (size of SOM in direction y)}
  \item{radius0}{Start radius (default: half of the map in direction min(x,y))}
  \item{radiusN}{End radius (default: 1)}
  \item{radiusCooling}{Radius cooling strategy: linear or exponential
    (default: linear)}
  \item{scale0}{Starting learning rate (default: 1.0)}
  \item{scaleN}{Finishing learning rate (default: 0.01)}
  \item{scaleCooling}{Learning rate cooling strategy: linear or
    exponential (default: linear)}
  \item{snapshots}{Save interim files (default: 0): 
                            0: Do not save interim files
                            1: Save U-matrix only
                            2: Also save codebook and best matching}
  \item{kernelType}{Kernel type
                            0: Dense CPU
                            1: Dense GPU
                            2: Sparse CPU}
  \item{mapType}{Map type: planar or toroid (default: planar)}
  %% \item{initialCodebookFilename}{Specify an initial codebook for the map}
}
\references{Peter Wittek, Shichao Gao, Ik Soo Lim, Li Zhao (2015). Somoclu: An Efficient Distributed
  Library for Self-Organizing Maps. arXiv:1305.1422.}
\author{Peter Wittek, Shichao Gao}
\examples{
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
res <- Rsomoclu.train(input_data, nEpoch, nSomX, nSomY,
               radius0, radiusN,
               radiusCooling, scale0, scaleN,
               scaleCooling,
               kernelType, mapType)
res$codebook
res$globalBmus
res$uMatrix
}
