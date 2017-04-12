\name{Rsomoclu.train}
\alias{Rsomoclu.train}
\title{Train function for Somoclu}
\description{
  A function call to Somoclu to train the Self Organizing Map.
}
\usage{
Rsomoclu.train(input_data, nEpoch, nSomX, nSomY,
                     radius0, radiusN,
                     radiusCooling, scale0, scaleN,
                     scaleCooling,
                     kernelType, mapType, gridType, compactSupport,
                     neighborhood, stdCoeff, codebook)
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
  \item{kernelType}{Kernel type
                            0: Dense CPU
                            1: Dense GPU
                            2: Sparse CPU
                            (default: 0)}
  \item{mapType}{Map type: planar or toroid (default: "planar")}
  \item{gridType}{Grid type: square or hexagonal (default: "rectangular")}
  \item{compactSupport}{Compact support for Gaussian neighborhood, (default:TRUE)}
  \item{neighborhood}{Neighborhood function: gaussian or bubble (default: "gaussian")}
  \item{codebook}{initial codebook, (default:NULL)}
  \item{stdCoeff}{The coefficient in the Gaussian neighborhood function exp(-||x-y||^2/(2*(coeff*radius)^2)), (default:0.5)}
  %% \item{initialCodebookFilename}{Specify an initial codebook for the map}
}
\value{
  a list including elements
  \item{codebook}{the codebook}
  \item{globalBmus}{global Best Matching Unit matrix}
  \item{uMatrix}{uMatrix}
}
\references{Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao (2015). Somoclu: An Efficient Parallel Library for Self-Organizing Maps. arXiv:1305.1422.}
\author{Peter Wittek, Shichao Gao}
\examples{
library('Rsomoclu')
data("rgbs", package = "Rsomoclu")
input_data <- rgbs
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
stdCoeff <- 0.5
res <- Rsomoclu.train(input_data, nEpoch, nSomX, nSomY,
                      radius0, radiusN,
                      radiusCooling, scale0, scaleN,
                      scaleCooling,
                      kernelType, mapType, gridType, compactSupport, neighborhood,
                      stdCoeff, codebook)
res$codebook
res$globalBmus
res$uMatrix
library('kohonen')
sommap = Rsomoclu.kohonen(input_data, res)
}
