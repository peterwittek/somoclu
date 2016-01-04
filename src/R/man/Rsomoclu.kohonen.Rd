\name{Rsomoclu.kohonen}
\alias{Rsomoclu.kohonen}
\title{convert Somoclu train result to kohonen class for plotting}
\description{
  A function call to convert Somoclu train result to kohonen class for plotting.
}
\usage{
Rsomoclu.kohonen(input_data, result, n.hood = NULL, toroidal = FALSE)
}
\arguments{
  \item{input_data}{input data, matrix format}
  \item{result}{The result returned by Rsomoclu.train}
  \item{n.hood}{Same as in koohonen, the shape of the neighbourhood, either "circular" or "square". The latter is the default for rectangular maps, the former for hexagonal maps.}
  \item{toroidal}{if TRUE, the edges of the map are joined. Note that in a hexagonal toroidal map, the number of rows must be even.}
}
\value{
  An object of class \code{kohonen} for plotting.
}
\examples{
library('Rsomoclu')
library('kohonen')
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
kernelType <- 1
mapType <- "planar"
gridType <- "rectangular"
compactSupport <- FALSE
codebook <- NULL
neighborhood <- "gaussian"
res <- Rsomoclu.train(input_data, nEpoch, nSomX, nSomY,
                      radius0, radiusN,
                      radiusCooling, scale0, scaleN,
                      scaleCooling,
                      kernelType, mapType, gridType, compactSupport, neighborhood, codebook)
## Convert to kohonen object for plotting
sommap = Rsomoclu.kohonen(input_data, res)
## Show 'codebook'
plot(sommap, type="codes", main = c("Codes X", "Codes Y"))
## Show 'component planes'
plot(sommap, type = "property", property = sommap$codes[,1],
     main = colnames(sommap$codes)[1])
## Show 'U-Matrix'
plot(sommap, type="dist.neighbours")
}
\seealso{\url{http://www.r-bloggers.com/self-organising-maps-for-customer-segmentation-using-r/}}