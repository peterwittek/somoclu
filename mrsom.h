/**
 * GPU-Accelerated MapReduce-Based Self-Organizing Maps
 *  Copyright (C) 2012 Peter Wittek
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
 
float *loadCodebook(const char *mapFilename, 
                    unsigned int SOM_X, unsigned int SOM_Y, 
                    unsigned int NDIMEN);
int saveCodebook(char* cbFileName, float *codebook, 
                unsigned int SOM_X, unsigned int SOM_Y, unsigned int NDIMEN);
int saveUMat(char* fname, float *codebook, unsigned int nSomX, 
              unsigned int nSomY, unsigned int nDimensions);
void printMatrix(float *A, int nRows, int nCols);
float *readMatrix(const char *inFileName, 
                  unsigned int &nRows, unsigned int &nCols);

extern "C" {
void setDevice(int commRank, int commSize);
void shutdownGpu();
void getBmusOnGpu(int *bmus, float *deviceCodebook, int SOM_X, int SOM_Y, int NDIMEN, int NVECSPERRANK);
void initializeGpu(float *hostData, int height, int width);
void my_abort(int err);
}
