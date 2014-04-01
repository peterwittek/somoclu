
#include"somoclu.h"
#include<cmath>

/** Get weight vector from a codebook using x, y index
 * @param codebook - the codebook to save
 * @param som_y - y coordinate of a node in the map
 * @param som_x - x coordinate of a node in the map
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 * @return the weight vector
 */

float* get_wvec(float *codebook, unsigned int som_y, unsigned int som_x,
                unsigned int nSomX, unsigned int nDimensions)
{
    float* wvec = new float[nDimensions];
    for (unsigned int d = 0; d < nDimensions; d++)
        wvec[d] = codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d]; /// CAUTION: (y,x) order
    return wvec;
}

/** Euclidean distance between vec1 and vec2
 * @param vec1
 * @param vec2
 * @param nDimensions
 * @return distance
 */

float get_distance(const float* vec1, const float* vec2,
                   unsigned int nDimensions) {
    float distance = 0.0f;
    float x1 = 0.0f;
    float x2 = 0.0f;
    for (unsigned int d = 0; d < nDimensions; d++) {
        x1 = std::min(vec1[d], vec2[d]);
        x2 = std::max(vec1[d], vec2[d]);
        distance += std::abs(x1-x2)*std::abs(x1-x2);
    }
    return sqrt(distance);
}


/** Calculate U-matrix
 * @param codebook - the codebook
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */

float *calculateUMatrix(float *codebook, unsigned int nSomX,
             unsigned int nSomY, unsigned int nDimensions, string mapType)
{
    float *uMatrix = new float[nSomX*nSomY];
    float min_dist = 1.5f;
    for (unsigned int som_y1 = 0; som_y1 < nSomY; som_y1++) {
        for (unsigned int som_x1 = 0; som_x1 < nSomX; som_x1++) {
            float dist = 0.0f;
            unsigned int nodes_number = 0;

            for (unsigned int som_y2 = 0; som_y2 < nSomY; som_y2++) {
                for (unsigned int som_x2 = 0; som_x2 < nSomX; som_x2++) {

                    if (som_x1 == som_x2 && som_y1 == som_y2) continue;
                    float tmp = 0.0f;
                    if (mapType == "planar") {
                        tmp = euclideanDistanceOnPlanarMap(som_x1, som_y1, som_x2, som_y2);
                    } else if (mapType == "toroid") {
                        tmp = euclideanDistanceOnToroidMap(som_x1, som_y1, som_x2, som_y2, nSomX, nSomY);
                    }
                    if (tmp <= min_dist) {
                        nodes_number++;
                        float* vec1 = get_wvec(codebook, som_y1, som_x1, nSomX, nDimensions);
                        float* vec2 = get_wvec(codebook, som_y2, som_x2, nSomX, nDimensions);
                        dist += get_distance(vec1, vec2, nDimensions);
                        delete [] vec1;
                        delete [] vec2;
                    }
                }
            }
            dist /= (float)nodes_number;
            uMatrix[som_y1*nSomX+som_x1] = dist;
        }
    }
    return uMatrix;
}
