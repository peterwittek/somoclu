#include "somoclu.h"
#include <cmath>

/** Calculate U-matrix
 * @param codebook - the codebook
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */

float *calculateUMatrix(float *uMatrix, float *codebook, unsigned int nSomX,
                        unsigned int nSomY, unsigned int nDimensions,
                        string mapType, string gridType,
                        const Distance& get_distance) {
    float min_dist = 1.5f;
#ifdef _OPENMP
#pragma omp parallel default(shared)
    {
#pragma omp for
#endif // _OPENMP
      for (omp_iter_t som_y1 = 0; som_y1 < nSomY; som_y1++) {
	for (omp_iter_t som_x1 = 0; som_x1 < nSomX; som_x1++) {
                float dist = 0.0f;
                unsigned int nodes_number = 0;
				for (omp_iter_t som_y2 = 0; som_y2 < nSomY; ++som_y2) {
					for (omp_iter_t som_x2 = 0; som_x2 < nSomX; ++som_x2) {
                        if (som_x1 == som_x2 && som_y1 == som_y2) continue;
                        float tmp = 0.0f;
                        if (gridType == "rectangular") {
                            if (mapType == "planar") {
                                tmp = euclideanDistanceOnPlanarMap(som_x1, som_y1, som_x2, som_y2);
                            }
                            else if (mapType == "toroid") {
                                tmp = euclideanDistanceOnToroidMap(som_x1, som_y1, som_x2, som_y2, nSomX, nSomY);
                            }
                        }
                        else {
                            if (mapType == "planar") {
                                tmp = euclideanDistanceOnHexagonalPlanarMap(som_x1, som_y1, som_x2, som_y2);
                            }
                            else if (mapType == "toroid") {
                                tmp = euclideanDistanceOnHexagonalToroidMap(som_x1, som_y1, som_x2, som_y2, nSomX, nSomY);
                            }
                        }
                        if (tmp <= min_dist) {
                            ++nodes_number;
                            dist += get_distance(codebook + som_y1 * nSomX * nDimensions + som_x1 * nDimensions,
                                                 codebook + som_y2 * nSomX * nDimensions + som_x2 * nDimensions);
                        }
                    }
                }
                dist /= (float)nodes_number;
                uMatrix[som_y1 * nSomX + som_x1] = dist;
            }
        }
#ifdef _OPENMP
    }
#endif
    return uMatrix;
}
