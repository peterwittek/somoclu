#include "somoclu.h"
#include <cmath>

/** Calculate U-matrix
 * @param codebook - the codebook
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */

void calculateUMatrix(som map) {
    float min_dist = 1.5f;
    #pragma omp parallel default(shared)
    {
      #pragma omp for
      for (omp_iter_t som_y1 = 0; som_y1 < map.nSomY; som_y1++) {
      for (omp_iter_t som_x1 = 0; som_x1 < map.nSomX; som_x1++) {
                float dist = 0.0f;
                unsigned int nodes_number = 0;
				for (omp_iter_t som_y2 = 0; som_y2 < map.nSomY; ++som_y2) {
					for (omp_iter_t som_x2 = 0; som_x2 < map.nSomX; ++som_x2) {
                        if (som_x1 == som_x2 && som_y1 == som_y2) continue;
                        float tmp = 0.0f;
                        if (map.gridType == "rectangular") {
                            if (map.mapType == "planar") {
                                tmp = euclideanDistanceOnPlanarMap(som_x1, som_y1, som_x2, som_y2);
                            }
                            else if (map.mapType == "toroid") {
                                tmp = euclideanDistanceOnToroidMap(som_x1, som_y1, som_x2, som_y2, map.nSomX, map.nSomY);
                            }
                        }
                        else {
                            if (map.mapType == "planar") {
                                tmp = euclideanDistanceOnHexagonalPlanarMap(som_x1, som_y1, som_x2, som_y2);
                            }
                            else if (map.mapType == "toroid") {
                                tmp = euclideanDistanceOnHexagonalToroidMap(som_x1, som_y1, som_x2, som_y2, map.nSomX, map.nSomY);
                            }
                        }
                        if (tmp <= min_dist) {
                            ++nodes_number;
                            dist += map.get_distance(map.codebook +
                                                     som_y1 * map.nSomX * map.nDimensions +
                                                     som_x1 * map.nDimensions,
                                                     map.codebook +
                                                     som_y2 * map.nSomX * map.nDimensions +
                                                     som_x2 * map.nDimensions);
                        }
                    }
                }
                dist /= (float)nodes_number;
                map.uMatrix[som_y1 * map.nSomX + som_x1] = dist;
            }
        }
    }
}
