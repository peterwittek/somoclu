#include"somoclu.h"
#include<cmath>

/** Euclidean distance between vec1 and vec2
 * @param vec1
 * @param vec2
 * @param nDimensions
 * @return distance
 */

float get_distance(float* vec1, float* vec2, unsigned int nDimensions) {
    float distance = 0.0f;
    for (unsigned int d = 0; d < nDimensions; ++d) {
        distance += (vec1[d] - vec2[d]) * (vec1[d] - vec2[d]);
    }
    return sqrt(distance);
}


/** Calculate U-matrix
 * @param codebook - the codebook
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */

float *calculateUMatrix(float *uMatrix, float *codebook, unsigned int nSomX,
                        unsigned int nSomY, unsigned int nDimensions,
                        string mapType, string gridType) {
    float min_dist = 1.5f;
#ifdef _OPENMP
    #pragma omp parallel default(shared)
#endif // _OPENMP
    {
#ifdef _OPENMP
        #pragma omp for
#endif // _OPENMP
#ifdef _WIN32
		for (int som_y1 = 0; som_y1 < nSomY; som_y1++) {
			for (int som_x1 = 0; som_x1 < nSomX; som_x1++) {
#else
        for (unsigned int som_y1 = 0; som_y1 < nSomY; som_y1++) {
            for (unsigned int som_x1 = 0; som_x1 < nSomX; som_x1++) {
#endif
                float dist = 0.0f;
                unsigned int nodes_number = 0;
#ifdef _WIN32
				for (int som_y2 = 0; som_y2 < nSomY; ++som_y2) {
					for (int som_x2 = 0; som_x2 < nSomX; ++som_x2) {
#else
                for (unsigned int som_y2 = 0; som_y2 < nSomY; ++som_y2) {
                    for (unsigned int som_x2 = 0; som_x2 < nSomX; ++som_x2) {
#endif
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
                                                 codebook + som_y2 * nSomX * nDimensions + som_x2 * nDimensions,
                                                 nDimensions);
                        }
                    }
                }
                dist /= (float)nodes_number;
                uMatrix[som_y1 * nSomX + som_x1] = dist;
            }
        }
    }
    return uMatrix;
}
