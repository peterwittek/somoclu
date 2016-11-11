/**
 * Self-Organizing Maps on a cluster
 *  Copyright (C) 2013 Peter Wittek
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
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#include "Windows/unistd.h"
#include "Windows/getopt.h"
#else
#include <unistd.h>
#include <getopt.h>
#endif

#include "somoclu.h"

using namespace std;

/// For synchronized timing
#ifndef MPI_WTIME_IS_GLOBAL
#define MPI_WTIME_IS_GLOBAL 1
#endif

// Default parameters
#define N_EPOCH 10
#define N_SOM_X 50
#define N_SOM_Y 50
#define KERNEL_TYPE 0
#define SNAPSHOTS 0

void printUsage() {
    cout << "Usage:\n" \
         "     [mpirun -np NPROC] somoclu [OPTIONs] INPUT_FILE OUTPUT_PREFIX\n" \
         "Arguments:\n" \
         "     -c FILENAME           Specify an initial codebook for the map.\n" \
         "     -d NUMBER             Coefficient in the Gaussian neighborhood function exp(-||x-y||^2/(2*(coeff*radius)^2)) (default: 0.5)\n" \
         "     -e NUMBER             Maximum number of epochs (default: " << N_EPOCH << ")\n" \
         "     -g TYPE               Grid type: rectangular or hexagonal (default: rectangular)\n"\
         "     -h, --help            This help text\n" \
         "     -k NUMBER             Kernel type (default: " << KERNEL_TYPE << "): \n" \
         "                              0: Dense CPU\n" \
         "                              1: Dense GPU\n" \
         "                              2: Sparse CPU\n" \
         "     -l NUMBER             Starting learning rate (default: 0.1)\n" \
         "     -L NUMBER             Finishing learning rate (default: 0.01)\n" \
         "     -m TYPE               Map type: planar or toroid (default: planar) \n" \
         "     -n FUNCTION           Neighborhood function (bubble or gaussian, default: gaussian)\n"\
         "     -p NUMBER             Compact support for Gaussian neighborhood (0: false, 1: true, default: 1)\n"\
         "     -r NUMBER             Start radius (default: half of the map in direction min(x,y))\n" \
         "     -R NUMBER             End radius (default: 1)\n" \
         "     -s NUMBER             Save interim files (default: 0):\n" \
         "                              0: Do not save interim files\n" \
         "                              1: Save U-matrix only\n" \
         "                              2: Also save codebook and best matching neurons\n" \
         "     -t STRATEGY           Radius cooling strategy: linear or exponential (default: linear)\n" \
         "     -T STRATEGY           Learning rate cooling strategy: linear or exponential (default: linear)\n" \
         "     -x, --columns NUMBER  Number of columns in map (size of SOM in direction x) (default: " << N_SOM_X << ")\n" \
         "     -y, --rows NUMBER     Number of rows in map (size of SOM in direction y) (default: " << N_SOM_Y << ")\n" \
         "Examples:\n" \
         "     somoclu data/rgbs.txt data/rgbs\n"
         "     mpirun -np 4 somoclu -k 0 -x 20 -y 20 data/rgbs.txt data/rgbs\n";
}

void processCommandLine(int argc, char** argv, string *inFilename,
                        string* outPrefix, unsigned int *nEpoch,
                        float *radius0, float *radiusN,
                        string *radiusCooling,
                        float *scale0, float *scaleN,
                        string *scaleCooling,
                        unsigned int *nSomX, unsigned int *nSomY,
                        unsigned int *kernelType, string *mapType,
                        unsigned int *snapshots,
                        string *gridType, unsigned int *compactSupport,
                        unsigned int *gaussian, float *std_coeff,
                        string *initialCodebookFilename) {

    // Setting default values
    *nEpoch = N_EPOCH;
    *nSomX = N_SOM_X;
    *nSomY = N_SOM_Y;
    *kernelType = KERNEL_TYPE;
    *snapshots = SNAPSHOTS;
    *mapType = "planar";
    *radius0 = 0;
    *radiusN = 0;
    *radiusCooling = "linear";
    *scale0 = 0.0;
    *scaleN = 0.01;
    *scaleCooling = "linear";
    *gridType = "rectangular";
    *compactSupport = 1;
    *gaussian = 1;
    *std_coeff = 0.5;
    string neighborhood_function = "gaussian";
    static struct option long_options[] = {
        {"help",  no_argument,       0,  'h'},
        {"rows",  required_argument, 0, 'y'},
        {"columns",    required_argument, 0, 'x'},
        {0, 0, 0, 0}
    };
    int c;
    extern int optind, optopt;
    int option_index = 0;
    while ((c = getopt_long (argc, argv, "hx:y:d:e:g:k:l:m:n:p:r:s:t:c:L:R:T:",
                             long_options, &option_index)) != -1) {
        switch (c) {
        case 'c':
            *initialCodebookFilename = optarg;
            break;
        case 'd':
            *std_coeff = atof(optarg);
            if (*std_coeff <= 0) {
                my_abort("The argument of option -l should be a positive float.");
            }
            break;
        case 'e':
            *nEpoch = atoi(optarg);
            if (*nEpoch <= 0) {
                my_abort("The argument of option -e should be a positive integer.");
            }
            break;
        case 'h':
            printUsage();
            exit(0);
            break;
        case 'k':
            *kernelType = atoi(optarg);
            if (*kernelType > SPARSE_CPU) {
                my_abort("The argument of option -k should be a valid kernel.");
            }
            break;
        case 'n':
            neighborhood_function = optarg;
            if (neighborhood_function == "bubble") {
                *gaussian = 0;
            } else if (neighborhood_function == "gaussian") {
                *gaussian = 1;
            } else {
                my_abort("The argument of option -n should be either bubble or Gaussian.");
            }
            break;
        case 'p':
            *compactSupport = atoi(optarg);
            if (*compactSupport != 0 && *compactSupport != 1) {
                my_abort("The argument of option -g should be either 0 (false) or 1 (true).");
            }
            break;
        case 'm':
            *mapType = optarg;
            if (*mapType != "planar" && *mapType != "toroid") {
                my_abort("The argument of option -m should be either planar or toroid.");
            }
            break;
        case 'g':
            *gridType = optarg;
            if (*gridType != "rectangular" && *gridType != "hexagonal") {
                my_abort("The argument of option -h should be either rectangular or hexagonal.");
            }
            break;
        case 'r':
            *radius0 = atof(optarg);
            if (*radius0 <= 0) {
                my_abort("The argument of option -r should be a positive integer.");
            }
            break;
        case 'R':
            *radiusN = atof(optarg);
            if (*radiusN <= 0) {
                my_abort("The argument of option -R should be a positive integer.");
            }
            break;
        case 't':
            *radiusCooling = optarg;
            if (*radiusCooling != "linear" && *radiusCooling != "exponential") {
                my_abort("The argument of option -t should be linear or exponential.");
            }
            break;
        case 'l':
            *scale0 = atof(optarg);
            if (*scale0 <= 0) {
                my_abort("The argument of option -l should be a positive float.");
            }
            break;
        case 'L':
            *scaleN = atof(optarg);
            if (*scaleN <= 0) {
                my_abort("The argument of option -L should be a positive float.");
            }
            break;
        case 'T':
            *scaleCooling = optarg;
            if (*scaleCooling != "linear" && *scaleCooling != "exponential") {
                my_abort("The argument of option -T should be linear or exponential.");
            }
            break;
        case 's':
            *snapshots = atoi(optarg);
            if (*snapshots > 2) {
                my_abort("The argument of option -s should be 0, 1, or 2.");
            }

            break;
        case 'x':
            *nSomX = atoi(optarg);
            if (*nSomX <= 0) {
                my_abort("The argument of option -x should be a positive integer.");
            }
            break;
        case 'y':
            *nSomY = atoi(optarg);
            if (*nSomY <= 0) {
                my_abort("The argument of option -y should be a positive integer.");
            }
            break;
        case '?':
            if (optopt == 'e' || optopt == 'k' || optopt == 's' ||
                    optopt == 'x'    || optopt == 'y') {
                stringstream sstm;
                sstm << "Option -" <<  optopt << " requires an argument.";
                my_abort(sstm.str());
            }
            else if (isprint (optopt)) {
                stringstream sstm;
                sstm << "Unknown option -" << optopt;
                my_abort(sstm.str());
            }
            else {
                stringstream sstm;
                sstm << "Unknown option character `\\x" << optopt << "'";
                my_abort(sstm.str());
            }
        default:
            abort ();
        }
    }
    if (argc - optind != 2) {
        my_abort("Incorrect number of mandatory parameters");
    }
    *inFilename = argv[optind++];
    *outPrefix = argv[optind++];
}

/* -------------------------------------------------------------------------- */
int main(int argc, char** argv)
/* -------------------------------------------------------------------------- */
{
    int rank = 0;
    int nProcs = 1;

#ifdef HAVE_MPI
    ///
    /// MPI init
    ///
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    unsigned int nEpoch = 0;
    unsigned int nSomX = 0;
    unsigned int nSomY = 0;
    unsigned int kernelType = 0;
    string mapType;
    string gridType;
    unsigned int compactSupport;
    unsigned int gaussian;
    float radius0 = 0;
    float radiusN = 0;
    string radiusCooling;
    float scale0 = 0.0;
    float scaleN = 0.0;
    string scaleCooling;
    unsigned int snapshots = 0;
    string inFilename;
    string initialCodebookFilename;
    string outPrefix;
    float std_coeff = 0.0;
    if (rank == 0) {
        processCommandLine(argc, argv, &inFilename, &outPrefix,
                           &nEpoch, &radius0, &radiusN, &radiusCooling,
                           &scale0, &scaleN, &scaleCooling,
                           &nSomX, &nSomY,
                           &kernelType, &mapType, &snapshots,
                           &gridType, &compactSupport, &gaussian, &std_coeff,
                           &initialCodebookFilename);
#ifndef CUDA
        if (kernelType == DENSE_GPU) {
            my_abort("Somoclu was compile without GPU support!");
        }
#endif
    }
#ifdef HAVE_MPI
    MPI_Bcast(&nEpoch, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&radius0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nSomX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nSomY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kernelType, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&compactSupport, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char *inFilenameCStr = new char[255];
    if (rank == 0) {
        strcpy(inFilenameCStr, inFilename.c_str());
    }
    MPI_Bcast(inFilenameCStr, 255, MPI_CHAR, 0, MPI_COMM_WORLD);
    inFilename = inFilenameCStr;

    char *mapTypeCStr = new char[255];
    if (rank == 0) {
        strcpy(mapTypeCStr, mapType.c_str());
    }
    MPI_Bcast(mapTypeCStr, 255, MPI_CHAR, 0, MPI_COMM_WORLD);
    mapType = mapTypeCStr;

    char *gridTypeCStr = new char[255];
    if (rank == 0) {
        strcpy(gridTypeCStr, gridType.c_str());
    }
    MPI_Bcast(gridTypeCStr, 255, MPI_CHAR, 0, MPI_COMM_WORLD);
    gridType = gridTypeCStr;


    double profile_time = MPI_Wtime();
#endif

    float * dataRoot = NULL;
    unsigned int nDimensions = 0;
    unsigned int nVectors = 0;
    if(rank == 0 ) {
        if (kernelType == DENSE_CPU || kernelType == DENSE_GPU) {
            dataRoot = readMatrix(inFilename, nVectors, nDimensions);
        }
        else {
            readSparseMatrixDimensions(inFilename, nVectors, nDimensions);
        }
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&nVectors, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nDimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    unsigned int nVectorsPerRank = ceil(nVectors / (1.0 * nProcs));

    // Allocate a buffer on each node
    float* data = NULL;
    svm_node **sparseData;
    sparseData = NULL;

    if (kernelType == DENSE_CPU || kernelType == DENSE_GPU) {
#ifdef HAVE_MPI
        // Dispatch a portion of the input data to each node
        data = new float[nVectorsPerRank * nDimensions];
        MPI_Scatter(dataRoot, nVectorsPerRank * nDimensions, MPI_FLOAT,
                    data, nVectorsPerRank * nDimensions, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
#else
        data = dataRoot;
#endif
    }
    else {
        int currentRankProcessed = 0;
        while (currentRankProcessed < nProcs) {
            if (rank == currentRankProcessed) {
                sparseData = readSparseMatrixChunk(inFilename, nVectors, nVectorsPerRank,
                                                   rank * nVectorsPerRank);
            }
            currentRankProcessed++;
#ifdef HAVE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
    }

    if(rank == 0) {
        // No need for root data any more if compiled with MPI
#ifdef HAVE_MPI
        if (kernelType == DENSE_CPU || kernelType == DENSE_GPU) {
            delete [] dataRoot;
        }
#endif
        cout << "nVectors: " << nVectors << " ";
        cout << "nVectorsPerRank: " << nVectorsPerRank << " ";
        cout << "nDimensions: " << nDimensions << " ";
        cout << endl;
    }

    ///
    /// Codebook
    ///
    float *codebook = new float[nSomY * nSomX * nDimensions];
    int *globalBmus = NULL;
    float *uMatrix = NULL;
    if (rank == 0) {
        globalBmus = new int[nVectorsPerRank * int(ceil(nVectors / (double)nVectorsPerRank)) * 2];
        uMatrix = new float[nSomX * nSomY];
        if (initialCodebookFilename.empty()) {
            initializeCodebook(0, codebook, nSomX, nSomY, nDimensions);
        }
        else {
            unsigned int nSomXY = 0;
            unsigned int tmpNDimensions = 0;
            delete [] codebook;
            codebook = readMatrix(initialCodebookFilename, nSomXY, tmpNDimensions);
            if (tmpNDimensions != nDimensions) {
                my_abort("Dimension of initial codebook does not match data!");
            }
            else if (nSomXY / nSomY != nSomX) {
                my_abort("Dimension of initial codebook does not match specified SOM grid!");
            }
            cout << "Read initial codebook: " << initialCodebookFilename << "\n";
        }
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    double training_time = MPI_Wtime();
#endif
    // TRAINING
    train(rank, data, sparseData, codebook, globalBmus, uMatrix, nSomX, nSomY,
          nDimensions, nVectors, nVectorsPerRank,
          nEpoch, radius0, radiusN, radiusCooling,
          scale0, scaleN, scaleCooling,
          kernelType, mapType,
          gridType, compactSupport == 1, gaussian == 1, std_coeff
#ifdef CLI
	, outPrefix, snapshots);
#else
	);
#endif

#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    training_time = MPI_Wtime() - training_time;
    if (rank == 0) {
        cerr << "Total training Time: " << training_time << endl;
    }
#endif
    ///
    /// Save SOM map and u-mat
    ///
    if (rank == 0) {
        ///
        /// Save U-mat
        ///
        calculateUMatrix(uMatrix, codebook, nSomX, nSomY, nDimensions, mapType,
                         gridType);
        int ret =  saveUMatrix(outPrefix + string(".umx"), uMatrix, nSomX, nSomY);
        if (ret < 0)
            cout << "    Failed to save u-matrix. !" << endl;
        else {
            cout << "    Done!" << endl;
        }
        saveBmus(outPrefix + string(".bm"), globalBmus, nSomX, nSomY, nVectors);
        ///
        /// Save codebook
        ///
        saveCodebook(outPrefix + string(".wts"), codebook, nSomX, nSomY, nDimensions);
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (kernelType == DENSE_CPU || kernelType == DENSE_GPU) {
        delete [] data;
    }
    else {
        delete [] sparseData;
    }
    delete [] codebook;
    delete [] globalBmus;
    delete [] uMatrix;
#ifdef HAVE_MPI
    profile_time = MPI_Wtime() - profile_time;
    if (rank == 0) {
        cerr << "Total Execution Time: " << profile_time << endl;
    }
#endif
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}

/// EOF
