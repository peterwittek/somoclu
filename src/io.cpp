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

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>

#include "somoclu.h"
#include "io.h"

void cli_abort(string err) {
#ifdef HAVE_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cerr << "Error: " << err << endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    cerr << "Error: " << err << endl;
    exit(1);
#endif  // HAVE_MPI
}

void Snapshot::write(unsigned int currentEpoch, som map) {
    calculateUMatrix(map);
    stringstream sstm;
    sstm << outPrefix << "." << currentEpoch;
    saveUMatrix(sstm.str() + string(".umx"), map);
    if (snapshots == 2) {
        saveBmus(sstm.str() + string(".bm"), map);
        saveCodebook(sstm.str() + string(".wts"), map);
    }
}

/** Save a SOM map.codebook
 * @param cbFileName - name of the file to save
 * @param map.codebook - the map.codebook to save
 * @param map.nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param map.nDimensions - dimensions of a data instance
 */
int saveCodebook(string cbFilename, som map) {
    FILE* file = fopen(cbFilename.c_str(), "wt");
    cout << "    Saving Codebook " << cbFilename << endl;
    fprintf(file, "%%%d %d\n", map.nSomY, map.nSomX);
    fprintf(file, "%%%d\n", map.nDimensions);
    if (file != 0) {
        for (unsigned int som_y = 0; som_y < map.nSomY; som_y++) {
            for (unsigned int som_x = 0; som_x < map.nSomX; som_x++) {
                for (unsigned int d = 0; d < map.nDimensions; d++) {
                    fprintf(file, "%0.10f ", map.codebook[som_y * map.nSomX * map.nDimensions + som_x * map.nDimensions + d]);
                }
                fprintf(file, "\n");
            }
        }
        fclose(file);
        return 0;
    }
    else {
        return 1;
    }
}

/** Save best matching units
 * @param filename - name of the file to save
 * @param bmus - the best matching units to save
 * @param map.nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param map.nVectors - the number of vectors
 */
int saveBmus(string filename, som map) {
    FILE* file = fopen(filename.c_str(), "wt");
    cout << "    Saving best matching units " << filename << endl;
    fprintf(file, "%%%d %d\n", map.nSomY, map.nSomX);
    fprintf(file, "%%%d\n", map.nVectors);
    if (file != 0) {
        for (unsigned int i = 0; i < map.nVectors; ++i) {
            // ESOM Tools swaps x and y!
            fprintf(file, "%d %d %d\n", i, map.bmus[2 * i + 1], map.bmus[2 * i]);
        }
        fclose(file);
        return 0;
    }
    else {
        return 1;
    }
}


/** Save u-matrix
 * @param fname
 * @param map.codebook - the map.codebook to save
 * @param map.nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param map.nDimensions - dimensions of a data instance
 */

int saveUMatrix(string fname, som map) {

    FILE* fp = fopen(fname.c_str(), "wt");
    fprintf(fp, "%%");
    fprintf(fp, "%d %d", map.nSomY, map.nSomX);
    fprintf(fp, "\n");
    if (fp != 0) {
        for (unsigned int som_y1 = 0; som_y1 < map.nSomY; som_y1++) {
            for (unsigned int som_x1 = 0; som_x1 < map.nSomX; som_x1++) {
                fprintf(fp, " %f", map.uMatrix[som_y1 * map.nSomX + som_x1]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        return 0;
    }
    else
        return -2;
}

void getMatrixDimensions(string inFilename, unsigned int &nRows, unsigned int &nColumns) {
    ifstream file;
    file.open(inFilename.c_str());
    if (file.is_open()) {
        string line;
        float tmp;
        while(getline(file, line)) {
            if (line.substr(0, 1) == "#") {
                continue;
            }
            std::istringstream iss(line);
            if (nRows == 0) {
                while (iss >> tmp) {
                    nColumns++;
                }
            }
            nRows++;
        }
        file.close();
    }
    else {
        cli_abort("Input file could not be opened!");
    }
}

unsigned int *readLrnHeader(string inFilename, unsigned int &nRows, unsigned int &nColumns) {
    ifstream file;
    file.open(inFilename.c_str());
    string line;
    unsigned int currentColumn = 0;
    unsigned int nAllColumns = 0;
    unsigned int *columnMap = NULL;
    while(getline(file, line)) {
        if (line.substr(0, 1) == "#") {
            continue;
        }
        if (line.substr(0, 1) == "%") {
            std::istringstream iss(line.substr(1, line.length()));
            if (nRows == 0) {
                iss >> nRows;
            }
            else if (nAllColumns == 0) {
                iss >> nAllColumns;
            }
            else if (columnMap == NULL) {
                columnMap = new unsigned int[nAllColumns];
                unsigned int itmp = 0;
                currentColumn = 0;
                while (iss >> itmp) {
                    columnMap[currentColumn++] = itmp;
                    if (itmp == 1) {
                        ++nColumns;
                    }
                }
            }
            else {
                break;
            }
        }
    }
    file.close();
    return columnMap;
}

unsigned int *readWtsHeader(string inFilename, unsigned int &nRows, unsigned int &nColumns) {
    ifstream file;
    file.open(inFilename.c_str());
    string line;
    while(getline(file, line)) {
        if (line.substr(0, 1) == "#") {
            continue;
        }
        if (line.substr(0, 1) == "%") {
            std::istringstream iss(line.substr(1, line.length()));
            if (nRows == 0) {
                iss >> nRows;
                unsigned int nSomY = 0;
                iss >> nSomY;
                nRows = nRows * nSomY;
            }
            else if (nColumns == 0) {
                iss >> nColumns;
            }
            else {
                break;
            }
        }
    }
    file.close();
    unsigned int *columnMap = new unsigned int[nColumns];
    for (unsigned int i = 0; i < nColumns; ++i) {
        columnMap[i] = 1;
    }
    return columnMap;
}

/** Reads a matrix
 * @param inFilename
 * @param nRows - returns the number of rows
 * @param nColumns - returns the number of columns
 * @return the matrix
 */
float *readMatrix(string inFilename, unsigned int &nRows, unsigned int &nColumns) {
    float *data = NULL;
    unsigned int *columnMap = NULL;
    if (inFilename.compare(inFilename.size() - 3, 3, "lrn") == 0) {
        columnMap = readLrnHeader(inFilename, nRows, nColumns);
    }
    else if (inFilename.compare(inFilename.size() - 3, 3, "wts") == 0) {
        columnMap = readWtsHeader(inFilename, nRows, nColumns);
    }
    else {
        getMatrixDimensions(inFilename, nRows, nColumns);
        columnMap = new unsigned int[nColumns];
        for (unsigned int i = 0; i < nColumns; ++i) {
            columnMap[i] = 1;
        }
    }
    ifstream file;
    file.open(inFilename.c_str());
    string line;
    float tmp;
    unsigned int j = 0;
    unsigned int currentColumn = 0;

    while(getline(file, line)) {
        if ( (line.substr(0, 1) == "#") | (line.substr(0, 1) == "%") ) {
            continue;
        }
        if (data == NULL) {
            data = new float[nRows * nColumns];
        }
        std::istringstream iss(line);
        currentColumn = 0;
        while (iss >> tmp) {
            if (columnMap[currentColumn++] != 1) {
                continue;
            }
            data[j++] = tmp;
        }
    }
    file.close();
    delete [] columnMap;
    return data;
}

void readSparseMatrixDimensions(string filename, unsigned int &nRows,
                                unsigned int &nColumns, bool & zerobased) {
    ifstream file;
    file.open(filename.c_str());
    if (file.is_open()) {
        string line;
        int max_index = -1;
        zerobased = false;
        while(getline(file, line)) {
            if (line.substr(0, 1) == "#") {
                continue;
            }
            stringstream linestream(line);
            string value;
            int dummy_index;
            while(getline(linestream, value, ' ')) {
                int separator = value.find(":");
                if (separator == -1) // it's a label ?
                    continue;
                istringstream myStream(value.substr(0, separator));
                myStream >> dummy_index;
                if(dummy_index > max_index) {
                    max_index = dummy_index;
                }
                if (dummy_index == 0) {
                    zerobased = true;
                }
            }
            ++nRows;
        }
        nColumns = max_index;
        if (zerobased)
            ++nColumns;
        file.close();
    }
    else {
        cli_abort("Input file could not be opened!");
    }
}

svm_node** readSparseMatrixChunk(string filename, unsigned int nRows,
                                 unsigned int nRowsToRead,
                                 unsigned int rowOffset,
                                 unsigned int colOffset) {
    ifstream file;
    file.open(filename.c_str());
    string line;
    for (unsigned int i = 0; i < rowOffset; i++) {
        getline(file, line);
    }
    if (rowOffset + nRowsToRead >= nRows) {
        nRowsToRead = nRows - rowOffset;
    }
    svm_node **x_matrix = new svm_node *[nRowsToRead];
    for(unsigned int i = 0; i < nRowsToRead; ) {
        getline(file, line);
        if (line[0] == '#') {
            continue;
        }
        stringstream tmplinestream(line);
        string value;
        int elements = 0;
        bool label = false;
        // count values
        while(getline(tmplinestream, value, ' ')) {
            // skip labels
            if (elements==0 && label==false && value.find(':') == string::npos)
            {
                label=true;
                continue;
            }
            elements++;
        }
        x_matrix[i] = new svm_node[elements+1];  // To account for the closing dummy node in the row
        stringstream linestream(line);
        if (label)
            linestream >> value; // skip the label
        int j = 0;
        while (1) {
            char sep;
            int idx;
            float val;
            linestream >> idx >> sep >> val;
            if (linestream.fail())
                break;
            //assert(sep==':');
            x_matrix[i][j].index = idx-colOffset;
            x_matrix[i][j].value = val;
            j++;
        }
        //assert(j==elements);
        x_matrix[i][j].index = -1;
        i++;
    }
    file.close();
    return x_matrix;
}
