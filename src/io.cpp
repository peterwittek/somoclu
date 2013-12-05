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
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>

#include "somoclu.h"

using namespace std;

/** Save a SOM codebook
 * @param cbFileName - name of the file to save
 * @param codebook - the codebook to save
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */
int saveCodebook(char* cbFileName, float *codebook, unsigned int nSomX, unsigned int nSomY, unsigned int nDimensions)
{
	char temp[80];
	cout << "    Codebook file = " << cbFileName << endl;       
	ofstream mapFile(cbFileName);
	cout << "    Saving Codebook..." << endl;
  mapFile << "%" << nSomY << " " << nSomX << endl;
  mapFile << "%" << nDimensions << endl;
	if (mapFile.is_open()) {
		for (unsigned int som_y = 0; som_y < nSomY; som_y++) { 
			for (unsigned int som_x = 0; som_x < nSomX; som_x++) { 
				for (unsigned int d = 0; d < nDimensions; d++) {
					sprintf(temp, "%0.10f", codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d]);
					mapFile << temp << " ";
				}
        mapFile << endl;
			}
		}
		mapFile.close();
		return 0;
	}
	else 
	{
		return 1;
	}
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
                unsigned int nSomX, unsigned int nSomY, unsigned int nDimensions)
{
  float* wvec = new float[nDimensions];
  for (unsigned int d = 0; d < nDimensions; d++)
      wvec[d] = codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d]; /// CAUTION: (y,x) order
  return wvec;
}

/** Save u-matrix
 * @param fname
 * @param codebook - the codebook to save
 * @param nSomX - dimensions of SOM map in the x direction
 * @param nSomY - dimensions of SOM map in the y direction
 * @param nDimensions - dimensions of a data instance
 */
 
int saveUMat(char* fname, float *codebook, unsigned int nSomX, 
              unsigned int nSomY, unsigned int nDimensions, unsigned int mapType)
{

  float min_dist = 1.5f;
  FILE* fp = fopen(fname, "wt");
  fprintf(fp, "%%");
  fprintf(fp, "%d %d", nSomY, nSomX);
  fprintf(fp, "\n");
  if (fp != 0) {
    for (unsigned int som_y1 = 0; som_y1 < nSomY; som_y1++) {
      for (unsigned int som_x1 = 0; som_x1 < nSomX; som_x1++) {
        float dist = 0.0f;
        unsigned int nodes_number = 0;
           
        for (unsigned int som_y2 = 0; som_y2 < nSomY; som_y2++) {   
            for (unsigned int som_x2 = 0; som_x2 < nSomX; som_x2++) {

              if (som_x1 == som_x2 && som_y1 == som_y2) continue;
              float tmp = 0.0f;
              if (mapType == PLANAR) {
                  tmp = euclideanDistanceOnPlanarMap(som_x1, som_y1, som_x2, som_y2);
              } else if (mapType == TOROID) {
                  tmp = euclideanDistanceOnToroidMap(som_x1, som_y1, som_x2, som_y2, nSomX, nSomY);
              }
              if (tmp <= min_dist) {
                  nodes_number++;
                  float* vec1 = get_wvec(codebook, som_y1, som_x1, nSomX, nSomY, nDimensions);
                  float* vec2 = get_wvec(codebook, som_y2, som_x2, nSomX, nSomY, nDimensions);
                  dist += get_distance(vec1, vec2, nDimensions);
                  delete [] vec1;
                  delete [] vec2;
              }
            }
        }
        dist /= (float)nodes_number;
        fprintf(fp, " %f", dist);
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
  }
  else
    return -2;
}

bool isLrnFile(const char *inFileName) {
  unsigned int size = 0;
  while(inFileName[size++]!='\0') {}
  unsigned int testResult = strcmp("lrn\0", inFileName + size - 4);
  if (testResult == 0) {
    return true;
  } else {
    return false;
  }
}

/** Reads an ESOM lrn file
 * @param inFileNae
 * @param nRows - returns the number of rows
 * @param nColumns - returns the number of columns
 * @return the matrix
 */

float *readLrnFile(const char *inFileName, unsigned int &nRows, unsigned int &nColumns)
{
  unsigned int nAllColumns = 0;
  float *data = NULL;
  ifstream file;
  file.open(inFileName);
  string line;
  float tmp;  
  unsigned int j = 0;
  unsigned int currentColumn = 0;
  unsigned int *columnMap = NULL;
  while(getline(file,line)){
    if (line.substr(0,1) == "#"){
      continue;
    }
    if (line.substr(0,1) == "%"){
      std::istringstream iss(line.substr(1,line.length()));
      if (nRows == 0) {
        iss >> nRows;
      } else if (nAllColumns == 0) {
        iss >> nAllColumns;
      } else if (columnMap == NULL) {
        columnMap = new unsigned int[nAllColumns];
        unsigned int itmp = 0;
        currentColumn = 0;
        while (iss >> itmp){    
          columnMap[currentColumn++] = itmp;
          if (itmp == 1){
            ++nColumns;
          }
        }
      }
      continue;
    }
    if (data == NULL) {
      data = new float[nRows*nColumns];
    }
    std::istringstream iss(line);
    currentColumn = 0;
    while (iss >> tmp){
      if (columnMap[currentColumn++] != 1) {
        continue;
      }
      data[j++] = tmp;
    }
  }
  file.close();
  return data;
}

/** Reads a matrix
 * @param inFileNae
 * @param nRows - returns the number of rows
 * @param nColumns - returns the number of columns
 * @return the matrix
 */

float *readMatrix(const char *inFileName, unsigned int &nRows, unsigned int &nColumns)
{
  if (isLrnFile(inFileName)) {
    return readLrnFile(inFileName, nRows, nColumns);
  }
  ifstream file;
  file.open(inFileName);
  float *data=NULL;
  if (file.is_open()){
    string line;
    float tmp;
    while(getline(file,line)){
      if (line.substr(0,1) == "#"){
        continue;
      }
      std::istringstream iss(line);
      if (nRows == 0){
        while (iss >> tmp){
          nColumns++;
        }
      }
      nRows++;
    }
    data=new float[nRows*nColumns];
    file.close();file.open(inFileName);
    int j=0;
    while(getline(file,line)){
      if (line.substr(0,1) == "#"){
        continue;
      }
      std::istringstream iss(line);
      while (iss >> tmp){
        data[j++] = tmp;
      }
    }
    file.close();
  } else {
    std::cerr << "Input file could not be opened!\n";
    my_abort(-1);
  }
  return data;
}

void readSparseMatrixDimensions(const char *filename, unsigned int &nRows, 
                            unsigned int &nColumns) {
  ifstream file;
  file.open(filename);
  if (file.is_open()){  
    string line;
    int max_index=-1;
    while(getline(file,line)) {
      if (line.substr(0,1) == "#"){
        continue;
      }
      stringstream linestream(line);
      string value;
      int dummy_index;
      while(getline(linestream,value,' ')) {
        int separator=value.find(":");
        istringstream myStream(value.substr(0,separator));
        myStream >> dummy_index;
        if(dummy_index > max_index){
          max_index = dummy_index;
        }
      }
      ++nRows;
    }
    nColumns=max_index+1;
    file.close();               
  } else {
    std::cerr << "Input file could not be opened!\n";
    my_abort(-1);
  }
}

svm_node** readSparseMatrixChunk(const char *filename, unsigned int nRows, 
                                 unsigned int nRowsToRead, 
                                 unsigned int rowOffset) {
  ifstream file;
  file.open(filename);
  string line;
  for (unsigned int i=0; i<rowOffset; i++) {
    getline(file, line);
  }
  if (rowOffset+nRowsToRead >= nRows) {
    nRowsToRead = nRows-rowOffset;
  }
  svm_node **x_matrix = new svm_node *[nRowsToRead];
  for(unsigned int i=0;i<nRowsToRead;i++) {	
    getline(file, line);
    if (line.substr(0,1) == "#"){
       --i;
       continue;
    }
    stringstream tmplinestream(line);
    string value;
    int elements = 0;
    while(getline(tmplinestream,value,' ')) {
      elements++;
    }
    elements++; // To account for the closing dummy node in the row
    x_matrix[i] = new svm_node[elements];
    stringstream linestream(line);
    int j=0;
    while(getline(linestream,value,' ')) {
      int separator=value.find(":");
      istringstream myStream(value.substr(0,separator));
      myStream >> x_matrix[i][j].index;
      istringstream myStream2(value.substr(separator+1));
      myStream2 >> x_matrix[i][j].value;
      j++;
    }
    x_matrix[i][j].index = -1;
  }
  file.close();
  return x_matrix;
}
