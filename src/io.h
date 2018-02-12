#include <string>
using namespace std;

#ifndef SOMOCLU_IO_H
#define SOMOCLU_IO_H

void cli_abort(string err);
int saveCodebook(string cbFileName, som map);
int saveUMatrix(string fname, som map);
int saveBmus(string filename, som map);
float *readMatrix(const string inFilename,
                  unsigned int &nRows, unsigned int &nCols);
void readSparseMatrixDimensions(const string filename, unsigned int &nRows,
                                unsigned int &nColumns, bool& zerobased);
svm_node** readSparseMatrixChunk(const string filename, unsigned int nRows,
                                 unsigned int nRowsToRead,
                                 unsigned int rowOffset,
                                 unsigned int colOffset=0);
#endif
