/**
 * Self-Organizing Maps on a cluster
 * MIT License
 * 
 * Copyright (c) 2013 Peter Wittek
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
**/
#include <cmath>
#include <iostream>
#include <algorithm>
#include "somoclu.h"

float euclideanDistanceOnPlanarMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y) {
    unsigned int x1 = std::min(som_x, x);
    unsigned int y1 = std::min(som_y, y);
    unsigned int x2 = std::max(som_x, x);
    unsigned int y2 = std::max(som_y, y);
    unsigned int xdist = x2 - x1;
    unsigned int ydist = y2 - y1;
    return sqrt(float(xdist * xdist + ydist * ydist));
}

float euclideanDistanceOnToroidMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y, const unsigned int nSomX, const unsigned int nSomY) {
    unsigned int x1 = std::min(som_x, x);
    unsigned int y1 = std::min(som_y, y);
    unsigned int x2 = std::max(som_x, x);
    unsigned int y2 = std::max(som_y, y);
    unsigned int xdist = std::min(x2 - x1, x1 + nSomX - x2);
    unsigned int ydist = std::min(y2 - y1, y1 + nSomY - y2);
    return sqrt(float(xdist * xdist + ydist * ydist));
}

float euclideanDistanceOnHexagonalPlanarMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y) {
    unsigned int x1 = std::min(som_x, x);
    unsigned int y1 = std::min(som_y, y);
    unsigned int x2 = std::max(som_x, x);
    unsigned int y2 = std::max(som_y, y);
    unsigned int xdist = x2 - x1;
    unsigned int ydist = y2 - y1;
    if (ydist & 1) {
        xdist += ((y1 & 1) ? -0.5 : 0.5);
    }
    return sqrt(float(xdist * xdist + ydist * ydist * 0.75));
}

float euclideanDistanceOnHexagonalToroidMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y, const unsigned int nSomX, const unsigned int nSomY) {
    unsigned int x1 = std::min(som_x, x);
    unsigned int y1 = std::min(som_y, y);
    unsigned int x2 = std::max(som_x, x);
    unsigned int y2 = std::max(som_y, y);
    unsigned int xdist = std::min(x2 - x1, x1 + nSomX - x2);
    unsigned int ydist = std::min(y2 - y1, y1 + nSomY - y2);
    if (ydist & 1) {
        xdist += ((y1 & 1) ? -0.5 : 0.5);
    }
    return sqrt(float(xdist * xdist + ydist * ydist * 0.75));
}

float gaussianNeighborhood(float distance, float radius, float std_coeff) {
    float std = radius * std_coeff;
    return exp((- distance * distance) / (2 * std * std));
}

float getWeight(float distance, float radius, float scaling, bool compact_support = false,
                bool gaussian = true, float std_coeff=0.5) {
    float result = 0.0;
    if (gaussian) {
        if (!compact_support || distance <= radius) {
            result = gaussianNeighborhood(distance, radius, std_coeff);
        }
    } else {
        if (distance <= radius) {
            result = 1.0;
        }
    }
    return scaling * result;
}
