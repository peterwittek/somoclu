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
#include <iostream>
#include <algorithm>
#include "somoclu.h"

float euclideanDistanceOnToroidMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y, const unsigned int nSomX, const unsigned int nSomY) {
    unsigned int x1 = std::min(som_x, x);
    unsigned int y1 = std::min(som_y, y);
    unsigned int x2 = std::max(som_x, x);
    unsigned int y2 = std::max(som_y, y);
    unsigned int xdist = std::min(x2-x1, x1+nSomX-x2);
    unsigned int ydist = std::min(y2-y1, y1+nSomY-y2);

    //unsigned int xdist = std::min(std::abs(x1-x2), std::abs(x1+nSomX-x2));
    //unsigned int ydist = std::min(std::abs(y1-y2), std::abs(y1+nSomY-y2));
    return sqrt(float(xdist*xdist+ydist*ydist));
}

float euclideanDistanceOnPlanarMap(const unsigned int som_x, const unsigned int som_y, const unsigned int x, const unsigned int y) {
    unsigned int x1 = std::min(som_x, x);
    unsigned int y1 = std::min(som_y, y);
    unsigned int x2 = std::max(som_x, x);
    unsigned int y2 = std::max(som_y, y);
    unsigned int xdist = x2-x1;
    unsigned int ydist = y2-y1;
    return sqrt(float(xdist*xdist+ydist*ydist));
}
