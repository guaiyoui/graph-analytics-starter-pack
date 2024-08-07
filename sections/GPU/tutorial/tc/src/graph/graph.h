/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2024/4/7.
 */
#ifndef TC_GRAPH_H
#define TC_GRAPH_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cstdlib>

#define MAX_IDS 99999999

class Graph {
public:
    explicit Graph(const std::string& filename);
    ~Graph();
public:
    int n{}, m{};
    int* offsets, *degrees, *neighbors, *edges, *edge_ids;
    unsigned long long cnt{};
};


#endif //TC_GRAPH_H
