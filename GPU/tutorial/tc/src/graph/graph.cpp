/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2024/4/7.
 */
#include "graph.h"

Graph::Graph(const std::string &filename) {
    auto file = std::ifstream(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    auto idx = 0;
    auto vertices_id = std::vector<int>(MAX_IDS, -1);
    auto tmp_edges = std::vector<std::vector<int>>();

    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == '%')  continue;

        // 提取两个数字并添加到数组中
        std::istringstream iss(line);
        int u, v;
        iss >> u >> v;

        if (vertices_id[u] == -1) {
            vertices_id[u] = idx;
            idx += 1;
        }

        if (vertices_id[v] == -1) {
            vertices_id[v] = idx;
            idx += 1;
        }

        tmp_edges.resize(idx);

        tmp_edges[vertices_id[u]].push_back(vertices_id[v]);
        tmp_edges[vertices_id[v]].push_back(vertices_id[u]);

        m += 1;
    }

    n = idx;
    degrees = new int[idx];
    offsets = new int[idx + 1];
    neighbors = new int[m * 2];
    edges = new int[m * 2];
    edge_ids = new int[m * 2];

    offsets[0] = 0;
    auto all_edges = std::vector<int>();
    for(auto i = 0; i < idx; i ++) {
        // sort neighbors
        std::sort(tmp_edges[i].begin(), tmp_edges[i].end());
        all_edges.insert(all_edges.end(), tmp_edges[i].begin(), tmp_edges[i].end());

        degrees[i] = int(tmp_edges[i].size());

        if (i > 0) offsets[i] = offsets[i - 1] + degrees[i - 1];
    }
    std::copy(all_edges.begin(), all_edges.end(), neighbors);

    auto eid = 0;
    for (auto u = 0; u < idx; u ++) {
        auto u_nbr = neighbors + offsets[u];

        for (auto i = 0; i < degrees[u]; i ++) {
            auto const v = u_nbr[i];

            if (u < v) {
                edge_ids[offsets[u] + i] = eid / 2;
                edges[eid] = u;
                edges[eid + 1] = v;

                eid += 2;
            } else {
                // if v < u, then use binary search to find the edge id
                auto left = 0;
                auto right = degrees[v] - 1;
                auto v_nbr = neighbors + offsets[v];

                while (left <= right) {
                    auto mid = (left + right) / 2;
                    if (v_nbr[mid] == u) {
                        edge_ids[offsets[u] + i] = edge_ids[offsets[v] + mid];
                        break;
                    } else if (v_nbr[mid] < u) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
            }
        }
    }

    // assign idx for edges
    std::cout << "Graph loaded: " << n << " vertices, " << m << " edges" << std::endl;

}

Graph::~Graph() {
    delete[] offsets;
    delete[] degrees;
    delete[] neighbors;
    delete[] edges;
    delete[] edge_ids;
}
