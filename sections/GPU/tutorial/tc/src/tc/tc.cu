/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2024/4/7.
 */
#include "tc.cuh"

__global__ auto tc_kernel(int const* d_offset, int const* d_edges, int const* d_degrees, int const* d_neighbors, unsigned long long* d_cnt) -> void {

    uint const bid = blockIdx.x;
    __shared__ unsigned long long cnt;

    int u = d_edges[bid];
    int v = d_edges[bid + 1];

    if (threadIdx.x == 0) {
        cnt = 0;

        if (d_degrees[u] > d_degrees[v]) {
            int tmp = u;
            u = v;
            v = tmp;
        }
    }

    __syncthreads();


    for (uint i = threadIdx.x; i < d_degrees[u]; i += blockDim.x) {
        // use binary search to find the common neighbors
        int const w = d_neighbors[d_offset[u] + i];
        int const* v_nbr = d_neighbors + d_offset[v];

        int l = 0, r = d_degrees[v] - 1;
        while (l <= r) {
            int const mid = (l + r) / 2;
            if (v_nbr[mid] == w) {
                atomicAdd(&cnt, 1);
                break;
            } else if (v_nbr[mid] < w) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(d_cnt, cnt);
    }

}

auto tc(Graph* g) -> void {

    int* d_offset, *d_edges, *d_degrees, *d_neighbors;
    unsigned long long* d_cnt;

    cudaMalloc(&d_offset, (g->n + 1) * sizeof(int));
    cudaMalloc(&d_edges, g->m * 2 * sizeof(int));
    cudaMalloc(&d_degrees, g->n * sizeof(int));
    cudaMalloc(&d_neighbors, g->m * 2 * sizeof(int));
    cudaMalloc(&d_cnt, sizeof(unsigned long long));

    cudaMemcpy(d_offset, g->offsets, (g->n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, g->edges, g->m * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_degrees, g->degrees, g->n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors, g->neighbors, g->m * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnt, &g->cnt, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    tc_kernel<<<g->m, 1024>>>(d_offset, d_edges, d_degrees, d_neighbors, d_cnt);

    cudaMemcpy(&g->cnt, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "Triangle Count: " << g->cnt << std::endl;
    // free
    cudaFree(d_offset);
    cudaFree(d_edges);
    cudaFree(d_degrees);
    cudaFree(d_neighbors);
    cudaFree(d_cnt);


}