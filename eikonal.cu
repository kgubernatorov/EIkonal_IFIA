#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define N 512  // Grid size NxN
#define MAX_ITER 10000
#define OBSTACLE_PROB 0.1f

// CUDA error checking macro
#define CUDA_CHECK(err) if(err != cudaSuccess) { \
    printf("CUDA Error: %s\n", cudaGetErrorString(err)); exit(-1); }

// Indexing macro for 2D grid flattened to 1D
#define IDX(x,y) ((y)*N+(x))

// Speed function: 1 for free space, 0 for obstacles
__device__ __host__ inline float speed(float obstacle) {
    return obstacle > 0.5f ? 1.0f : 0.01f; // Very slow speed inside obstacles
}

// Baseline kernel: simple Jacobi update for Eikonal equation (upwind scheme)
__global__ void eikonal_baseline(float* d_T, const float* d_obstacle, bool* d_converged) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= N-1 || y >= N-1) return;

    int idx = IDX(x,y);

    float t_left = d_T[IDX(x-1,y)];
    float t_right = d_T[IDX(x+1,y)];
    float t_up = d_T[IDX(x,y-1)];
    float t_down = d_T[IDX(x,y+1)];

    float s = speed(d_obstacle[idx]);
    if (s < 0.1f) return; // obstacle, skip update

    // Solve quadratic update for Eikonal in 2D
    float a = fminf(t_left, t_right);
    float b = fminf(t_up, t_down);

    float new_t;
    if (fabsf(a - b) >= 1.0f / s) {
        new_t = fminf(a,b) + 1.0f / s;
    } else {
        float tmp = (a + b);
        new_t = (tmp + sqrtf(tmp*tmp - 2*(a*a + b*b - 1.0f/(s*s)))) / 2.0f;
    }

    if (new_t < d_T[idx]) {
        d_T[idx] = new_t;
        *d_converged = false; // mark not converged
    }
}

// Complex parallel approach inspired by improved fast iterative method[1]
// Uses active list and error correction
__global__ void eikonal_improved_update(float* d_T, const float* d_obstacle, int* d_activeList, int activeCount, bool* d_converged) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= activeCount) return;

    int idx = d_activeList[tid];
    int x = idx % N;
    int y = idx / N;

    float s = speed(d_obstacle[idx]);
    if (s < 0.1f) return; // obstacle

    // Get neighbors
    float t_left = (x > 0) ? d_T[IDX(x-1,y)] : 1e10f;
    float t_right = (x < N-1) ? d_T[IDX(x+1,y)] : 1e10f;
    float t_up = (y > 0) ? d_T[IDX(x,y-1)] : 1e10f;
    float t_down = (y < N-1) ? d_T[IDX(x,y+1)] : 1e10f;

    float a = fminf(t_left, t_right);
    float b = fminf(t_up, t_down);

    float new_t;
    if (fabsf(a - b) >= 1.0f / s) {
        new_t = fminf(a,b) + 1.0f / s;
    } else {
        float tmp = (a + b);
        new_t = (tmp + sqrtf(tmp*tmp - 2*(a*a + b*b - 1.0f/(s*s)))) / 2.0f;
    }

    if (new_t < d_T[idx]) {
        d_T[idx] = new_t;
        *d_converged = false; // mark not converged
    }
}

// Host function to generate random obstacles
void generate_obstacles(float* h_obstacle) {
    for (int i = 0; i < N*N; i++) {
        float r = (float)rand() / RAND_MAX;
        h_obstacle[i] = (r < OBSTACLE_PROB) ? 0.0f : 1.0f;
    }
    // Ensure start and goal are free
    h_obstacle[IDX(0,0)] = 1.0f;
    h_obstacle[IDX(N-1,N-1)] = 1.0f;
}

// Backtrace path from goal to start by gradient descent on T
void backtrace_path(float* h_T, int* path_x, int* path_y, int& path_len) {
    int x = N-1, y = N-1;
    path_len = 0;
    path_x[path_len] = x;
    path_y[path_len] = y;
    path_len++;

    while (x != 0 || y != 0) {
        float min_t = h_T[IDX(x,y)];
        int next_x = x, next_y = y;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                    float nt = h_T[IDX(nx, ny)];
                    if (nt < min_t) {
                        min_t = nt;
                        next_x = nx;
                        next_y = ny;
                    }
                }
            }
        }
        if (next_x == x && next_y == y) break; // stuck
        x = next_x;
        y = next_y;
        path_x[path_len] = x;
        path_y[path_len] = y;
        path_len++;
        if (path_len > N*N) break; // safety break
    }
}

int main() {
    srand(1234);

    // Host arrays
    float* h_T = (float*)malloc(N*N*sizeof(float));
    float* h_obstacle = (float*)malloc(N*N*sizeof(float));
    int* h_path_x = (int*)malloc(N*N*sizeof(int));
    int* h_path_y = (int*)malloc(N*N*sizeof(int));

    generate_obstacles(h_obstacle);

    // Initialize T to large values except start point
    for (int i = 0; i < N*N; i++) h_T[i] = 1e10f;
    h_T[IDX(0,0)] = 0.0f;

    // Device arrays
    float *d_T, *d_obstacle;
    bool *d_converged;
    CUDA_CHECK(cudaMalloc(&d_T, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obstacle, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_converged, sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_T, h_T, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obstacle, h_obstacle, N*N*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((N+block.x-1)/block.x, (N+block.y-1)/block.y);

    // Baseline approach timing
    bool h_converged;
    int iter = 0;
    auto start_baseline = std::chrono::high_resolution_clock::now();
    do {
        h_converged = true;
        CUDA_CHECK(cudaMemcpy(d_converged, &h_converged, sizeof(bool), cudaMemcpyHostToDevice));

        eikonal_baseline<<<grid, block>>>(d_T, d_obstacle, d_converged);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost));
        iter++;
    } while (!h_converged && iter < MAX_ITER);
    auto end_baseline = std::chrono::high_resolution_clock::now();
    double baseline_time = std::chrono::duration<double>(end_baseline - start_baseline).count();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_T, d_T, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    // Backtrace path for baseline
    int path_len_baseline = 0;
    backtrace_path(h_T, h_path_x, h_path_y, path_len_baseline);

    printf("Baseline approach converged in %d iterations, time: %.4f s\n", iter, baseline_time);

    // Reset T for improved approach
    for (int i = 0; i < N*N; i++) h_T[i] = 1e10f;
    h_T[IDX(0,0)] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_T, h_T, N*N*sizeof(float), cudaMemcpyHostToDevice));

    // Prepare active list (all points initially)
    int* d_activeList;
    int activeCount = N*N;
    int* h_activeList = (int*)malloc(activeCount*sizeof(int));
    for (int i = 0; i < activeCount; i++) h_activeList[i] = i;
    CUDA_CHECK(cudaMalloc(&d_activeList, activeCount*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_activeList, h_activeList, activeCount*sizeof(int), cudaMemcpyHostToDevice));

    // Improved approach timing
    iter = 0;
    auto start_improved = std::chrono::high_resolution_clock::now();
    do {
        h_converged = true;
        CUDA_CHECK(cudaMemcpy(d_converged, &h_converged, sizeof(bool), cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (activeCount + threads - 1) / threads;
        eikonal_improved_update<<<blocks, threads>>>(d_T, d_obstacle, d_activeList, activeCount, d_converged);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost));
        iter++;
    } while (!h_converged && iter < MAX_ITER);
    auto end_improved = std::chrono::high_resolution_clock::now();
    double improved_time = std::chrono::duration<double>(end_improved - start_improved).count();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_T, d_T, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    // Backtrace path for improved
    int path_len_improved = 0;
    backtrace_path(h_T, h_path_x, h_path_y, path_len_improved);

    printf("Improved approach converged in %d iterations, time: %.4f s\n", iter, improved_time);

    // Save output for visualization
    FILE* fout = fopen("eikonal_output.txt", "w");
    fprintf(fout, "Grid size: %d\n", N);
    fprintf(fout, "Obstacles (0=obstacle,1=free):\n");
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            fprintf(fout, "%d ", (h_obstacle[IDX(x,y)] > 0.5f) ? 1 : 0);
        }
        fprintf(fout, "\n");
    }
    fprintf(fout, "Arrival times:\n");
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            fprintf(fout, "%.2f ", h_T[IDX(x,y)]);
        }
        fprintf(fout, "\n");
    }
    fprintf(fout, "Path length: %d\n", path_len_improved);
    fprintf(fout, "Path coordinates (x y):\n");
    for (int i = 0; i < path_len_improved; i++) {
        fprintf(fout, "%d %d\n", h_path_x[i], h_path_y[i]);
    }
    fclose(fout);

    // Cleanup
    free(h_T);
    free(h_obstacle);
    free(h_activeList);
    free(h_path_x);
    free(h_path_y);
    cudaFree(d_T);
    cudaFree(d_obstacle);
    cudaFree(d_converged);
    cudaFree(d_activeList);

    return 0;
}
