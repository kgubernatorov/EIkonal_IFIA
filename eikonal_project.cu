#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define N 1024  // Grid size (NxN or NxNxN)
#define MAX_ITER 10000
#define OBSTACLE_PROB 0.1f

#define CUDA_CHECK(err) if(err != cudaSuccess) { \
    printf("CUDA Error: %s\n", cudaGetErrorString(err)); exit(-1); }

#define IDX(x,y) ((y)*N+(x))
#define IDX3(x,y,z) (((z)*N + (y))*N + (x))

__device__ __host__ inline float speed(float obstacle) {
    return obstacle > 0.5f ? 1.0f : 0.01f;
}

// ===================== 2D Kernels =========================

__global__ void eikonal_baseline_2d(float* d_T, const float* d_obstacle, bool* d_converged) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || y <= 0 || x >= N-1 || y >= N-1) return;

    int idx = IDX(x,y);

    float t_left = d_T[IDX(x-1,y)];
    float t_right = d_T[IDX(x+1,y)];
    float t_up = d_T[IDX(x,y-1)];
    float t_down = d_T[IDX(x,y+1)];

    float s = speed(d_obstacle[idx]);
    if (s < 0.1f) return;

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
        *d_converged = false;
    }
}

__global__ void eikonal_improved_update_2d(float* d_T, const float* d_obstacle, int* d_activeList, int activeCount, bool* d_converged) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= activeCount) return;

    int idx = d_activeList[tid];
    int x = idx % N;
    int y = idx / N;

    float s = speed(d_obstacle[idx]);
    if (s < 0.1f) return;

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
        *d_converged = false;
    }
}

// ===================== 3D Kernels =========================

__global__ void eikonal_baseline_3d(float* d_T, const float* d_obstacle, bool* d_converged) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x <= 0 || y <= 0 || z <= 0 || x >= N-1 || y >= N-1 || z >= N-1) return;

    int idx = IDX3(x,y,z);

    float t_left = d_T[IDX3(x-1,y,z)];
    float t_right = d_T[IDX3(x+1,y,z)];
    float t_up = d_T[IDX3(x,y-1,z)];
    float t_down = d_T[IDX3(x,y+1,z)];
    float t_front = d_T[IDX3(x,y,z-1)];
    float t_back = d_T[IDX3(x,y,z+1)];

    float s = speed(d_obstacle[idx]);
    if (s < 0.1f) return;

    float a = fminf(t_left, t_right);
    float b = fminf(t_up, t_down);
    float c = fminf(t_front, t_back);

    // 3D quadratic update
    float vals[3] = {a, b, c};
    // Sort vals so that vals[0] <= vals[1] <= vals[2]
    if (vals[0] > vals[1]) { float tmp = vals[0]; vals[0] = vals[1]; vals[1] = tmp; }
    if (vals[1] > vals[2]) { float tmp = vals[1]; vals[1] = vals[2]; vals[2] = tmp; }
    if (vals[0] > vals[1]) { float tmp = vals[0]; vals[0] = vals[1]; vals[1] = tmp; }

    float t = vals[0] + 1.0f/s;
    if (t > vals[1]) {
        float sum = vals[0] + vals[1];
        float disc = sum*sum - 2*(vals[0]*vals[0] + vals[1]*vals[1] - 1.0f/(s*s));
        if (disc > 0) {
            t = (sum + sqrtf(disc)) / 2.0f;
        }
        if (t > vals[2]) {
            float sum3 = vals[0] + vals[1] + vals[2];
            float disc3 = sum3*sum3 - 3*(vals[0]*vals[0] + vals[1]*vals[1] + vals[2]*vals[2] - 1.0f/(s*s));
            if (disc3 > 0) {
                t = (sum3 + sqrtf(disc3)) / 3.0f;
            }
        }
    }
    if (t < d_T[idx]) {
        d_T[idx] = t;
        *d_converged = false;
    }
}

__global__ void eikonal_improved_update_3d(float* d_T, const float* d_obstacle, int* d_activeList, int activeCount, bool* d_converged) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= activeCount) return;

    int idx = d_activeList[tid];
    int x = idx % N;
    int y = (idx / N) % N;
    int z = idx / (N*N);

    float s = speed(d_obstacle[idx]);
    if (s < 0.1f) return;

    float t_left = (x > 0) ? d_T[IDX3(x-1,y,z)] : 1e10f;
    float t_right = (x < N-1) ? d_T[IDX3(x+1,y,z)] : 1e10f;
    float t_up = (y > 0) ? d_T[IDX3(x,y-1,z)] : 1e10f;
    float t_down = (y < N-1) ? d_T[IDX3(x,y+1,z)] : 1e10f;
    float t_front = (z > 0) ? d_T[IDX3(x,y,z-1)] : 1e10f;
    float t_back = (z < N-1) ? d_T[IDX3(x,y,z+1)] : 1e10f;

    float a = fminf(t_left, t_right);
    float b = fminf(t_up, t_down);
    float c = fminf(t_front, t_back);

    float vals[3] = {a, b, c};
    if (vals[0] > vals[1]) { float tmp = vals[0]; vals[0] = vals[1]; vals[1] = tmp; }
    if (vals[1] > vals[2]) { float tmp = vals[1]; vals[1] = vals[2]; vals[2] = tmp; }
    if (vals[0] > vals[1]) { float tmp = vals[0]; vals[0] = vals[1]; vals[1] = tmp; }

    float t = vals[0] + 1.0f/s;
    if (t > vals[1]) {
        float sum = vals[0] + vals[1];
        float disc = sum*sum - 2*(vals[0]*vals[0] + vals[1]*vals[1] - 1.0f/(s*s));
        if (disc > 0) {
            t = (sum + sqrtf(disc)) / 2.0f;
        }
        if (t > vals[2]) {
            float sum3 = vals[0] + vals[1] + vals[2];
            float disc3 = sum3*sum3 - 3*(vals[0]*vals[0] + vals[1]*vals[1] + vals[2]*vals[2] - 1.0f/(s*s));
            if (disc3 > 0) {
                t = (sum3 + sqrtf(disc3)) / 3.0f;
            }
        }
    }
    if (t < d_T[idx]) {
        d_T[idx] = t;
        *d_converged = false;
    }
}

// ============== Obstacle Generation ===============

void generate_obstacles_2d(float* h_obstacle) {
    for (int i = 0; i < N*N; i++) {
        float r = (float)rand() / RAND_MAX;
        h_obstacle[i] = (r < OBSTACLE_PROB) ? 0.0f : 1.0f;
    }
    h_obstacle[IDX(0,0)] = 1.0f;
    h_obstacle[IDX(N-1,N-1)] = 1.0f;
}

void generate_obstacles_3d(float* h_obstacle) {
    for (int i = 0; i < N*N*N; i++) {
        float r = (float)rand() / RAND_MAX;
        h_obstacle[i] = (r < OBSTACLE_PROB) ? 0.0f : 1.0f;
    }
    h_obstacle[IDX3(0,0,0)] = 1.0f;
    h_obstacle[IDX3(N-1,N-1,N-1)] = 1.0f;
}

// ============== Path Backtracing ===============

void backtrace_path_2d(float* h_T, int* path_x, int* path_y, int& path_len) {
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
        if (next_x == x && next_y == y) break;
        x = next_x;
        y = next_y;
        path_x[path_len] = x;
        path_y[path_len] = y;
        path_len++;
        if (path_len > N*N) break;
    }
}

void backtrace_path_3d(float* h_T, int* path_x, int* path_y, int* path_z, int& path_len) {
    int x = N-1, y = N-1, z = N-1;
    path_len = 0;
    path_x[path_len] = x;
    path_y[path_len] = y;
    path_z[path_len] = z;
    path_len++;

    while (x != 0 || y != 0 || z != 0) {
        float min_t = h_T[IDX3(x,y,z)];
        int next_x = x, next_y = y, next_z = z;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    if (nx >= 0 && nx < N && ny >= 0 && ny < N && nz >= 0 && nz < N) {
                        float nt = h_T[IDX3(nx, ny, nz)];
                        if (nt < min_t) {
                            min_t = nt;
                            next_x = nx;
                            next_y = ny;
                            next_z = nz;
                        }
                    }
                }
            }
        }
        if (next_x == x && next_y == y && next_z == z) break;
        x = next_x;
        y = next_y;
        z = next_z;
        path_x[path_len] = x;
        path_y[path_len] = y;
        path_z[path_len] = z;
        path_len++;
        if (path_len > N*N*N) break;
    }
}

// ============== Main Experiment Loop ===============

void run_2d_case() {
    float* h_T = (float*)malloc(N*N*sizeof(float));
    float* h_obstacle = (float*)malloc(N*N*sizeof(float));
    int* h_path_x = (int*)malloc(N*N*sizeof(int));
    int* h_path_y = (int*)malloc(N*N*sizeof(int));

    generate_obstacles_2d(h_obstacle);

    // Baseline 2D
    for (int i = 0; i < N*N; i++) h_T[i] = 1e10f;
    h_T[IDX(0,0)] = 0.0f;

    float *d_T, *d_obstacle;
    bool *d_converged;
    CUDA_CHECK(cudaMalloc(&d_T, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obstacle, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_converged, sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_T, h_T, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obstacle, h_obstacle, N*N*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((N+block.x-1)/block.x, (N+block.y-1)/block.y);

    bool h_converged;
    int iter = 0;
    auto start_baseline = std::chrono::high_resolution_clock::now();
    do {
        h_converged = true;
        CUDA_CHECK(cudaMemcpy(d_converged, &h_converged, sizeof(bool), cudaMemcpyHostToDevice));
        eikonal_baseline_2d<<<grid, block>>>(d_T, d_obstacle, d_converged);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost));
        iter++;
    } while (!h_converged && iter < MAX_ITER);
    auto end_baseline = std::chrono::high_resolution_clock::now();
    double baseline_time = std::chrono::duration<double>(end_baseline - start_baseline).count();

    CUDA_CHECK(cudaMemcpy(h_T, d_T, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    int path_len_baseline = 0;
    backtrace_path_2d(h_T, h_path_x, h_path_y, path_len_baseline);
    printf("2D Baseline: %d iterations, %.4f s, path length: %d\n", iter, baseline_time, path_len_baseline);

    // Improved 2D
    for (int i = 0; i < N*N; i++) h_T[i] = 1e10f;
    h_T[IDX(0,0)] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_T, h_T, N*N*sizeof(float), cudaMemcpyHostToDevice));

    int* d_activeList;
    int activeCount = N*N;
    int* h_activeList = (int*)malloc(activeCount*sizeof(int));
    for (int i = 0; i < activeCount; i++) h_activeList[i] = i;
    CUDA_CHECK(cudaMalloc(&d_activeList, activeCount*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_activeList, h_activeList, activeCount*sizeof(int), cudaMemcpyHostToDevice));

    iter = 0;
    auto start_improved = std::chrono::high_resolution_clock::now();
    do {
        h_converged = true;
        CUDA_CHECK(cudaMemcpy(d_converged, &h_converged, sizeof(bool), cudaMemcpyHostToDevice));
        int threads = 256;
        int blocks = (activeCount + threads - 1) / threads;
        eikonal_improved_update_2d<<<blocks, threads>>>(d_T, d_obstacle, d_activeList, activeCount, d_converged);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost));
        iter++;
    } while (!h_converged && iter < MAX_ITER);
    auto end_improved = std::chrono::high_resolution_clock::now();
    double improved_time = std::chrono::duration<double>(end_improved - start_improved).count();

    CUDA_CHECK(cudaMemcpy(h_T, d_T, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    int path_len_improved = 0;
    backtrace_path_2d(h_T, h_path_x, h_path_y, path_len_improved);
    printf("2D Improved: %d iterations, %.4f s, path length: %d\n", iter, improved_time, path_len_improved);

    // Cleanup
    free(h_T); free(h_obstacle); free(h_activeList); free(h_path_x); free(h_path_y);
    cudaFree(d_T); cudaFree(d_obstacle); cudaFree(d_converged); cudaFree(d_activeList);
}

void run_3d_case() {
    float* h_T = (float*)malloc(N*N*N*sizeof(float));
    float* h_obstacle = (float*)malloc(N*N*N*sizeof(float));
    int* h_path_x = (int*)malloc(N*N*N*sizeof(int));
    int* h_path_y = (int*)malloc(N*N*N*sizeof(int));
    int* h_path_z = (int*)malloc(N*N*N*sizeof(int));

    generate_obstacles_3d(h_obstacle);

    // Baseline 3D
    for (int i = 0; i < N*N*N; i++) h_T[i] = 1e10f;
    h_T[IDX3(0,0,0)] = 0.0f;

    float *d_T, *d_obstacle;
    bool *d_converged;
    CUDA_CHECK(cudaMalloc(&d_T, N*N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obstacle, N*N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_converged, sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_T, h_T, N*N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obstacle, h_obstacle, N*N*N*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(8,8,8);
    dim3 grid((N+block.x-1)/block.x, (N+block.y-1)/block.y, (N+block.z-1)/block.z);

    bool h_converged;
    int iter = 0;
    auto start_baseline = std::chrono::high_resolution_clock::now();
    do {
        h_converged = true;
        CUDA_CHECK(cudaMemcpy(d_converged, &h_converged, sizeof(bool), cudaMemcpyHostToDevice));
        eikonal_baseline_3d<<<grid, block>>>(d_T, d_obstacle, d_converged);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost));
        iter++;
    } while (!h_converged && iter < MAX_ITER);
    auto end_baseline = std::chrono::high_resolution_clock::now();
    double baseline_time = std::chrono::duration<double>(end_baseline - start_baseline).count();

    CUDA_CHECK(cudaMemcpy(h_T, d_T, N*N*N*sizeof(float), cudaMemcpyDeviceToHost));
    int path_len_baseline = 0;
    backtrace_path_3d(h_T, h_path_x, h_path_y, h_path_z, path_len_baseline);
    printf("3D Baseline: %d iterations, %.4f s, path length: %d\n", iter, baseline_time, path_len_baseline);

    // Improved 3D
    for (int i = 0; i < N*N*N; i++) h_T[i] = 1e10f;
    h_T[IDX3(0,0,0)] = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_T, h_T, N*N*N*sizeof(float), cudaMemcpyHostToDevice));

    int* d_activeList;
    int activeCount = N*N*N;
    int* h_activeList = (int*)malloc(activeCount*sizeof(int));
    for (int i = 0; i < activeCount; i++) h_activeList[i] = i;
    CUDA_CHECK(cudaMalloc(&d_activeList, activeCount*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_activeList, h_activeList, activeCount*sizeof(int), cudaMemcpyHostToDevice));

    iter = 0;
    auto start_improved = std::chrono::high_resolution_clock::now();
    do {
        h_converged = true;
        CUDA_CHECK(cudaMemcpy(d_converged, &h_converged, sizeof(bool), cudaMemcpyHostToDevice));
        int threads = 256;
        int blocks = (activeCount + threads - 1) / threads;
        eikonal_improved_update_3d<<<blocks, threads>>>(d_T, d_obstacle, d_activeList, activeCount, d_converged);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost));
        iter++;
    } while (!h_converged && iter < MAX_ITER);
    auto end_improved = std::chrono::high_resolution_clock::now();
    double improved_time = std::chrono::duration<double>(end_improved - start_improved).count();

    CUDA_CHECK(cudaMemcpy(h_T, d_T, N*N*N*sizeof(float), cudaMemcpyDeviceToHost));
    int path_len_improved = 0;
    backtrace_path_3d(h_T, h_path_x, h_path_y, h_path_z, path_len_improved);
    printf("3D Improved: %d iterations, %.4f s, path length: %d\n", iter, improved_time, path_len_improved);

    // Cleanup
    free(h_T); free(h_obstacle); free(h_activeList); free(h_path_x); free(h_path_y); free(h_path_z);
    cudaFree(d_T); cudaFree(d_obstacle); cudaFree(d_converged); cudaFree(d_activeList);
}

int main() {
    srand(1234);
    printf("==== Eikonal Equation: 2D and 3D, Baseline and Improved ====\n");
    run_2d_case();
    run_3d_case();
    return 0;
}
