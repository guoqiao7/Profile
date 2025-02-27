#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cublas_v2.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CHECK(call){\
    cudaError_t e = call;\
    if (e != cudaSuccess){\
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", e);   \
        printf("    Error text: %s\n", cudaGetErrorString(e));  \
        exit(1);   \
    }\
}

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum){
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

void initiaMatrix(float *matrix, int size){
    time_t t;
    srand((unsigned)time(&t));
    // time(NULL)可直接获取当前时间
    // srand((unsigned int)time(NULL));

    // 并行填充
    #pragma omp parallel for
    for (int i = 0; i < size; i++){
        matrix[i] = (float)rand() / (float)RAND_MAX;
    }
}

// N >= 128
__global__ void Sgem_v1(float *__restrict__ A, float *__restrict__ x, float *__restrict__ y,const int M, const int N){
    int bx = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size = 32;
    int laneId = tx % warp_size;
    int current_row = blockDim.y * bx + ty;

    if(current_row < M){
        float res = 0;
        int iter = ((N + warp_size - 1) / warp_size) / 4;
        if(iter == 0) iter = 1;
        A = &A[current_row * N];
        #pragma unroll
        for(int i = 0; i < iter; i++){
            int current_col = i * warp_size + laneId;
            float4 current_val = reinterpret_cast<float4 *>(A)[current_col];
            float4 current_x = reinterpret_cast<float4 *>(x)[current_col];
            res += current_val.x * current_x.x;
            res += current_val.y * current_x.y;
            res += current_val.z * current_x.z;
            res += current_val.w * current_x.w;
        }
        res = warpReduceSum<warp_size>(res);
        if (laneId == 0) y[current_row] = res;
    }
}

int main(int argc, char** argv){
    if(argc != 3){
        printf("usage: ./main [M] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);

    size_t bytes_A = sizeof(float) * M * N;
    size_t bytes_x = sizeof(float) * N;
    size_t bytes_y = sizeof(float) * M;

    float *h_A = (float *)malloc(bytes_A);
    float *h_x = (float *)malloc(bytes_x);
    float *h_y = (float *)malloc(bytes_y);
    float *h_y1 = (float *)malloc(bytes_y);

    float *d_A;
    float *d_x;
    float *d_y;
    CHECK(cudaMalloc(&d_A, bytes_A));
    CHECK(cudaMalloc(&d_x, bytes_x));
    CHECK(cudaMalloc(&d_y, bytes_y));
    
    initiaMatrix(h_A, M * N);
    initiaMatrix(h_x, N);

    memset(h_y, 0, bytes_y);
    memset(h_y1, 0, bytes_y);

    int loop = 3;
    CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice));

    dim3 block(32, 4);
    dim3 grid((M + block.y -1) / block.y);
    for (int i = 0; i < loop; i++){
        Sgem_v1<<<grid, block>>>(d_A, d_x, d_y, M, N);
    }
    
    CHECK(cudaMemcpy(h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0.0;
    CHECK(cudaMemcpy(d_y, h_y1, bytes_y, cudaMemcpyHostToDevice));
    for (int i = 0; i < loop; i++){
        cublasSgemv(blas_handle, CUBLAS_OP_T,
                    N, M, &alpha,
                    d_A, N, d_x, 1, &beta,
                    d_y, 1);
    }
    CHECK(cudaMemcpy(h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));
    cublasDestroy(blas_handle);

    double eps = 1.e-6;
    bool correct = true;
    for (int i = 0; i < M; i++){
        double abs_err = fabs(h_y[i] - h_y1[i]);
        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if(rel_err > eps){
            printf("Unequal! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_y[i], h_y1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result is equal" : "Result is FAIL");

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y1);
}