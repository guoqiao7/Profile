#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <omp.h>

#define THREAD_PER_BLOCK 256

#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void add(float *a, float *b, float *c){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    c[idx] = a[idx] + b[idx];
}

__global__ void vec2_add(float *a, float *b, float *c){
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    float2 reg_a = FETCH_FLOAT2(a[idx]);
    float2 reg_b = FETCH_FLOAT2(b[idx]);
    float2 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    FETCH_FLOAT2(c[idx]) = reg_c;
}

__global__ void vec4_add(float *a, float *b, float *c){
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FETCH_FLOAT4(c[idx]) = reg_c;
}

bool check(float *out, float *res, int n){
    for(int i = 0; i < n; i++){
        if(out[i] != res[i]){
            printf("unequal! out[%d] = %f, res[%d] = %f\n",i,out[i],i,res[i]);
            return false;
        }
    }
    return true;
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

int main(){
    const int N = 32 * 1024 * 1024;
    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *res = (float *)malloc(N * sizeof(float));
    float *out = (float *)malloc(N * sizeof(float));
    float *d_a;
    float *d_b;
    float *d_out;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    initiaMatrix(a, N);
    initiaMatrix(b, N);
    for(int i = 0; i < N; i++){
        res[i] = a[i] + b[i];
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(N / THREAD_PER_BLOCK);
    dim3 grid_2(N / THREAD_PER_BLOCK / 2);
    dim3 grid_4(N / THREAD_PER_BLOCK / 4);
    dim3 block(THREAD_PER_BLOCK);

    int iter = 3;
    for(int i = 0; i < iter; i++){
        add<<<grid, block>>>(d_a, d_b, d_out);
    }
    cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(out, res, N)) printf("the ans is equal\n");

    // -----------------------------------------------------------------------
    for(int i = 0; i < iter; i++){
        vec2_add<<<grid_2, block>>>(d_a, d_b, d_out);
    }
    cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (check(out, res, N)) printf("the ans is equal\n");
    
    // -----------------------------------------------------------------------
    for(int i = 0; i < iter; i++){
        vec4_add<<<grid_4, block>>>(d_a, d_b, d_out);
    }
    cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (check(out, res, N)) printf("the ans is equal\n");

    free(a);
    free(b);
    free(res);
    free(out);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}