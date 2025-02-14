#include <bits/stdc++.h>

#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <ctime>
#include <sys/time.h>

#include <cublas_v2.h>

void __global__ add1(float *x, float *y, float *z){
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    z[n] = x[n] + y[n];
}

void __global__ add2(float *x, float *y, float *z){
    int n = threadIdx.x + blockIdx.x * blockDim.x + 1;
    z[n] = x[n] + y[n];
}

void __global__ add3(float *x, float *y, float *z){
    int tid_permuted = threadIdx.x ^ 0x1;
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    z[n] = x[n] + y[n];
}


int main(){
    const int N = 32 * 1024 * 1024;
    float *input_x = (float *)malloc(N * sizeof(float));
    float *input_y = (float *)malloc(N * sizeof(float));
    float *d_input_x;
    float *d_input_y;
    cudaMalloc((void **)&d_input_x, N * sizeof(float));
    cudaMalloc((void **)&d_input_y, N * sizeof(float));
    cudaMemcpy(d_input_x, input_x, N *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_y, input_y, N *sizeof(float), cudaMemcpyHostToDevice);

    float *output = (float *)malloc(N * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, N * sizeof(float));

    dim3 grid(N / 256);
    dim3 block(64);

    for (int i = 0; i < 2; i++){
        add1<<<grid, block>>>(d_input_x, d_input_y, d_output);
        cudaDeviceSynchronize();
    }

    cudaFree(d_input_x);
    cudaFree(d_input_y);
    cudaFree(d_output);
    free(input_x);
    free(input_y);
    free(output);
    
    return 0;
}