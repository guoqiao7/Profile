#include <bits/stdc++.h>
#include <cuda.h>
#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

int main(){
    const int N = 32 * 1024 * 1024;
    int block_num = N / THREAD_PER_BLOCK;
    float *in = (float *)malloc(N * sizeof(float));
    float *out = (float *)malloc(block_num * sizeof(float));
    float *res = (float *)malloc(block_num * sizeof(float));

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, block_num * sizeof(float));


    free(in);
    free(out);
    free(res);
    cudaFree(d_in);
    cudaFree(d_out);
}