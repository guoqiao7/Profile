/*
    naive achivement
    just like a tree, a thread processes neighboring data
*/
#include <bits/stdc++.h>
#include <cuda.h>
#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include <sys/time.h>
#include "utils.h"

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

    // initial data
    for(int i = 0; i < N; i++){
        in[i] = 1;
    }

    // naive cpu 
    for(int i = 0; i < block_num; i++){
        float temp = 0;
        for(int j = 0; j < THREAD_PER_BLOCK; j++){
            temp += in[i * THREAD_PER_BLOCK + j];  
        }
        res[i] = temp;
    }

    // naive gpu 
    cudaMemcpy(d_in, in, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(block_num, 1);
    dim3 block(THREAD_PER_BLOCK, 1);

    // ------------------------------------------------------------------------
    reduce0<<<grid, block>>>(d_in, d_out);

    cudaMemcpy(out, d_out, block_num*sizeof(float), cudaMemcpyDeviceToHost);

    if(check(out, res, block_num)){
        printf("the result of cpu and gpu is equal\n");
    }

    // -------------------------------------------------------------------------
    reduce1<<<grid, block>>>(d_in, d_out);

    cudaMemcpy(out, d_out, block_num*sizeof(float), cudaMemcpyDeviceToHost);

    if(check(out, res, block_num)){
        printf("the result of cpu and gpu is equal\n");
    }

    // --------------------------------------------------------------------------
    reduce2<<<grid, block>>>(d_in, d_out);

    cudaMemcpy(out, d_out, block_num*sizeof(float), cudaMemcpyDeviceToHost);

    if(check(out, res, block_num)){
        printf("the result of cpu and gpu is equal\n");
    }

    // --------------------------------------------------------------------------
    int block_num_3 = N / (2 * THREAD_PER_BLOCK);
    float *out_3 = (float *)malloc(block_num_3 * sizeof(float));
    float *res_3 = (float *)malloc(block_num_3 * sizeof(float));

    float *d_out_3;
    cudaMalloc(&d_out_3, block_num_3 * sizeof(float));

    for(int i = 0; i < block_num_3; i++){
        float temp = 0;
        for(int j = 0; j < 2 * THREAD_PER_BLOCK; j++){
            temp += in[i * 2 * THREAD_PER_BLOCK + j];  
        }
        res_3[i] = temp;
    }

    dim3 grid_3(block_num_3, 1);
    dim3 block_3(THREAD_PER_BLOCK, 1);

    reduce3<<<grid_3, block_3>>>(d_in, d_out_3);

    cudaMemcpy(out_3, d_out_3, block_num_3*sizeof(float), cudaMemcpyDeviceToHost);

    if(check(out_3, res_3, block_num_3)){
        printf("the result of cpu and gpu is equal\n");
    }
    
    // --------------------------------------------------------------------------
    reduce4<<<grid_3, block_3>>>(d_in, d_out_3);

    cudaMemcpy(out_3, d_out_3, block_num_3*sizeof(float), cudaMemcpyDeviceToHost);

    if(check(out_3, res_3, block_num_3)){
        printf("the result of cpu and gpu is equal\n");
    }
    // --------------------------------------------------------------------------
    

    free(in);
    free(out);
    free(res);
    free(out_3);
    free(res_3);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_3);
}