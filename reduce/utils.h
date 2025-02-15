#include <stdio.h>

bool check(float *A, float *B, int n){
    for(int i = 0; i < n; i++){
        if(A[i] != B[i]){
            printf("unequal in matrix[%d]\n  gpu:%f,    cpu:%f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

__global__ void reduce0(float *d_in, float *d_out);
__global__ void reduce1(float *d_in, float *d_out);
__global__ void reduce2(float *d_in, float *d_out);
__global__ void reduce3(float *d_in, float *d_out);
__global__ void reduce4(float *d_in, float *d_out);