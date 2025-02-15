#include <cuda_runtime.h>
#include <stdio.h>

#define THREAD_PER_BLOCK 256

/*
    naive achivement
    just like a tree, a thread processes neighboring data
*/
__global__ void reduce0(float *d_in, float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        if (tid % (2*s) == 0){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}


/*
    v0存在warp divergent问题，一个block中的所有线程都会执行同一条指令
    如果存在if-else这样的分支，每个线程都会执行
    也有资料说是因为cuda中取余操作费时过多
    v1尽可能地让所有线程走到同一个分支里面
*/
__global__ void reduce1(float *input, float *output){
    __shared__ float sdata[THREAD_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2){
        int index = 2 * s * tid;
        if (index < blockDim.x){
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if (tid ==0) output[blockIdx.x] = sdata[0];
}


/*
    v1存在bank冲突，同一个warp中有多个线程需要取同一bank中的数，
    例如，第一次迭代中，warp0中，thread0需要取地址0，1的数，thread16需要取地址32，31的数
    v2修改了stride从128到0
*/
__global__ void reduce2(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}


/*
    v2存在idle线程，每次迭代结束后，干活的线程会减少一半
    v3在取数到sharedMem的过程中，加入了一次加法，
    减少了block的数量，每个block处理的数据多了一倍，由256增加到512
*/
__global__ void reduce3(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}


/*
    最后几轮迭代，只有warp0在工作，此时syncthreads可能造成浪费
    将最后一维进行展开，减少同步
*/
__device__ void warpReduce(volatile float* cache, unsigned int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce4(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

/*
    将for循环完全展开，但现代编译器可能已对此进行优化，提升有限
*/
