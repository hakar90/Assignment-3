#include <stdio.h>
#include <cuda.h>

#define WIDTH 1024   // Define the size of the matrix
#define TILE_WIDTH 16  // Define the size of the tile

__global__ void matMulTiled(float *A, float *B, float *C, int width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float sum = 0.0;

    for (int m = 0; m < (width-1)/TILE_WIDTH+1; ++m) {
        if (m*TILE_WIDTH + tx < width && row < width)
            tile_A[ty][tx] = A[row*width + m*TILE_WIDTH + tx];
        else
            tile_A[ty][tx] = 0.0;

        if (m*TILE_WIDTH + ty < width && col < width)
            tile_B[ty][tx] = B[(m*TILE_WIDTH + ty)*width + col];
        else
            tile_B[ty][tx] = 0.0;

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += tile_A[ty][k] * tile_B[k][tx];
        __syncthreads();
    }
    if (row < width && col < width)
        C[row*width + col] = sum;
}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    int size = WIDTH * WIDTH * sizeof(float);

    A = (float*) malloc(size);
    B = (float*) malloc(size);
    C = (float*) malloc(size);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize A and B with some values
    for (int i = 0; i < WIDTH*WIDTH; i++) {
        A[i] = 0.01f * i;
        B[i] = 0.02f * i;
    }

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Record event on the default stream
    cudaEventRecord(start);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH, (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH);
    matMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH);

    // Record event on the default stream
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
