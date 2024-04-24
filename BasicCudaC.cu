#include <stdio.h>
#include <cuda.h>

#define WIDTH 1024  // Define the size of the matrix

__global__ void matMulBasic(float *A, float *B, float *C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < width && col < width) {
        float sum = 0.0;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
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

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + 15) / 16, (WIDTH + 15) / 16);
    matMulBasic<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH);

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
