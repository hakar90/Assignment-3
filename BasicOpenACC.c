#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

int main() {
    float *A, *B, *C;
    A = (float*) malloc(N * N * sizeof(float));
    B = (float*) malloc(N * N * sizeof(float));
    C = (float*) malloc(N * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 0.01f * i;
        B[i] = 0.02f * i;
    }

    clock_t start = clock();

    #pragma acc data copyin(A[0:N*N], B[0:N*N]) copy(C[0:N*N])
    {
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0;
                #pragma acc loop vector reduction(+:sum) // Optimize vector length based on GPU capabilities
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Elapsed time: %.3f seconds\n", time_spent-1);

    free(A);
    free(B);
    free(C);

    return 0;
}
