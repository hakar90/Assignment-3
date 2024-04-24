#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024             // Matrix size
#define TILE_SIZE 32       // Tile size

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

    // Optimize data movement by keeping data on device between phases
    #pragma acc data copyin(A[0:N*N], B[0:N*N]) copyout(C[0:N*N])
    {
        // Use tiles to reduce memory traffic and improve cache usage
        #pragma acc parallel loop tile(TILE_SIZE, TILE_SIZE)
        for (int i = 0; i < N; i += TILE_SIZE) {
            for (int j = 0; j < N; j += TILE_SIZE) {
                float C_tile[TILE_SIZE][TILE_SIZE] = {0};

                #pragma acc loop seq
                for (int k = 0; k < N; k++) {
                    float A_tile[TILE_SIZE], B_tile[TILE_SIZE];

                    // Load one row and one column of the tiles into local arrays
                    #pragma acc loop seq
                    for (int kk = 0; kk < TILE_SIZE; kk++) {
                        A_tile[kk] = A[(i + kk) * N + k];
                        B_tile[kk] = B[k * N + (j + kk)];
                    }

                    // Compute product of tiles
                    #pragma acc loop vector
                    for (int ii = 0; ii < TILE_SIZE; ii++) {
                        for (int jj = 0; jj < TILE_SIZE; jj++) {
                            C_tile[ii][jj] += A_tile[ii] * B_tile[jj];
                        }
                    }
                }

                // Write the results from the tile back to the matrix C
                #pragma acc loop seq
                for (int ii = 0; ii < TILE_SIZE; ii++) {
                    for (int jj = 0; jj < TILE_SIZE; jj++) {
                        C[(i + ii) * N + (j + jj)] = C_tile[ii][jj];
                    }
                }
            }
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3f seconds\n", time_spent);

    free(A);
    free(B);
    free(C);

    return 0;
}
