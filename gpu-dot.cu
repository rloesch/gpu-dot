#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>


int main(int argc, char const *argv[])
{
    /* Number of elements */
    const int nvals = 1024;

    /* Number of bytes to store nvals elements */
    const size_t sz = sizeof(double) * (size_t)nvals;

    /* Host vectors */
    double x[nvals], y[nvals];

    /* Device vectors */
    double *x_, *y_;

    /* Host result */
    double result = 0.0;


    /* Initialize vectors */
    for(int i=0; i < nvals; i++) {
        x[i] = 1.0;
        y[i] = 1.0;
    }

    printf("Initializing two vectors with %d elements (%d bytes) each\n", nvals, sz);

    /* Create the CUBLAS library context */
    cublasHandle_t h;
    cublasCreate(&h);

    /* Allocate memory for both vectors on the device */
    cudaMalloc( (void **)(&x_), sz);
    cudaMalloc( (void **)(&y_), sz);

    /* Copy the vectors from the host to the device */
    cudaMemcpy(x_, x, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(y_, y, sz, cudaMemcpyHostToDevice);

    /* Run the dot calculation on the device using CUBLAS */
    cublasDdot(h, nvals, x_, 1, y_, 1, &result);

    /* Print result */
    printf("%.3f\n", result);

    return 0;
}
