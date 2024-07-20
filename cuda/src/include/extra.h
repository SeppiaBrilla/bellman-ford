#include "data.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#ifndef _EXTRA_
#define _EXTRA_

void checkCudaErrors(cudaError_t err);

int int_pow(int base, int exponent);

int find_infinite(int* matrix, int N, cudaStream_t stream);

char* array_to_json(int_array* arr);

int mmax(int a, int b);

int mmin(int a, int b);

__global__ void Max_Sequential_Addressing_Shared(int* data, int data_size);

#endif // !_EXTRA_
