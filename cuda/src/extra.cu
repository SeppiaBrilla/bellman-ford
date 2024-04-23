#include "include/extra.h"

int int_pow(int base, int exponent){
    int result = 1;
    for(int i = 0; i < exponent; i ++){
        result *= base;
    }
    return result;
}

char* array_to_json(int_array* arr) {
    char *json_string = (char*)malloc(((arr->size * 8) + 2) * sizeof(char)); 
    if (json_string == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    int pos = 0;
    pos += sprintf(json_string + pos, "[");
    for (int i = 0; i < arr->size; i++) {
        pos += sprintf(json_string + pos, "%d", arr->values[i]);
        if (i != arr->size - 1)
            pos += sprintf(json_string + pos, ",");
    }
    pos += sprintf(json_string + pos, "]");

    return json_string;
}

int mmax(int a, int b){
    if( a > b)
        return a;
    return b;
}

int mmin(int a, int b){
    int a_min = a < b;
    return (a * a_min) + (b * !a_min);
}

__global__ void Max_Sequential_Addressing_Shared(int* data, int data_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float sdata[];
    if(sdata[threadIdx.x] != 0)
        sdata[threadIdx.x] = 0;
    __syncthreads();
    if (idx < data_size){
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for(int stride=blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                float lhs = sdata[threadIdx.x];
                float rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
}

int find_infinite(int* matrix, int N){
    int* result_ptr = (int*)malloc(sizeof(int) * N);

    int* d_matrix;
    cudaMalloc((void**) &d_matrix, sizeof(int)* N);
    cudaMemcpy(d_matrix, matrix, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    Max_Sequential_Addressing_Shared<<<512,1024, 1024 * sizeof(int)>>>(d_matrix, N);
    cudaDeviceSynchronize();
    cudaMemcpy(result_ptr, d_matrix, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    
    int result = result_ptr[0];
    free(result_ptr);

    return result + 1;
}
