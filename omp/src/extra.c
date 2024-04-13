#include "include/extra.h"

int int_pow(int base, int exponent){
    int result = 1;
    for(int i = 0; i < exponent; i ++){
        result *= base;
    }
    return result;
}

int find_infinite(int_matrix2d* matrix){
    int r = matrix->shape.values[0];
    int c = matrix->shape.values[1];
    int max_val = matrix->values[0][0];
    #pragma omp parallel for reduction(max:max_val) collapse(2)
    for(int i = 0; i < r; i++){
        for(int j = 0; j < c; j++){
            if(max_val < matrix->values[i][j]){
                max_val = matrix->values[i][j];
            }
        }
    }
    return max_val + 1;
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

int max(int a, int b){
    if( a > b)
        return a;
    return b;
}

int min(int a, int b){
    int a_min = a < b;
    return (a * a_min) + (b * !a_min);
}

int get_num_threads(){
    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads =  omp_get_num_threads();
    }
    return num_threads;
}
