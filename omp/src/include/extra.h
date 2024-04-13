#include "data.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#ifndef _EXTRA_
#define _EXTRA_

int int_pow(int base, int exponent);

int find_infinite(int_matrix2d* matrix);

char* array_to_json(int_array* arr);

int max(int a, int b);

int min(int a, int b);

int get_num_threads();

#endif // !_EXTRA_
