#include "data.h"
#include "extra.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef _BELLMAN_
#define _BELLMAN_
typedef struct bf_retrurn{
    int_array distances;
    int_array predecessors;
    int negative_cycles;
    float time;
} bellman_ford_return;

bellman_ford_return* shortest_paths(graph* graph, int source);

#endif // !_BELLMAN_
