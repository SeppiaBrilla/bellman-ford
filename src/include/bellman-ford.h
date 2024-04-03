#include "data.h"
#include "extra.h"
#include "lists.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#ifndef _BELLMAN_
#define _BELLMAN_
typedef struct bf_retrurn{
    int_array distances;
    int_array predecessors;
    int negative_cycles;
    float time;
} bellman_ford_return;

bellman_ford_return* find_distances_iterate_over_nodes(graph* graph, int source);
bellman_ford_return* find_distances_iterate_over_edges(graph* graph, int source);

#endif // !_BELLMAN_
