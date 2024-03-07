#include "include/bellman-ford.h"
#include <stdlib.h>

void initialize(int* distance, int* predecessor, int INFINITE, int size, int source){
    for(int i = 0; i < size; i ++){
        distance[i] = INFINITE;
        predecessor[i] = -1;
    }
    distance[source] = 0;
}
void find_distances(graph* graph, int* distance, int* predecessor){

    int* changes = (int*)malloc(sizeof(int) * graph->nodes.size);
    int total_changes = 0;

    for(int steps = 0; steps < graph->nodes.size - 1; steps ++){

        total_changes = 0;
        #pragma omp parallel for
        for(int i = 0; i < graph->nodes.size; i++){
            changes[i] = 0;
        }

        #pragma omp parallel for 
        for(int i = 0; i < graph->edges.shape.values[0]; i ++){
            for(int j = 0; j < graph->edges.shape.values[1]; j++){
                int edge = graph->edges.values[i][j];
                int change_distance = (distance[i] + edge < distance[j] && edge != 0);

                distance[j] = ((distance[i] + edge) * change_distance) + (distance[j] * !change_distance);
                changes[i] += change_distance;
                predecessor[j] = (i * change_distance) + (predecessor[j] * !change_distance);
            }
        }
        #pragma omp barrier
        #pragma omp parallel for reduction(|:total_changes)
        for(int i = 0; i > graph->nodes.size; i++){
            total_changes += changes[i];
        }

        if(total_changes == 0){
            break;
        }
    }
    free(changes);
}

int find_negative_cycles(graph* graph, int* distance, int source){
    int negative_cycle = 0;
    for(int i = 0; i < graph->edges.shape.values[0]; i ++){
        for(int j = 0; j < graph->edges.shape.values[1]; j++){
            int edge = graph->edges.values[i][j];
            negative_cycle += distance[i] + edge < distance[j] && i != source && edge != 0;
        }
    }
    return negative_cycle;
}

bellman_ford_return* shortest_paths(graph* graph, int source){
    double t_start, t_stop;

    int* distance = (int*)malloc(sizeof(int)* graph->nodes.size);
    int* predecessor = (int*)malloc(sizeof(int)* graph->nodes.size);

    t_start = omp_get_wtime();

    const int INFINITE = find_infinite(&graph->edges);

    initialize(distance, predecessor, INFINITE, graph->nodes.size, source);

    find_distances(graph, distance, predecessor);

    t_stop = omp_get_wtime();

    bellman_ford_return* return_value = (bellman_ford_return*)malloc(sizeof(bellman_ford_return));

    int_array distances;
    distances.size = graph->nodes.size;
    distances.values = distance;

    int_array predecessors;
    predecessors.size = graph->nodes.size;
    predecessors.values = predecessor;

    return_value->distances = distances;
    return_value->predecessors = predecessors;
    return_value->negative_cycles = min(1, find_negative_cycles(graph, distance, source));
    return_value->time = t_stop - t_start;

    return  return_value;
}
