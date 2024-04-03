#include "include/bellman-ford.h"
#include <stdio.h>
#include <stdlib.h>

void initialize(int* distance, int* predecessor, int INFINITE, int size, int source){
    #pragma omp parallel for
    for(int i = 0; i < size; i ++){
        distance[i] = INFINITE;
        predecessor[i] = -1;
    }
    distance[source] = 0;
}

void find_distances_nodes(graph* graph, int* distance, int* predecessor){

    int threads = omp_get_num_threads();
    int* changes = (int*)malloc(sizeof(int) * threads);
    int total_changes = 0;

    for(int steps = 0; steps < graph->nodes.size - 1; steps ++){

        total_changes = 0;
        memset(changes, 0, sizeof(int) * threads);

        for(int i = 0; i < graph->edges.shape.values[0]; i ++){
            #pragma omp parallel
            {
                int id = omp_get_thread_num();
                #pragma omp for
                for(int j = 0; j < graph->edges.shape.values[1]; j++){
                    int edge = graph->edges.values[i][j];
                    int change_distance = ( edge != 0 && distance[i] + edge < distance[j]);
                    distance[j] = ((distance[i] + edge) * change_distance) + (distance[j] * !change_distance);
                    changes[id] += change_distance;
                    predecessor[j] = (i * change_distance) + (predecessor[j] * !change_distance);
                }
            }
        }
        #pragma omp parallel reduction(+:total_changes)
        for(int i = 0; i < graph->nodes.size; i++){
            total_changes += changes[i];
        }

        if(total_changes == 0){
            break;
        }
    }
    free(changes);
}

void find_distances_edges(list* edges, int* distance, int* predecessor, int max_steps){
    if(edges->length == 0)
        return;
    for(int steps = 0; steps < max_steps; steps ++){
        int changes = 0;
        list_element* current_element = edges->head;
        for(int i = 0; i < edges->length; i ++){
            edge* current_edge = (edge*)current_element->content;
            int change_distance = (distance[current_edge->source] + current_edge->weight < distance[current_edge->destination]);
            int new_distance = ((distance[current_edge->source] + current_edge->weight) * change_distance) + (distance[current_edge->destination] * !change_distance);
            distance[current_edge->destination] = new_distance;
            changes += change_distance;
            predecessor[current_edge->destination] = (current_edge->source * change_distance) + (predecessor[current_edge->destination] * !change_distance);
            if(current_element->next == NULL)
                break;
            current_element = current_element->next;
        }
        if(changes == 0)
            return;
    }
}

int find_negative_cycles(graph* graph, int* distance, int source){
    int *negative_cycles = malloc(sizeof(int) * graph->nodes.size);

    memset(negative_cycles, 0, sizeof(int) * graph->nodes.size);

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < graph->edges.shape.values[0]; i ++){
        for(int j = 0; j < graph->edges.shape.values[1]; j++){
            int edge = graph->edges.values[i][j];
            negative_cycles[j] = edge != 0 && i != source && distance[i] + edge < distance[j];
        }
    }

    int negative_cycle = 0;

    #pragma omp parallel reduction(+:negative_cycle)
    for(int i = 0; i < graph->nodes.size; i++){
        negative_cycle += negative_cycles[i];
    }

    free(negative_cycles);
    return negative_cycle;
}

bellman_ford_return* find_distances_iterate_over_nodes(graph* graph, int source){
    double t_start, t_stop;

    int* distance = (int*)malloc(sizeof(int)* graph->nodes.size);
    int* predecessor = (int*)malloc(sizeof(int)* graph->nodes.size);

    t_start = omp_get_wtime();

    const int INFINITE = find_infinite(&graph->edges);

    initialize(distance, predecessor, INFINITE, graph->nodes.size, source);

    find_distances_nodes(graph, distance, predecessor);

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

list* get_edges(graph* graph){

    list* edges = new_list();
    for(int i = 0; i < graph->nodes.size; i++){
        for(int j = 0; j < graph->nodes.size; j++){
            if(graph->edges.values[i][j] != 0){
                edge* current_edge = (edge*)malloc(sizeof(edge));
                current_edge->source = i;
                current_edge->destination = j;
                current_edge->weight = graph->edges.values[i][j];
                append(edges, current_edge);
            }
        }
    }
    return edges;
}

bellman_ford_return* find_distances_iterate_over_edges(graph* graph, int source){
    double t_start, t_stop;

    int* distance = (int*)malloc(sizeof(int)* graph->nodes.size);
    int* predecessor = (int*)malloc(sizeof(int)* graph->nodes.size);

    t_start = omp_get_wtime();

    const int INFINITE = find_infinite(&graph->edges);

    initialize(distance, predecessor, INFINITE, graph->nodes.size, source);
    list* edges = get_edges(graph);

    find_distances_edges(edges, distance, predecessor, graph->nodes.size - 1);

    t_stop = omp_get_wtime();

    free_list(edges);

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
