#include "include/bellman-ford.h"
#include <omp.h>
#include <stdio.h>

void initialize(int* distance, int* predecessor, int INFINITE, int size, int source){
    #pragma omp parallel for
    for(int i = 0; i < size; i ++){
        distance[i] = INFINITE;
        predecessor[i] = -1;
    }
    distance[source] = 0;
}

void find_distances_nodes(graph* graph, int* distance, int* predecessor, int threads){

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

        #pragma omp barrier
        #pragma omp parallel for reduction(+:total_changes)
        for(int i = 0; i < threads; i++){
            total_changes += changes[i];
        }

        if(total_changes == 0){
            free(changes);
            return;
        }
    }
    free(changes);
}

void find_distances_edges(edge_array* edges, int* distance, int* predecessor, int max_steps, int threads){
    if(edges->size == 0)
        return;
    int* changes = (int*)malloc(sizeof(int) * threads);
    if(changes == NULL){
        printf("failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }

    int total_changes;
    for(int steps = 0; steps < max_steps; steps ++){
        
        total_changes = 0;
        memset(changes, 0, sizeof(int) * threads);
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            #pragma omp for
            for(int i = 0; i < edges->size; i ++){
                edge current_edge = edges->values[i];
                int change_distance = 0;
                #pragma omp critical
                {
                    change_distance = (distance[current_edge.source] + current_edge.weight < distance[current_edge.destination]);
                    int new_distance = ((distance[current_edge.source] + current_edge.weight) * change_distance) + (distance[current_edge.destination] * !change_distance);
                    distance[current_edge.destination] = new_distance;
                }
                changes[id] += change_distance;
                predecessor[current_edge.destination] = (current_edge.source * change_distance) + (predecessor[current_edge.destination] * !change_distance);
            }

            #pragma omp barrier
            #pragma omp parallel for reduction(+:total_changes)
            for(int i = 0; i < threads; i++){
                total_changes += changes[i];
            }
        }
        if(total_changes== 0){
            free(changes);
            return;
        }
    }
    free(changes);
}

int find_negative_cycles_nodes(graph* graph, int* distance, int source, int threads){
    int *negative_cycles = malloc(sizeof(int) * threads);

    memset(negative_cycles, 0, sizeof(int) * threads);

    #pragma omp parallel 
    {
        int id = omp_get_thread_num();
        #pragma omp for collapse(2)
        for(int i = 0; i < graph->edges.shape.values[0]; i ++){
            for(int j = 0; j < graph->edges.shape.values[1]; j++){
                int edge = graph->edges.values[i][j];
                negative_cycles[id] = edge != 0 && i != source && distance[i] + edge < distance[j];
            }
        }
    }

    int negative_cycle = 0;

    #pragma omp parallel reduction(+:negative_cycle)
    for(int i = 0; i < threads; i++){
        negative_cycle += negative_cycles[i];
    }

    free(negative_cycles);
    return negative_cycle;
}

int find_negative_cycles_edges(edge_array* edges, int* distance, int source, int threads){
    int *negative_cycles = malloc(sizeof(int) * threads);

    memset(negative_cycles, 0, sizeof(int) * threads);

    #pragma omp parallel 
    {
        int id = omp_get_thread_num();
        #pragma omp for 
        for(int i = 0; i < edges->size; i ++){
            edge edge = edges->values[i];
            negative_cycles[id] = distance[edge.source] + edge.weight < distance[edge.destination];
        }
    }

    int negative_cycle = 0;

    #pragma omp parallel reduction(+:negative_cycle)
    for(int i = 0; i < threads; i++){
        negative_cycle += negative_cycles[i];
    }

    free(negative_cycles);
    return negative_cycle;
}

bellman_ford_return* find_distances_iterate_over_nodes(graph* graph, int source){
    double t_start;

    int* distance = (int*)malloc(sizeof(int)* graph->nodes.size);
    int* predecessor = (int*)malloc(sizeof(int)* graph->nodes.size);
    int threads = get_num_threads();

    t_start = omp_get_wtime();
    const int INFINITE = find_infinite(&graph->edges);
    float inf_time = omp_get_wtime() - t_start;

    t_start = omp_get_wtime();
    initialize(distance, predecessor, INFINITE, graph->nodes.size, source);
    float init_time = omp_get_wtime() - t_start;

    t_start = omp_get_wtime();
    find_distances_nodes(graph, distance, predecessor, threads);
    float rel_time = omp_get_wtime() - t_start;
    
    t_start = omp_get_wtime();
    int negative_cycles = find_negative_cycles_nodes(graph, distance, source, threads);
    float neg_time = omp_get_wtime() - t_start;


    bellman_ford_return* return_value = (bellman_ford_return*)malloc(sizeof(bellman_ford_return));

    int_array distances;
    distances.size = graph->nodes.size;
    distances.values = distance;

    int_array predecessors;
    predecessors.size = graph->nodes.size;
    predecessors.values = predecessor;

    return_value->distances = distances;
    return_value->predecessors = predecessors;
    return_value->negative_cycles = min(1, negative_cycles);
    return_value->infinite_time = inf_time;
    return_value->init_time = init_time;
    return_value->relaxation_time = rel_time;
    return_value->negative_cycle_time = neg_time;

    return  return_value;
}

edge_array* get_edges(graph* graph){

    int n_edges = 0;

    for(int i = 0; i < graph->nodes.size; i++){
        for(int j = 0; j < graph->nodes.size; j++){
            n_edges += graph->edges.values[i][j] != 0;
        }
    }

    int current_edge = 0;
    edge* edges = malloc(sizeof(edge) * n_edges);
    {
        for(int i = 0; i < graph->nodes.size; i++){
            for(int j = 0; j < graph->nodes.size; j++){
                if(graph->edges.values[i][j] != 0){
                    edge e;
                    e.source = i;
                    e.destination = j;
                    e.weight = graph->edges.values[i][j];
                    edges[current_edge] = e;
                    current_edge ++;
                }
            }
        }
    }
    edge_array* edges_array = malloc(sizeof(edge_array));
    edges_array->size = n_edges;
    edges_array->values = edges;
    return edges_array;
}

bellman_ford_return* find_distances_iterate_over_edges(graph* graph, int source){
    double t_start;

    int* distance = (int*)malloc(sizeof(int)* graph->nodes.size);
    int* predecessor = (int*)malloc(sizeof(int)* graph->nodes.size);
    int threads = get_num_threads();

    edge_array* edges = get_edges(graph);
    int n_nodes = graph->nodes.size;


    t_start = omp_get_wtime();
    const int INFINITE = find_infinite(&graph->edges);
    float inf_time = omp_get_wtime() - t_start;
    
    t_start = omp_get_wtime();
    initialize(distance, predecessor, INFINITE, n_nodes, source);
    float init_time = omp_get_wtime() - t_start;

    t_start = omp_get_wtime();
    find_distances_edges(edges, distance, predecessor, n_nodes - 1, threads);
    float rel_time = omp_get_wtime() - t_start;

    t_start = omp_get_wtime();
    int negative_cycles = find_negative_cycles_edges(edges, distance, source, threads);
    float neg_time = omp_get_wtime() - t_start;

    bellman_ford_return* return_value = (bellman_ford_return*)malloc(sizeof(bellman_ford_return));

    int_array distances;
    distances.size = n_nodes;
    distances.values = distance;

    int_array predecessors;
    predecessors.size = n_nodes;
    predecessors.values = predecessor;

    return_value->distances = distances;
    return_value->predecessors = predecessors;
    return_value->negative_cycles = min(1, negative_cycles);
    return_value->init_time = init_time;
    return_value->infinite_time = inf_time;
    return_value->relaxation_time = rel_time;
    return_value->negative_cycle_time = neg_time;

    return  return_value;
}
