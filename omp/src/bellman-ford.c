#include "include/bellman-ford.h"

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

void find_distances_edges(edge_array* edges, int* distance, int* predecessor, int max_steps, int* negative_cycles){

    int total_changes;
    int steps = 0;
    while(1){
        
        total_changes = 0;
        for(int i = 0; i < max_steps + 1; i ++){
                #pragma omp parallel for
                for(int j = 0; j < edges[i].size; j++){
                    edge edge = edges[i].values[j];
                    if(distance[edge.source] + edge.weight < distance[edge.destination]){
                        distance[edge.destination] = distance[edge.source] + edge.weight;
                        total_changes = 1;
                        predecessor[edge.destination] = edge.source;
                    }
                }
        }

        steps ++;

        #pragma omp barrier
        if(total_changes== 0){
            break;
        }
        if(steps > max_steps){
            *negative_cycles = 1;
            break;
        }

    }
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

    int* n_edges = (int*)malloc(sizeof(int) * graph->nodes.size);
    memset(n_edges, 0, sizeof(int) * graph->nodes.size);

    for(int i = 0; i < graph->nodes.size; i++){
        for(int j = 0; j < graph->nodes.size; j++){
            n_edges[i] += graph->edges.values[i][j] != 0;
        }
    }

    int* current_edges = (int*)malloc(sizeof(int) * graph->nodes.size);
    memset(current_edges, 0, sizeof(int) * graph->nodes.size);

    edge_array* edges = (edge_array*)malloc(sizeof(edge_array) * graph->nodes.size);
    for(int i = 0; i < graph->nodes.size; i++){
        edges[i].values = (edge*)malloc(sizeof(edge) * n_edges[i]);
        edges[i].size = n_edges[i];
        for(int j = 0; j < graph->nodes.size; j++){
            if(graph->edges.values[i][j] != 0){
                edge e;
                e.source = i;
                e.destination = j;
                e.weight = graph->edges.values[i][j];
                edges[i].values[current_edges[i]] = e;
                current_edges[i] ++;
            }
        }
    }
    free(n_edges);
    free(current_edges);
    
    return edges;
}

bellman_ford_return* find_distances_iterate_over_edges(graph* graph, int source){
    double t_start;

    int* distance = (int*)malloc(sizeof(int)* graph->nodes.size);
    int* predecessor = (int*)malloc(sizeof(int)* graph->nodes.size);

    edge_array* edges = get_edges(graph);
    int n_nodes = graph->nodes.size;

    t_start = omp_get_wtime();
    const int INFINITE = find_infinite(&graph->edges);
    float inf_time = omp_get_wtime() - t_start;
    
    t_start = omp_get_wtime();
    initialize(distance, predecessor, INFINITE, n_nodes, source);
    float init_time = omp_get_wtime() - t_start;

    int* negative_cycles = (int*)malloc(sizeof(int));
    *negative_cycles = 0;
    t_start = omp_get_wtime();
    find_distances_edges(edges, distance, predecessor, n_nodes - 1, negative_cycles);
    float rel_time = omp_get_wtime() - t_start;

    bellman_ford_return* return_value = (bellman_ford_return*)malloc(sizeof(bellman_ford_return));

    int_array distances;
    distances.size = n_nodes;
    distances.values = distance;

    int_array predecessors;
    predecessors.size = n_nodes;
    predecessors.values = predecessor;

    return_value->distances = distances;
    return_value->predecessors = predecessors;
    return_value->negative_cycles = min(1, *negative_cycles);
    return_value->init_time = init_time;
    return_value->infinite_time = inf_time;
    return_value->relaxation_time = rel_time;
    return_value->negative_cycle_time = 0;

    return  return_value;
}
