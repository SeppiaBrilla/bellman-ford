#include "include/bellman-ford.h"
#include <cstdio>

void unroll(int** matrix, int* vector, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cudaMemcpy(&vector[(i * N)], matrix[i], sizeof(int) * N, cudaMemcpyHostToDevice);
        }
    }
}

__global__ void initialize(int* distance, int* predecessor, int INFINITE, int source){
    int i = blockIdx.x;
    if (i != source)
        distance[i] = INFINITE;
    else
        distance[i] = 0;
    predecessor[i] = -1;
}

__global__ void bf_iter_nodes(int* edges, int* distance, int* predecessor, int* changes, int n){

    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int increment = blockDim.x * gridDim.x;
    if(global_tid >= n)
        return;
    for(int i = 0; i < n; i++){
        for(int j = global_tid; j < n; j+= increment){
            int edge = edges[(i * n) + j];
            int change_distance = (edge != 0 && distance[i] + edge < distance[j]);
            distance[j] = ((distance[i] + edge) * change_distance) + (distance[j] * !change_distance);
            changes[j] += change_distance;
            predecessor[j] = (i * change_distance) + (predecessor[j] * !change_distance);
        }
    }
}

void find_distances_nodes(int* edges, int* distance, int* predecessor, int n){
    int* d_changes;

    unsigned int size = sizeof(int) * n;
    cudaMalloc((void**) &d_changes, size);
    int *total_changes = (int*)malloc(size);

    for(int steps = 0; steps < n - 1; steps ++){

        cudaMemset(d_changes, 0, size);
        bf_iter_nodes<<<512, 1024>>>(edges, distance, predecessor, d_changes, n);
        Max_Sequential_Addressing_Shared<<<512, 1024, 1024 * sizeof(int)>>>(d_changes, n);
        cudaMemcpy(total_changes, d_changes, size, cudaMemcpyDeviceToHost);

        if(total_changes[0] == 0)
            break;
    }

    free(total_changes);
    cudaFree(d_changes);
}

__global__ void bf_iter_edge(edge_array* edges, int* distance, int* predecessor, int* changes, int n){

    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int increment = blockDim.x * gridDim.x;
    for(int i = 0; i < n; i++){
        if(global_tid < *edges[i].size){
            // printf("(%d, %d)", *edges[i].size, global_tid);
            for(int idx = global_tid; idx < *edges[i].size; idx+= increment){
                edge edge = edges[i].values[idx];
                int change_distance = distance[edge.source] + edge.weight < distance[edge.destination];
                if(distance[edge.source] + edge.weight < distance[edge.destination])
                    changes[edge.destination] += 1;
                // printf("(%d, %d, %d) %d\n", edge.source, edge.destination, edge.weight, n);
                // distance[edge.destination] = ((distance[edge.source] + edge.weight) * change_distance) + (distance[edge.destination] * !change_distance);
                // changes[edge.destination] += change_distance;
                // printf("%d,%d\n", edges[i].values[idx].source,edges[i].values[idx].destination);
                // predecessor[edge.destination] = (edge.source * change_distance) + (predecessor[edge.destination] * !change_distance);
            }
        }
    }
}

void find_distances_edges(edge_array* edges, int* distance, int* predecessor, int n){

    int* d_changes;
    unsigned int size = sizeof(int) * n;

    int* total_changes = (int*)malloc(size);

    cudaMalloc((void**) &d_changes, size);

    for(int steps = 0; steps < n - 1; steps ++){
        cudaMemset(d_changes, 0, size);
        bf_iter_edge<<<n,n>>>(edges, distance, predecessor, d_changes, n);
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        // Max_Sequential_Addressing_Shared<<<512,  1024, 1024 * sizeof(int)>>>(d_changes, n);
        cudaMemcpy(total_changes, d_changes, size, cudaMemcpyDeviceToHost);
        if(total_changes[0] == 0)
            break;
    }
    free(total_changes);
    cudaFree(d_changes);
}

__global__ void check_negative_cycle(int* edges, int source, int* distance, int* negative_cycles, int N){
    int i = blockIdx.x;
    int j = threadIdx.x;
    if((i * N) + j >= N || j >= N || i >= N)
        return;
    int edge = edges[(i * N) + j];
    negative_cycles[j] = edge != 0 && i != source && distance[i] + edge < distance[j];
}

int find_negative_cycles_nodes(int* edges, int* distance, int source, unsigned int N){
    int *d_negative_cycles;

    cudaMalloc(&d_negative_cycles, sizeof(int) * N);

    cudaMemset(d_negative_cycles, 0, sizeof(int) * N);

    check_negative_cycle<<<512,1024>>>(edges, source, distance, d_negative_cycles, N);
    int negative_cycle = 0;

    Max_Sequential_Addressing_Shared<<<512,1024, 1024 * sizeof(int)>>>(d_negative_cycles, N);
    cudaMemcpy(&negative_cycle, d_negative_cycles, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_negative_cycles);
    return negative_cycle;
}

__global__ void internal_neg_cycles_edges(edge_array* edges, int* distance, int* negative_cycles, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= N)
        return;
    // edge edge = edges[i];
    // negative_cycles[edge.destination] = distance[edge.source] + edge.weight < distance[edge.destination];
}

int find_negative_cycles_edges(edge_array* edges, int* distance, int N, int size_edges){
    int *d_negative_cycles;
    int* negative_cycle = (int*)malloc(sizeof(int) * N);
    memset(negative_cycle, 0, sizeof(int) * N);

    cudaMalloc(&d_negative_cycles, sizeof(int) * N);

    cudaMemset(d_negative_cycles, 0, sizeof(int) * N);

    internal_neg_cycles_edges<<<512,1024>>>(edges, distance, d_negative_cycles, size_edges);
    Max_Sequential_Addressing_Shared<<<512,1024, 1024 * sizeof(int)>>>(d_negative_cycles, N);
    cudaMemcpy(negative_cycle, d_negative_cycles, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_negative_cycles);
    int neg = negative_cycle[0];
    free(negative_cycle);
    return neg;
}


bellman_ford_return* find_distances_iterate_over_nodes(graph* graph, int source){
    double t_start;

    const unsigned int N = graph->nodes.size;
    int* distance = (int*)malloc(sizeof(int)* N);
    int* predecessor = (int*)malloc(sizeof(int)* N);

    int* d_distance;
    int* d_predecessor;
    int* d_edges;

    cudaMalloc((void**)&d_distance, sizeof(int) * N);
    cudaMalloc((void**)&d_predecessor, sizeof(int) * N);
    cudaMalloc((void**)&d_edges, sizeof(int) * graph->nodes.size * graph->nodes.size);

    unroll(graph->edges.values, d_edges, graph->nodes.size);

    t_start = omp_get_wtime();
    const int INFINITE = find_infinite(d_edges, N * N);
    float inf_time = omp_get_wtime() - t_start;

    t_start = omp_get_wtime();
    initialize<<<N,1>>>(d_distance, d_predecessor, INFINITE, 0);
    float init_time = omp_get_wtime() - t_start;

    cudaMemcpy(distance, d_distance, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(predecessor, d_predecessor, sizeof(int) * N, cudaMemcpyDeviceToHost);

    t_start = omp_get_wtime();
    find_distances_nodes(d_edges, d_distance, d_predecessor, N);
    float rel_time = omp_get_wtime() - t_start;

    t_start = omp_get_wtime();
    int negative_cycles = find_negative_cycles_nodes(d_edges, d_distance, source, N);
    float neg_time = omp_get_wtime() - t_start;

    bellman_ford_return* return_value = (bellman_ford_return*)malloc(sizeof(bellman_ford_return));

    cudaMemcpy(distance, d_distance, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(predecessor, d_predecessor, sizeof(int) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_distance);
    cudaFree(d_predecessor);
    cudaFree(d_edges);

    int_array distances;
    distances.size = graph->nodes.size;
    distances.values = distance;

    int_array predecessors;
    predecessors.size = graph->nodes.size;
    predecessors.values = predecessor;

    return_value->distances = distances;
    return_value->predecessors = predecessors;
    return_value->negative_cycles = mmin(1, 0);
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
        edges[i].size = (int*)malloc(sizeof(int));
        *edges[i].size = n_edges[i];
        for(int j = 0; j < graph->nodes.size; j++){
            if(graph->edges.values[i][j] != 0){
                edges[i].values[current_edges[i]].source = i;
                edges[i].values[current_edges[i]].destination = j;
                edges[i].values[current_edges[i]].weight = graph->edges.values[i][j];
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

    int N = graph->nodes.size;
    int* distance = (int*)malloc(sizeof(int)* N);
    int* predecessor = (int*)malloc(sizeof(int)* N);

    int* d_distance;
    int* d_predecessor;
    int* d_edges;
    edge_array* d_edge_array;

    cudaMalloc((void**)&d_distance, sizeof(int) * N);
    cudaMalloc((void**)&d_predecessor, sizeof(int) * N);
    cudaMalloc((void**)&d_edges, sizeof(int) * graph->nodes.size * graph->nodes.size);

    unroll(graph->edges.values, d_edges, graph->nodes.size);

    edge_array* edges_array = get_edges(graph);
    cudaMalloc(&d_edge_array, sizeof(edge_array*) * N);
    edge_array host_array[N];
    for(int i = 0; i < N; i++){
        cudaMalloc(&host_array[i].values, sizeof(edge) * *edges_array[i].size);
        cudaMalloc(&host_array[i].size, sizeof(int));
        cudaMemcpy(host_array[i].values, edges_array[i].values, sizeof(edge) * *edges_array[i].size, cudaMemcpyHostToDevice); 
        cudaMemcpy(host_array[i].size, edges_array[i].size, sizeof(int), cudaMemcpyHostToDevice); 
    }

    cudaMemcpy(d_edge_array, host_array, sizeof(edge_array*) * N, cudaMemcpyHostToDevice);

    t_start = omp_get_wtime();
    const int INFINITE = find_infinite(d_edges, N * N);
    float inf_time = omp_get_wtime() - t_start;
    t_start = omp_get_wtime();
    initialize<<<N,1>>>(d_distance, d_predecessor, INFINITE, 0);
    float init_time = omp_get_wtime() - t_start;

    cudaFree(d_edges);
    
    t_start = omp_get_wtime();
    find_distances_edges(d_edge_array, d_distance, d_predecessor, N);
    float rel_time = omp_get_wtime() - t_start;

    t_start = omp_get_wtime();
    // int negative_cycles = find_negative_cycles_edges(d_edge_array, d_distance, N, edges_array->size);
    float neg_time = omp_get_wtime() - t_start;

    bellman_ford_return* return_value = (bellman_ford_return*)malloc(sizeof(bellman_ford_return));

    memset(distance, 0, sizeof(int) * N);
    memset(predecessor, 0, sizeof(int) * N);
    // cudaMemcpy(distance, d_distance, sizeof(int) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(predecessor, d_predecessor, sizeof(int) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_distance);
    cudaFree(d_predecessor);
    cudaFree(d_edge_array);
    for(int i = 0; i < N; i++){
        cudaFree(host_array[i].values);
        cudaFree(host_array[i].size);
    }
    printf("\n%s\n", cudaGetErrorString(cudaGetLastError()));

    int_array distances;
    distances.size = N;
    distances.values = distance;

    int_array predecessors;
    predecessors.size = N;
    predecessors.values = predecessor;

    return_value->distances = distances;
    return_value->predecessors = predecessors;
    // return_value->negative_cycles = mmin(1, negative_cycles);
    return_value->init_time = init_time;
    return_value->infinite_time = inf_time;
    return_value->relaxation_time = rel_time;
    return_value->negative_cycle_time = neg_time;

    return  return_value;
}
