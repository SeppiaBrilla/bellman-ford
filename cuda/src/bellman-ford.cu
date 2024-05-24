#include "include/bellman-ford.h"

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(err);
    }
}

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

__global__ void bf_iter_edge(edge* edges, int* starts, int* distance, int* predecessor, int* changes, int n){

    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int increment = blockDim.x * gridDim.x;
    for(int i = 0; i < n; i++){
        if(global_tid < starts[i+1]){
            for(int idx = starts[i] + global_tid; idx < starts[i+1]; idx+= increment){
                edge edge = edges[idx];
                if(distance[edge.source] + edge.weight < distance[edge.destination]){
                    distance[edge.destination] = distance[edge.source] + edge.weight;
                    *changes = 1;
                    predecessor[edge.destination] = edge.source;
                }
            }
        }
    }
}

void find_distances_edges(edge* edges, int* starts, int* distance, int* predecessor, int* neg_cycle, int n){

    int* d_changes;

    int* total_changes = (int*)malloc(sizeof(int));
    cudaMalloc((void**)&d_changes, sizeof(int));
    checkCudaErrors(cudaGetLastError());
    int steps = 0;
    *neg_cycle = 0;

    while(1){
        cudaMemset(d_changes, 0, sizeof(int));
        checkCudaErrors(cudaGetLastError());
        bf_iter_edge<<<n,n>>>(edges, starts, distance, predecessor, d_changes, n);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(total_changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost));
        if(total_changes[0] == 0)
            break;
        if(steps > n -1){
            *neg_cycle = 1;
            break;
        }
        steps ++;
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

bellman_ford_return* find_distances_iterate_over_nodes(graph* graph, int source){
    double t_start;

    cudaStream_t infinite_stream;
    cudaStreamCreate(&infinite_stream);

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
    const int INFINITE = find_infinite(d_edges, N * N, infinite_stream);
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

edge* get_edges(graph* graph, int* d_starts){

    int* starts = (int*)malloc(sizeof(int) * (graph->nodes.size + 1));

    int n_edges = 0;

    for(int i = 0; i < graph->nodes.size; i++){
        for(int j = 0; j < graph->nodes.size; j++){
            n_edges += graph->edges.values[i][j] != 0;
        }
    }
    
    int current_edge = 0;
    edge* edges = (edge*)malloc(sizeof(edge) * n_edges);
    edge* d_edges;

    for(int i = 0; i < graph->nodes.size; i++){
        starts[i] = current_edge;
        for(int j = 0; j < graph->nodes.size; j++){
            if(graph->edges.values[i][j] != 0){
                edges[current_edge].source = i;
                edges[current_edge].destination = j;
                edges[current_edge].weight = graph->edges.values[i][j];
                current_edge ++;
            }
        }
    }
    starts[graph->nodes.size] = current_edge;
    cudaMemcpy(d_starts, starts, sizeof(int) * (graph->nodes.size + 1), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_edges, sizeof(edge) * current_edge);
    cudaMemcpy(d_edges, edges, sizeof(edge) * current_edge, cudaMemcpyHostToDevice);
    return d_edges;
}

bellman_ford_return* find_distances_iterate_over_edges(graph* graph, int source){
    double t_start;

    cudaStream_t infinite_stream;
    cudaStreamCreate(&infinite_stream);

    int N = graph->nodes.size;
    int* distance = (int*)malloc(sizeof(int)* N);
    int* predecessor = (int*)malloc(sizeof(int)* N);

    int* d_distance;
    int* d_predecessor;
    int* d_edge_matrix;
    int* d_starts;

    cudaMalloc((void**)&d_distance, sizeof(int) * N);
    cudaMalloc((void**)&d_predecessor, sizeof(int) * N);
    cudaMalloc((void**)&d_starts, sizeof(int) * (N + 1));
    cudaMalloc((void**)&d_edge_matrix, sizeof(int) * graph->nodes.size * graph->nodes.size);

    unroll(graph->edges.values, d_edge_matrix, graph->nodes.size);
    edge* d_edges = get_edges(graph, d_starts);

    t_start = omp_get_wtime();
    const int INFINITE = find_infinite(d_edge_matrix, N * N, infinite_stream);
    checkCudaErrors(cudaGetLastError());
    float inf_time = omp_get_wtime() - t_start;
    t_start = omp_get_wtime();
    initialize<<<N,1>>>(d_distance, d_predecessor, INFINITE, 0);
    float init_time = omp_get_wtime() - t_start;

    checkCudaErrors(cudaGetLastError());
    cudaFreeAsync(d_edge_matrix, infinite_stream);
    checkCudaErrors(cudaGetLastError());

    int *neg_cycle = (int*)malloc(sizeof(int));
    t_start = omp_get_wtime();
    find_distances_edges(d_edges, d_starts, d_distance, d_predecessor, neg_cycle, N);
    float rel_time = omp_get_wtime() - t_start;

    bellman_ford_return* return_value = (bellman_ford_return*)malloc(sizeof(bellman_ford_return));

    memset(distance, 0, sizeof(int) * N);
    memset(predecessor, 0, sizeof(int) * N);
    cudaMemcpy(distance, d_distance, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(predecessor, d_predecessor, sizeof(int) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_distance);
    cudaFree(d_predecessor);
    cudaFree(d_edges);

    int_array distances;
    distances.size = N;
    distances.values = distance;

    int_array predecessors;
    predecessors.size = N;
    predecessors.values = predecessor;

    return_value->distances = distances;
    return_value->predecessors = predecessors;
    return_value->negative_cycles = *neg_cycle;
    return_value->init_time = init_time;
    return_value->infinite_time = inf_time;
    return_value->relaxation_time = rel_time;
    return_value->negative_cycle_time = 0;

    return  return_value;
}
