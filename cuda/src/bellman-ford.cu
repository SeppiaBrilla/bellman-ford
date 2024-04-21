#include "include/bellman-ford.h"
#include <cstdio>

template <unsigned int blockSize>
__global__ void cuda_reduce_sum(int *g_idata, int *g_odata, unsigned int n){
    __shared__ int sdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while(i < n){
        sdata[tid] += g_idata[i] + g_idata[i+blockSize]; 
        i += gridSize;
    }
    __syncthreads();
    if(blockSize >= 512){ 
        if(tid < 256){ 
            sdata[tid] += sdata[tid + 256]; 
        } 
        __syncthreads(); 
    }
    if(blockSize >= 256){
        if(tid < 128){
            sdata[tid] += sdata[tid + 128]; 
        }
        __syncthreads(); 
    }
    if(blockSize >= 128){
        if(tid < 64){
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if(tid < 32){
        if(blockSize >= 64) 
            sdata[tid] += sdata[tid + 32];
        if(blockSize >= 32) 
            sdata[tid] += sdata[tid + 16];
        if(blockSize >= 16) 
            sdata[tid] += sdata[tid + 8];
        if(blockSize >= 8) 
            sdata[tid] += sdata[tid + 4];
        if(blockSize >= 4) 
            sdata[tid] += sdata[tid + 2];
        if(blockSize >= 2) 
            sdata[tid] += sdata[tid + 1];
    }
    if(tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
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

__global__ void bf_iter(int* edges, int* distance, int* predecessor, int* changes, int n, int steps){

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

// void bellman_ford(int blocksPerGrid, int threadsPerBlock, int n, int *mat, int *dist, bool *has_negative_cycle) {
// 	dim3 blocks(blocksPerGrid);
// 	dim3 threads(threadsPerBlock);
//
// 	int iter_num = 0;
// 	int *d_mat, *d_dist;
// 	bool *d_has_next, h_has_next;
//
// 	cudaMalloc(&d_mat, sizeof(int) * n * n);
// 	cudaMalloc(&d_dist, sizeof(int) *n);
// 	cudaMalloc(&d_has_next, sizeof(bool));
//
//
// 	*has_negative_cycle = false;
//
// 	for(int i = 0 ; i < n; i ++){
// 		dist[i] = INF;
// 	}
//
// 	dist[0] = 0;
// 	cudaMemcpy(d_mat, mat, sizeof(int) * n * n, cudaMemcpyHostToDevice);
// 	cudaMemcpy(d_dist, dist, sizeof(int) * n, cudaMemcpyHostToDevice);
//
// 	for(;;){
// 		h_has_next = false;
// 		cudaMemcpy(d_has_next, &h_has_next, sizeof(bool), cudaMemcpyHostToDevice);
//
// 		bellman_ford_one_iter<<<blocks, threads>>>(n, d_mat, d_dist, d_has_next, iter_num);
// 		CHECK(cudaDeviceSynchronize());
// 		cudaMemcpy(&h_has_next, d_has_next, sizeof(bool), cudaMemcpyDeviceToHost);
//
// 		iter_num++;
// 		if(iter_num >= n-1){
// 			*has_negative_cycle = true;
// 			break;
// 		}
// 		if(!h_has_next){
// 			break;
// 		}
//
// 	}
// 	if(! *has_negative_cycle){
// 		cudaMemcpy(dist, d_dist, sizeof(int) * n, cudaMemcpyDeviceToHost);
// 	}
//
// 	cudaFree(d_mat);
// 	cudaFree(d_dist);
// 	cudaFree(d_has_next);
// }

void find_distances_nodes(int* edges, int* distance, int* predecessor, int n){
    printf("here\n");
    int* d_changes;
    int* d_changes_out;
    int* dist = (int*)malloc(sizeof(int) * n);

    unsigned int size = sizeof(int) * n;
    cudaMalloc((void**) &d_changes, size);
    cudaMalloc((void**) &d_changes_out, size);
    cudaMemset(d_changes_out, 0, size);
    int *total_changes = (int*)malloc(size);
    printf("here2\n");
    for(int steps = 0; steps < n - 1; steps ++){

        cudaMemset(d_changes, 0, size);
        cudaMemcpy(total_changes, d_changes, size, cudaMemcpyDeviceToHost);
        bf_iter<<<512, 1024>>>(edges, distance, predecessor, d_changes, n, steps);
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(dist, distance, size, cudaMemcpyDeviceToHost);
        for(int i = 0; i < n; i++)
            printf("%d ", dist[i]);
        printf("\n");
        cuda_reduce_sum<1024><<<512,1024>>>(d_changes, d_changes_out, n);
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(total_changes, d_changes_out, size, cudaMemcpyDeviceToHost);
        printf("step: %d, changes: %d\n", steps, total_changes[0]);
        if(total_changes[0] == 0)
            break;
    }
    printf("out!");
    free(total_changes);
    cudaFree(d_changes);
    cudaFree(d_changes_out);
}

// void find_distances_edges(edge_array* edges, int* distance, int* predecessor, int max_steps, int threads){
//     if(edges->size == 0)
//         return;
//     int* changes = (int*)malloc(sizeof(int) * threads);
//     if(changes == NULL){
//         printf("failed to allocate memory\n");
//         exit(EXIT_FAILURE);
//     }
//
//     int total_changes;
//     for(int steps = 0; steps < max_steps; steps ++){
//         
//         total_changes = 0;
//         memset(changes, 0, sizeof(int) * threads);
//         #pragma omp parallel
//         {
//             int id = omp_get_thread_num();
//             #pragma omp for
//             for(int i = 0; i < edges->size; i ++){
//                 edge current_edge = edges->values[i];
//                 int change_distance = 0;
//                 #pragma omp critical
//                 {
//                     change_distance = (distance[current_edge.source] + current_edge.weight < distance[current_edge.destination]);
//                     int new_distance = ((distance[current_edge.source] + current_edge.weight) * change_distance) + (distance[current_edge.destination] * !change_distance);
//                     distance[current_edge.destination] = new_distance;
//                 }
//                 changes[id] += change_distance;
//                 predecessor[current_edge.destination] = (current_edge.source * change_distance) + (predecessor[current_edge.destination] * !change_distance);
//             }
//
//             #pragma omp barrier
//             #pragma omp parallel for reduction(+:total_changes)
//             for(int i = 0; i < threads; i++){
//                 total_changes += changes[i];
//             }
//         }
//         if(total_changes== 0){
//             free(changes);
//             return;
//         }
//     }
//     free(changes);
// }

__global__ void check_negative_cycle(int** edges, int source, int* distance, int* negative_cycles, int N){
    int i = blockIdx.x;
    int j = threadIdx.x;
    if(i >= N || j >=N)
        return;
    int edge = edges[i][j];
    negative_cycles[j] = edge != 0 && i != source && distance[i] + edge < distance[j];
}

int find_negative_cycles_nodes(int** edges, int* distance, int source, unsigned int N){
    int *d_negative_cycles;
    int *d_negative_cycles_out;

    cudaMalloc(&d_negative_cycles, sizeof(int) * N);
    cudaMalloc(&d_negative_cycles_out, sizeof(int) * N);

    cudaMemset(d_negative_cycles, 0, sizeof(int) * N);

    check_negative_cycle<<<512,1024>>>(edges, source, distance, d_negative_cycles, N);
    int negative_cycle = 0;

    cuda_reduce_sum<1024><<<512,1024>>>(d_negative_cycles, d_negative_cycles_out, N);
    cudaMemcpy(d_negative_cycles_out, &negative_cycle, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_negative_cycles);
    cudaFree(d_negative_cycles_out);
    return negative_cycle;
}
//
// __global__ void internal_neg_cycles_edges(edge_array* edges, int* distance){
//     edge edge = edges->values[blockIdx.x];
// }
//
// int find_negative_cycles_edges(edge_array* edges, int* distance, int threads){
//     int *negative_cycles = (int*)malloc(sizeof(int) * threads);
//     int *negative_cycles_d;
//     cudaMalloc((void **)&negative_cycles_d, sizeof(int) * threads);
//     memset(negative_cycles, 0, sizeof(int) * threads);
//     cudaMemcpy(negative_cycles_d, negative_cycles, sizeof(int) * threads, cudaMemcpyHostToDevice);
//
//     #pragma omp parallel 
//     {
//         int id = omp_get_thread_num();
//         #pragma omp for 
//         for(int i = 0; i < edges->size; i ++){
//             edge edge = edges->values[i];
//             negative_cycles[id] = distance[edge.source] + edge.weight < distance[edge.destination];
//         }
//     }
//
//     int negative_cycle = 0;
//
//     #pragma omp parallel reduction(+:negative_cycle)
//     for(int i = 0; i < threads; i++){
//         negative_cycle += negative_cycles[i];
//     }
//
//     free(negative_cycles);
//     return negative_cycle;
// }
//

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
    printf("fino a qua tutto bene\n");
    unroll(graph->edges.values, d_edges, graph->nodes.size);

    t_start = omp_get_wtime();
    const int INFINITE = find_infinite(d_edges, N * N);
    float inf_time = omp_get_wtime() - t_start;
    printf("infinito: %d \n", INFINITE);
    t_start = omp_get_wtime();
    initialize<<<N,1>>>(d_distance, d_predecessor, INFINITE, source);
    float init_time = omp_get_wtime() - t_start;
    cudaMemcpy(distance, d_distance, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(predecessor, d_predecessor, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; i++){
        printf("%d ", distance[i]);
    }
    printf("\n");
    for(int i = 0; i < N; i++){
        printf("%d ", predecessor[i]);
    }
    printf("\n");
    printf("inizializzato\n");
    t_start = omp_get_wtime();
    printf("distanze...\n");
    find_distances_nodes(d_edges, d_distance, d_predecessor, N);
    printf("distanze");
    float rel_time = omp_get_wtime() - t_start;

    t_start = omp_get_wtime();
    // int negative_cycles = find_negative_cycles_nodes(d_edges, d_distance, source, N);
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
    return_value->negative_cycles = mmin(1, 0);
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
    edge* edges = (edge*)malloc(sizeof(edge) * n_edges);
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
    edge_array* edges_array = (edge_array*)malloc(sizeof(edge_array));
    edges_array->size = n_edges;
    edges_array->values = edges;
    return edges_array;
}
//
// bellman_ford_return* find_distances_iterate_over_edges(graph* graph, int source){
//     double t_start;
//
//     int N = graph->nodes.size;
//     int* distance = (int*)malloc(sizeof(int)* N);
//     int* predecessor = (int*)malloc(sizeof(int)* N);
//     int threads = get_num_threads();
//
//     edge_array* edges = get_edges(graph);
//     int n_nodes = graph->nodes.size;
//
//     int* INFINITE = (int*)malloc(sizeof(int));
//     t_start = omp_get_wtime();
//     find_infinite(&graph->edges, INFINITE);
//     float inf_time = omp_get_wtime() - t_start;
//     
//     t_start = omp_get_wtime();
//     initialize<<<N,1>>>(distance, predecessor, INFINITE, n_nodes);
//     float init_time = omp_get_wtime() - t_start;
//
//     t_start = omp_get_wtime();
//     find_distances_edges(edges, distance, predecessor, n_nodes - 1, threads);
//     float rel_time = omp_get_wtime() - t_start;
//
//     t_start = omp_get_wtime();
//     int negative_cycles = find_negative_cycles_edges(edges, distance, source, threads);
//     float neg_time = omp_get_wtime() - t_start;
//
//     bellman_ford_return* return_value = (bellman_ford_return*)malloc(sizeof(bellman_ford_return));
//
//     int_array distances;
//     distances.size = n_nodes;
//     distances.values = distance;
//
//     int_array predecessors;
//     predecessors.size = n_nodes;
//     predecessors.values = predecessor;
//
//     return_value->distances = distances;
//     return_value->predecessors = predecessors;
//     return_value->negative_cycles = mmin(1, negative_cycles);
//     return_value->init_time = init_time;
//     return_value->infinite_time = inf_time;
//     return_value->relaxation_time = rel_time;
//     return_value->negative_cycle_time = neg_time;
//
//     return  return_value;
// }
