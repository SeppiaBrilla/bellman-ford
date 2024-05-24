#include <stdio.h>
#include "include/graph.h"
#include "include/bellman-ford.h"
#include "include/extra.h"

int main(int argc, char** argv)
{
    if(argc < 2){
        printf("Error: provide a graph file to continue\n");
        return 1;
    }

    graph* g = load_from_file(argv[1]);
    bellman_ford_return* result_edges = find_distances_iterate_over_edges(g, 0);
    char* distances_edges = array_to_json(&result_edges->distances);
    char* predecessors_edges = array_to_json(&result_edges->predecessors);
    printf("{");
    printf("\"edges\":\n{\n\"distances\":%s,\n\"predecessors\":%s,\n", distances_edges, predecessors_edges);
    printf("\"negative_cycles\":%d,\n\"inf_time\":%f,\n\"init_time\":%f,\n\"relaxation_time\":%f,\n\"negative_cycles_time\":%f\n},",
           result_edges->negative_cycles, result_edges->infinite_time, result_edges->init_time, result_edges->relaxation_time, result_edges->negative_cycle_time);
    free(distances_edges);
    free(predecessors_edges);
    free(result_edges);
    bellman_ford_return* result_nodes = find_distances_iterate_over_nodes(g, 0);
    free(g);
    char* distances_nodes = array_to_json(&result_nodes->distances);
    char* predecessors_nodes = array_to_json(&result_nodes->predecessors);
    printf("\"nodes\":{\n\"distances\":%s,\n\"predecessors\":%s,\n", distances_nodes, predecessors_nodes);
    printf("\"negative_cycles\":%d,\n\"inf_time\":%f,\n\"init_time\":%f,\n\"relaxation_time\":%f,\n\"negative_cycles_time\":%f\n},",
           result_nodes->negative_cycles, result_nodes->infinite_time, result_nodes->init_time, result_nodes->relaxation_time, result_nodes->negative_cycle_time);
    printf("\"input_file\":\"%s\"\n}", argv[1]);

    return 0;
}
