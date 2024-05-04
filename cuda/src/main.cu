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

// "distances":[-1,24,35,24,40,3,43,12,37,18,40,4,49,40,22,33,4,26,35,47,11,36,17,30,35,6,25,39,46,17,8,14,7,18,11,11,23,22,24,35,17,19,21,0,6,37,47,16,16,41],
// "predecessors":[0,103,27,-71,-77,-35,60,20,64,-56,63,-100,29,25,-43,70,-111,9,-82,95,-126,58,2,24,-49,72,38,-105,31,-5,43,48,-11,-71,-24,-134,-10,-14,20,-75,-34,83,56,26,88,-34,23,48,-28,69],
