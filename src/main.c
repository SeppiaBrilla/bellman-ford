#include <stdio.h>
#include "include/graph.h"
#include "include/bellman-ford.h"
#include "include/extra.h"

int main(int argc, char** argv){

    if(argc < 2){
        printf("Error: provide a graph file to continue\n");
        return 1;
    }

    graph* g = load_from_file(argv[1]);
    bellman_ford_return* result = find_distances_iterate_over_edges(g, 0);
    char* distances = array_to_json(&result->distances);
    char* predecessors = array_to_json(&result->predecessors);
    const char* out_string = "\{\n\"distances\":%s,\n\"predecessors\":%s,\n\"negative_cycles\":%d,\n\"execution_time\":%f,\n\"input_file\":\"%s\",\n\"cores\":%d\n}";
    printf(out_string, distances, predecessors, result->negative_cycles, result->time, argv[1], get_num_threads());
    printf("\n");

    return 0;
}
