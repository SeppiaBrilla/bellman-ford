#include <stdio.h>
#include <stdlib.h>
#include "include/graph.h"
#include "include/bellman-ford.h"
#include "include/extra.h"

int main(int argc, char** argv){
    if(argc < 2){
        printf("Error: provide a graph file to continue\n");
        return 1;
    }
    graph* g = load_from_file(argv[1]);
    bellman_ford_return* result = shortest_paths(g, 0);
    char* distances = array_to_json(&result->distances);
    char* predecessors = array_to_json(&result->predecessors);
    printf("\{\n\"distances\":%s,\n\"predecessors\":%s,\n\"negative_cycles\":%d\n\"execution_time\":%f\n}", distances, predecessors, result->negative_cycles, result->time);
    printf("\n");
    free(g);
    free(distances);
    free(result);
    free(predecessors);
    return 0;
}
