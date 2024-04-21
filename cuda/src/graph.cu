#include "include/graph.h"
#include <stdio.h>
#include <stdlib.h>

graph* load_from_file(char* file_name){
    FILE* file;
    file = fopen(file_name,"r");
    if(NULL == file){
        perror(str_concat((char*)"cannot open file ", file_name));
        exit(EXIT_FAILURE);
    }
 
    fseek(file, 0, SEEK_END);
    int length = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the characters
    char* characters = (char*)malloc((length + 1) * sizeof(char));

    // Check if memory allocation was successful
    if (characters == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }

    // Read characters from the file into the array
    fread(characters, sizeof(char), length, file);

    // Add a null terminator at the end of the array
    characters[length] = '\0';

    // Close the file
    fclose(file);
    string_array* lines = split(characters, '\n');
    string_array* nodes = split(lines->values[0], ',');

    int** edges = (int**)malloc(sizeof(int*) * nodes->size);
    for(int i = 1; i < lines->size; i++){
        int* ints = (int*)malloc(sizeof(int) * nodes->size);
        string_array* current_edeges_str = split(lines->values[i], ',');
        for(int j = 0; j < current_edeges_str->size; j++){
            ints[j] = to_int(current_edeges_str->values[j]);
        }
        edges[i-1] = ints;
    }
    int_matrix2d edge_matrix;
    tuple matrix_shape;
    matrix_shape.size = 2;
    int* values = (int*)malloc(sizeof(int) * 2);
    values[0] = nodes->size;
    values[1] = nodes->size;
    matrix_shape.values = values;
    edge_matrix.shape = matrix_shape;
    edge_matrix.values = edges;
    graph* g = (graph*)malloc(sizeof(graph));
    free(characters);
    free(lines);
    g->nodes = *nodes;
    g->edges = edge_matrix;
    return g;
}

