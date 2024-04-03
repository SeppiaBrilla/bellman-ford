#ifndef DATA
#define DATA

typedef struct int_array{
    int size;
    int* values;
} int_array;

typedef struct string_array{
    int size;
    char** values;
} string_array;

typedef int_array tuple;

typedef struct matrix{
    tuple shape;
    int** values;
} int_matrix2d;

typedef struct graph{
    string_array nodes;
    int_matrix2d edges;
} graph;
typedef struct edge{
    int source;
    int destination;
    int weight;
} edge;

typedef struct list_element{
    struct list_element* next;
    void* content;
} list_element;

typedef struct list{
    struct list_element* head;
    int length;
} list;

#endif
