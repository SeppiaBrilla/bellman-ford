#include "include/lists.h"
#include <stdio.h>
#include <stdlib.h>

void free_list(list* list){
    if(list == NULL){
        return;
    }

    list_element* next;
    list_element* current = list->head;
    for(int i = 0; i < list->length; i++){
        next = current->next;
        free(current);
        current = next;
    }
    if(current != NULL){
        perror("inconsistency detected: the given list has more elements than declered");
        exit(EXIT_FAILURE);
    }
}

void append(list* list, void* new_datum){
    struct list_element* new_element = (list_element*)malloc(sizeof(list_element));
    new_element->content = new_datum;
    new_element->next = NULL;
    if(list->length == 0){
        list->head = new_element;
        list->length ++;
        return;
    }
    list_element* element = list->head;
    for(int i = 0; i < list->length - 1; i++){
        element = element->next;
    }
    if(element->next != NULL){
        perror("inconsistency detected: the given list has more elements than declared");
        exit(EXIT_FAILURE);
    }
    element->next = new_element;
    list->length ++;
}

list* new_list(){
    list* l = (list*) malloc(sizeof(list));
    l->length = 0;
    l->head = NULL;
    return l;
}
