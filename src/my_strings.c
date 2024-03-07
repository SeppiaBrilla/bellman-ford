#include "include/my_strings.h"

char* str_concat(char* str1, char* str2){
    char* final = malloc(strlen(str1) + strlen(str2) + 1);
    strcpy(final, str1);
    strcat(final, str2);
    return final;
}

char* list_to_string(list* list){
    if(list->length == 0){
        return "";
    }

    char* string = malloc(list->length + 1);
    list_element* current = list->head;
    for(int i = 0; i < list->length; i++){
        char c = *(char*)current->content; 
        string[i] = c;
    }
    string[list->length + 1] = '\0';
    return string;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

string_array* split(const char* input, char split_char){
    int count = 0;
    for(size_t i = 0; input[i] != '\0'; ++i){
        if (input[i] == split_char){
            (count)++;
        }
    }

   char** lines = (char**)malloc((count + 1) * sizeof(char*));

    if (lines == NULL){
        perror("Error allocating memory");
        return NULL;
    }

    size_t lineIndex = 0;
    size_t lineStart = 0;
    for(size_t i = 0; input[i] != '\0'; ++i){
        if (input[i] == split_char){
            size_t lineLength = i - lineStart;

            lines[lineIndex] = (char*)malloc((lineLength + 1) * sizeof(char));
            if (lines[lineIndex] == NULL){
                perror("Error allocating memory");
                for(size_t j = 0; j < lineIndex; ++j){
                    free(lines[j]);
                }
                free(lines);
                return NULL;
            }
            strncpy(lines[lineIndex], &input[lineStart], lineLength);
            lines[lineIndex][lineLength] = '\0';

            lineIndex++;
            lineStart = i + 1;
        }
    }

    size_t lastLineLength = strlen(&input[lineStart]);
    lines[lineIndex] = (char*)malloc((lastLineLength + 1) * sizeof(char));
    if (lines[lineIndex] == NULL){
        perror("Error allocating memory");
        for(size_t j = 0; j <= lineIndex; ++j){
            free(lines[j]);
        }
        free(lines);
        return NULL;
    }
    strcpy(lines[lineIndex], &input[lineStart]);
    string_array* array = (string_array*)malloc(sizeof(string_array));
    array->size = count + 1;
    array->values = lines;
    return array;
}

int to_int(char* c){
    int final = 0;
    int current_digit = 0;
    int multiplier = 1;
    if(c[0] == '-'){
        multiplier = -1;
        c++;
    }
    int len = strlen(c);
    for(int i = 0; c[i] != '\0'; c++){
        current_digit = c[i] - '0';
        final += current_digit * int_pow(10, len - (i + 1));
    }
    return final * multiplier;
}

