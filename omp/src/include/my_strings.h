#include "data.h"
#include "extra.h"
#include <string.h>
#include <stdlib.h>

#ifndef _STRINGS_
#define _STRINGS_

char* str_concat(char* str1, char* str2);

string_array* split(const char* input, char split_char);

char* list_to_string(list* list);

int to_int(char* c);

#endif // !_STRINGS_
