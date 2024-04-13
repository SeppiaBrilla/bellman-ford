#include "data.h"
#include <stdio.h>
#include <stdlib.h>
// types definition can be found in file data.c

#ifndef _LISTS_
#define _LISTS_

void free_list(list* list);

void append(list* list, void* new_datum);

list* new_list();

#endif // !_LISTS_

