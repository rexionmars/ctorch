#include <stdio.h>
#include <time.h>

#define NN_IMPLEMENTATION
#include "neural_network.h"

int main(void)
{
    srand(time(0));
    Math a = mathematical_allocation(3, 3);
    math_fill(a, 1);

    Math b = mathematical_allocation(3, 3);
    math_fill(b, 1);

    math_print(a);
    printf("---------------------\n");
    math_sum(a, b);
    math_print(a);

    return 0;
}
