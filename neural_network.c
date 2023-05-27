#include <stdio.h>
#include <time.h>

#define NN_IMPLEMENTATION
#include "neural_network.h"

int main(void)
{
    srand(time(0));
    Math a = mathematical_allocation(1, 2);
    math_rand(a, 5, 10);

    float id_date[4] = {
        1, 0,
        0, 1
    };

    Math b = { .rows = 2, .colluns = 2, .es = id_date };

    Math distance = mathematical_allocation(1, 2);

    math_print(a);
    printf("---------------\n");
    math_dot(distance, a, b);
    math_print(distance);

    return 0;
}
