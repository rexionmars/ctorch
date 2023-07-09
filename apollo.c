#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NN_IMPLEMENTATION
#include "apollo.h"

float xor_train_data[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float or_train_data[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 1,
};

int main(void)
{
    srand(time(0));

    float *td = or_train_data;

    size_t stride = 3;
    size_t n = 4;

    Matrix train_input = {
        .rows = n,
        .colluns = 2,
        .stride = stride,
        .elements_start = td
    };

    Matrix train_output = {
        .rows = n,
        .colluns = 1,
        .stride = stride,
        .elements_start = td + 2
    };

    size_t architecture[] = {2, 4, 1};
    NN nn = nn_allocation(architecture, ARRAY_LEN(architecture));
    NN gate = nn_allocation(architecture, ARRAY_LEN(architecture));
    nn_rand(nn, 0, 1);

    float eps = 1e-1;
    float rate = 1e-1;

    printf("COST: %f\n", nn_cost(nn, train_input, train_output));
    for (size_t i = 0;i < 1000; ++i) {
      nn_finite_difference(nn, gate, eps, train_input, train_output);
      nn_learn_rate(nn, gate, rate);
      printf("%zu: COST: %f\n", i, nn_cost(nn, train_input, train_output));
    }

    // Print neural network
    NN_PRINT(nn);

    for (size_t i=0; i < 2; ++i) {
      for (size_t j=0; j < 2; ++j) {
        MATRIX_AT(NN_INPUT(nn), 0, 0) = i;
        MATRIX_AT(NN_INPUT(nn), 0, 0) = j;
        nn_forward(nn);
          printf("%zu ^ %zu = %f\n", i, j, MATRIX_AT(NN_OUTPUT(nn), 0, 0));
      }
    }

    return 0;
}
