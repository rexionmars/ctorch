#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NN_IMPLEMENTATION
#include "apollo.h"

typedef struct {
    Matrix a0, a1, a2;
    Matrix w1, b1;
    Matrix w2, b2;
} Xor;

Xor xor_alloc(void)
{
    Xor model;
    model.a0 = matrix_alloc(1, 2);
    model.w1 = matrix_alloc(2, 2);
    model.b1 = matrix_alloc(1, 2);
    model.a1 = matrix_alloc(1, 2);
    model.w2 = matrix_alloc(2, 1);
    model.b2 = matrix_alloc(1, 1);
    model.a2 = matrix_alloc(1, 1);

    return model;
}

void forward_xor(Xor model)
{
    matrix_dot(model.a1, model.a0, model.w1);
    matrix_sum(model.a1, model.b1);
    matrix_sig(model.a1);

    matrix_dot(model.a2, model.a1, model.w2);
    matrix_sum(model.a2, model.b2);
    matrix_sig(model.a2);
}

float train_data[] = {
    0, 0 ,0,
    0, 1 ,1,
    1, 0 ,1,
    1, 1 ,0,
};


float cost(Xor model, Matrix train_input, Matrix train_output)
{
    assert(train_input.rows == train_output.rows);
    assert(train_output.colluns == model.a2.colluns);
    size_t start_interation = train_input.rows;

    float cost = 0;
    for (size_t i = 0; i < start_interation; ++i) {
        Matrix x = matrix_row(train_input, i);
        Matrix y = matrix_row(train_output, i);

        matrix_copy(model.a0, x);
        forward_xor(model);

        size_t q = train_output.colluns;
        for (size_t j = 0; j < q; ++j) {
            float diference = MATRIX_AT(model.a2, 0, j) - MATRIX_AT(y, 0, j);
            cost = diference * diference;
        }
    }

    return cost / start_interation;
}

void finite_difference(Xor model, Xor gate, float eps, Matrix train_input, Matrix train_output)
{
    float stored_value;
    float c = cost(model, train_input, train_output);

    for (size_t i = 0; i < model.w1.rows; ++i) {
        for (size_t j = 0; j < model.w1.rows; ++j) {
            stored_value = MATRIX_AT(model.w1, i, j);
            MATRIX_AT(model.w1, i, j) += eps;
            MATRIX_AT(gate.w1, i, j) = (cost(model, train_input, train_output) - c) / eps;
            MATRIX_AT(model.w1, i, j) = stored_value;
        }
    }

    for (size_t i = 0; i < model.b1.rows; ++i) {
        for (size_t j = 0; j < model.b1.rows; ++j) {
            stored_value = MATRIX_AT(model.w1, i, j);
            MATRIX_AT(model.b1, i, j) += eps;
            MATRIX_AT(gate.b1, i, j) = (cost(model, train_input, train_output) - c) / eps;
            MATRIX_AT(model.b1, i, j) = stored_value;
        }
    }

    for (size_t i = 0; i < model.w2.rows; ++i) {
        for (size_t j = 0; j < model.w2.rows; ++j) {
            stored_value = MATRIX_AT(model.w2, i, j);
            MATRIX_AT(model.w2, i, j) += eps;
            MATRIX_AT(gate.w2, i, j) = (cost(model, train_input, train_output) - c) / eps;
            MATRIX_AT(model.w2, i, j) = stored_value;
        }
    }

    for (size_t i = 0; i < model.b2.rows; ++i) {
        for (size_t j = 0; j < model.b2.rows; ++j) {
            stored_value = MATRIX_AT(model.b2, i, j);
            MATRIX_AT(model.b2, i, j) += eps;
            MATRIX_AT(gate.b2, i, j) = (cost(model, train_input, train_output) - c) / eps;
            MATRIX_AT(model.b2, i, j) = stored_value;
        }
    }
}

void xor_learn(Xor model, Xor gate, float rate)
{
    for (size_t i = 0; i < model.w1.rows; ++i) {
        for (size_t j = 0; j < model.w1.rows; ++j) {
            MATRIX_AT(model.w1, i, j) -= rate * MATRIX_AT(gate.w1, i, j);
        }
    }

    for (size_t i = 0; i < model.b1.rows; ++i) {
        for (size_t j = 0; j < model.b1.rows; ++j) {
            MATRIX_AT(model.b1, i, j) -= rate * MATRIX_AT(gate.b1, i, j);
        }
    }

    for (size_t i = 0; i < model.w2.rows; ++i) {
        for (size_t j = 0; j < model.w2.rows; ++j) {
            MATRIX_AT(model.w2, i, j) -= rate * MATRIX_AT(gate.w2, i, j);
        }
    }

    for (size_t i = 0; i < model.b2.rows; ++i) {
        for (size_t j = 0; j < model.b2.rows; ++j) {
            MATRIX_AT(model.b2, i, j) -= rate * MATRIX_AT(gate.b2, i, j);
        }
    }
}

int main(void)
{
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(train_data) / sizeof(train_data[0]) / stride;

    Matrix train_input = {
        .rows = n,
        .colluns = 2,
        .stride = stride,
        .elements_start = train_data
    };

    Matrix train_output = {
        .rows = n,
        .colluns = 1,
        .stride = stride,
        .elements_start = train_data + 2
    };

    size_t architecture[] = {2, 2, 1};
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

    for (size_t i=0; i < 2; ++i) {
      for (size_t j=0; j < i; ++j) {
        MATRIX_AT(NN_INPUT(nn), 0, 0) = i;
        MATRIX_AT(NN_INPUT(nn), 0, 0) = j;
        nn_forward(nn);
        print("%zu ^ %zu = %f\n", i, j, NN_OUTPUT(nn));
      }
    }

    return 0;
}
