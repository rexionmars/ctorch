#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define NN_IMPLEMENTATION
#include "neural_network.h"

typedef struct {
    Math a0, a1, a2;

    Math w1, b1;
    Math w2, b2;
} Xor;

Xor xor_alloc(void)
{
    Xor model;
    model.a0 = mathematical_allocation(1, 2);
    model.w1 = mathematical_allocation(2, 2);
    model.b1 = mathematical_allocation(1, 2);
    model.a1 = mathematical_allocation(1, 2);
    model.w2 = mathematical_allocation(2, 1);
    model.b2 = mathematical_allocation(1, 1);
    model.a2 = mathematical_allocation(1, 1);

    return model;
}

void forward_xor(Xor model)
{
    // First layer
    math_dot(model.a1, model.a0, model.w1);
    math_sum(model.a1, model.b1);
    math_sig(model.a1);

    // Second layer
    math_dot(model.a2, model.a1, model.w2);
    math_sum(model.a2, model.b2);
    math_sig(model.a2);
}

float train_date[] = {
    0, 0 ,0,
    0, 1 ,1,
    1, 0 ,1,
    1, 1 ,0,
};


float cost(Xor model, Math train_input, Math train_output)
{
    assert(train_input.rows == train_output.rows);
    assert(train_output.colluns == model.a2.colluns);
    size_t start_interation = train_input.rows;

    float cost = 0;
    for (size_t i = 0; i < start_interation; ++i) {
        Math x = math_row(train_input, i);
        Math y = math_row(train_output, i);

        math_copy(model.a0, x);
        forward_xor(model);

        size_t q = train_output.colluns;
        for (size_t j = 0; j < q; ++j) {
            float diference = MATH_AT(model.a2, 0, j) - MATH_AT(y, 0, j);
            cost = diference * diference;
        }
    }

    return cost / start_interation;
}

void finite_difference(Xor model, Xor gate, float eps, Math train_input, Math train_output)
{
    float stored_value;
    float c = cost(model, train_input, train_output);

    for (size_t i = 0; i < model.w1.rows; ++i){
        for (size_t j = 0; j < model.w1.rows; ++j){
            stored_value = MATH_AT(model.w1, i, j);
            MATH_AT(model.w1, i, j) += eps;
            MATH_AT(gate.w1, i, j) = (cost(model, train_input, train_output) - c) / eps;
            MATH_AT(model.w1, i, j) = stored_value;
        }
    }

    for (size_t i = 0; i < model.b1.rows; ++i){
        for (size_t j = 0; j < model.b1.rows; ++j){
            stored_value = MATH_AT(model.w1, i, j);
            MATH_AT(model.b1, i, j) += eps;
            MATH_AT(gate.b1, i, j) = (cost(model, train_input, train_output) - c) / eps;
            MATH_AT(model.b1, i, j) = stored_value;
        }
    }

    for (size_t i = 0; i < model.w2.rows; ++i){
        for (size_t j = 0; j < model.w2.rows; ++j){
            stored_value = MATH_AT(model.w2, i, j);
            MATH_AT(model.w2, i, j) += eps;
            MATH_AT(gate.w2, i, j) = (cost(model, train_input, train_output) - c) / eps;
            MATH_AT(model.w2, i, j) = stored_value;
        }
    }

    for (size_t i = 0; i < model.b2.rows; ++i){
        for (size_t j = 0; j < model.b2.rows; ++j){
            stored_value = MATH_AT(model.b2, i, j);
            MATH_AT(model.b2, i, j) += eps;
            MATH_AT(gate.b2, i, j) = (cost(model, train_input, train_output) - c) / eps;
            MATH_AT(model.b2, i, j) = stored_value;
        }
    }
}

void xor_learn(Xor model, Xor gate, float rate)
{
    for (size_t i = 0; i < model.w1.rows; ++i){
        for (size_t j = 0; j < model.w1.rows; ++j){
            MATH_AT(model.w1, i, j) -= rate * MATH_AT(gate.w1, i, j);
        }
    }

    for (size_t i = 0; i < model.b1.rows; ++i){
        for (size_t j = 0; j < model.b1.rows; ++j){
            MATH_AT(model.b1, i, j) -= rate * MATH_AT(gate.b1, i, j);
        }
    }

    for (size_t i = 0; i < model.w2.rows; ++i){
        for (size_t j = 0; j < model.w2.rows; ++j){
            MATH_AT(model.w2, i, j) -= rate * MATH_AT(gate.w2, i, j);
        }
    }

    for (size_t i = 0; i < model.b2.rows; ++i){
        for (size_t j = 0; j < model.b2.rows; ++j){
            MATH_AT(model.b2, i, j) -= rate * MATH_AT(gate.b2, i, j);
        }
    }
}

int main(void)
{
    srand(time(0));
    size_t stride = 3;
    size_t n = sizeof(train_date) / sizeof(train_date[0]) / stride;

    Math train_input = {
        .rows = n,
        .colluns = 2,
        .stride = stride,
        .elements_start = train_date
    };

    Math train_output = {
        .rows = n,
        .colluns = 1,
        .stride = stride,
        .elements_start = train_date + 2
    };

    Xor model = xor_alloc();
    Xor gate = xor_alloc();

    math_rand(model.w1, 0, 1);
    math_rand(model.b1, 0, 1);
    math_rand(model.w2, 0, 1);
    math_rand(model.b2, 0, 1);

    float eps = 1e-1;
    float rate = 1e-1;

    printf("COST: %f\n", cost(model, train_input, train_output));
    for (size_t i = 0; i < 10; ++i) {
        finite_difference(model, gate, eps, train_input, train_output);
        xor_learn(model, gate, rate);
        printf("%zu: COST %f\n",i , cost(model, train_input, train_output));
    }

    printf("-------------\n");
    #if 1
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            MATH_AT(model.a0, 0, 0) = i;
            MATH_AT(model.a0, 0, 0) = j;
            forward_xor(model);

            float y = *model.a2.elements_start;

            printf("%zu ^ %zu = %f\n", i, j, y);
        }
    }
    #endif

    return 0;
}
