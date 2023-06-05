#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float x);

typedef struct {
    size_t rows;
    size_t colluns;
    size_t stride;
    float *elements_start;
} Math;

#define MATH_AT(m, i, j) (m).elements_start[(i)*(m).stride + (j)]

Math mathematical_allocation(size_t rows, size_t colluns);
void math_fill(Math m, float x);
void math_rand(Math m, float low, float high);
Math math_row(Math m, size_t row);
void math_copy(Math distination, Math src);
void math_dot(Math destination, Math a, Math b);
void math_sum(Math destination, Math a);
void math_sig(Math m);
void math_print(Math m, const char *name, size_t padding);
#define MATH_PRINT(m) math_print(m, #m, 0)

typedef struct {
    size_t count_layers;
    Math *ws;
    Math *bs;
    Math *as; // The amount of activations is count + 1
} NN;

NN nn_allocation(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

Math mathematical_allocation(size_t rows, size_t colluns)
{
    Math model;
    model.rows = rows;
    model.colluns = colluns;
    model.stride = colluns;
    model.elements_start = malloc(sizeof(*model.elements_start)*rows*colluns);
    NN_ASSERT(model.elements_start != NULL);
    return model;
}

void math_dot(Math destination, Math a, Math b)
{
    NN_ASSERT(a.colluns == b.rows);
    size_t n = a.colluns;
    NN_ASSERT(destination.rows == a.rows);
    NN_ASSERT(destination.colluns ==b.colluns);

    for (size_t i = 0; i < destination.rows; ++i) {
        for (size_t j = 0; j < destination.colluns; ++j) {
            MATH_AT(destination, i, j) = 0;
            for (size_t k = 0; k < n; ++k) {
                MATH_AT(destination, i, j) += MATH_AT(a, i, k) * MATH_AT(b, k, j);
            }
        }
    }
}

Math math_row(Math m, size_t row)
{
    return (Math) {
        .rows = 1,
        .colluns = m.colluns,
        .stride = m.stride,
        .elements_start = &MATH_AT(m, row, 0),
    };
}

void math_copy(Math destination, Math src)
{
    NN_ASSERT(destination.rows == src.rows);
    NN_ASSERT(destination.colluns == src.colluns);
    for (size_t i = 0; i < destination.rows; ++i) {
        for (size_t j = 0; j < destination.colluns; ++j){
            MATH_AT(destination, i, j) = MATH_AT(src, i, j);
        }
    }
}

void math_sum(Math destination, Math a)
{
    NN_ASSERT(destination.rows == a.rows);
    NN_ASSERT(destination.colluns == a.colluns);

    for (size_t i = 0; i < destination.rows; ++i) {
        for (size_t j = 0; j < destination.colluns; ++j) {
            MATH_AT(destination, i, j) += MATH_AT(a, i, j);
        }
    }
}

void math_sig(Math m)
{
    for (size_t i = 0; i < m.rows; ++i){
        for (size_t j = 0; j < m.colluns; ++j) {
            MATH_AT(m, i, j) = sigmoidf(MATH_AT(m, i, j));
        }
    }
}

void math_print(Math m, const char *name, size_t padding)
{
    printf("%*s%s = [\n",(int) padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < m.colluns; ++j) {
            printf("%f ", MATH_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int ) padding, "");
}

void math_fill(Math m, float x)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.colluns; ++j) {
            MATH_AT(m, i, j) = x;
        }
    }
}

void math_rand(Math m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.colluns; ++j) {
            MATH_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

//size_t arch[] = {2, 2, 1};
//NN nn = neural_network_allocation(arch, ARRAY_LEN(arch));
/*  NN nn = neural_network_allocation({2, 2 1})
 *  The first argument is the number of inputs.
 *  The second argument is the number of hidden layers.
 *  The third argument is the number of outputs. */

NN nn_allocation(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);
    NN nn;
    nn.count_layers = arch_count -1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count_layers);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count_layers);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as) * nn.count_layers + 1);
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mathematical_allocation(1, arch[0]);
    for (size_t i = 0; i < arch_count; ++i) {
        nn.ws[i - 1] = mathematical_allocation(nn.as[i - 1].colluns, arch[i]);
        nn.bs[i - 1] = mathematical_allocation(1, arch[i]);
        nn.as[i] = mathematical_allocation(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buffer[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count_layers; ++i) {
        snprintf(buffer, sizeof(buffer), "ws%zu", i);
        math_print(nn.ws[i], "ws", 4);
        math_print(nn.bs[i], "bs", 4);
    }
    printf("]\n");
}

#endif // NN_IMPLEMENTATION
