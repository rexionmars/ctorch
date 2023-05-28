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

typedef struct {
    size_t rows;
    size_t colluns;
    size_t stride;
    float *elements_start;
} Math;

#define MATH_AT(m, i, j) (m).elements_start[(i)*(m).stride + (j)]

float rand_float(void);
float sigmoidf(float x);

Math mathematical_allocation(size_t rows, size_t colluns);
void math_fill(Math m, float x);
void math_rand(Math m, float low, float high);
Math math_row(Math m, size_t row);
void math_copy(Math distination, Math src);
void math_dot(Math destination, Math a, Math b);
void math_sum(Math destination, Math a);
void math_sig(Math m);
void math_print(Math m, const char *name);
#define MATH_PRINT(m) math_print(m, #m)

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

void math_print(Math m, const char *name)
{
    printf("%s = [\n", name);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.colluns; ++j) {
            printf("    %f ", MATH_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
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

#endif // NN_IMPLEMENTATION
