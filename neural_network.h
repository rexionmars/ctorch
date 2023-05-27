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
    float *es;
} Math;

#define MATH_AT(m, i, j) (m).es[(i)*(m).colluns + (j)]

float rand_float(void);

Math mathematical_allocation(size_t rows, size_t colluns);
void math_fill(Math m, float x);
void math_rand(Math m, float low, float high);
void math_dot(Math distance, Math a, Math b);
void math_sum(Math distance, Math a);
void math_print(Math m);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

Math mathematical_allocation(size_t rows, size_t colluns)
{
    Math m;
    m.rows = rows;
    m.colluns = colluns;
    m.es = malloc(sizeof(*m.es)*rows*colluns);
    NN_ASSERT(m.es != NULL);
    return m;
}

void math_dot(Math distance, Math a, Math b)
{
    (void) distance;
    (void) a;
    (void) b;
}

void math_sum(Math distance, Math a)
{
    NN_ASSERT(distance.rows == a.rows);
    NN_ASSERT(distance.colluns == a.colluns);

    for (size_t i = 0; i < distance.rows; ++i) {
        for (size_t j = 0; j < distance.rows; ++j) {
            MATH_AT(distance, i, j) += MATH_AT(a, i, j);
        }
    }
}

void math_print(Math m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.colluns; ++j) {
            printf("%f ", MATH_AT(m, i, j));
        }
        printf("\n");
    }
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
