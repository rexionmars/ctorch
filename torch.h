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
} Mat;

#define MAT_AT(m, i, j) (m).elements_start[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t colluns);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat distination, Mat src);
void mat_dot(Mat destination, Mat a, Mat b);
void mat_sum(Mat destination, Mat a);
void mat_sig(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct {
    size_t count_layers;
    Mat *ws;
    Mat *bs;
    Mat *as; // The amount of activations is count + 1
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count_layers]

NN nn_allocation(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat in, Mat out);
void nn_finite_difference(NN nn, NN gate, float eps, Mat in, Mat out);
void nn_learn_rate(NN nn, NN gate, float rate);

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

Mat mat_alloc(size_t rows, size_t colluns)
{
    Mat model;
    model.rows = rows;
    model.colluns = colluns;
    model.stride = colluns;
    model.elements_start = malloc(sizeof(*model.elements_start)*rows*colluns);
    NN_ASSERT(model.elements_start != NULL);
    return model;
}

void mat_dot(Mat destination, Mat a, Mat b)
{
    NN_ASSERT(a.colluns == b.rows);
    size_t n = a.colluns;
    NN_ASSERT(destination.rows == a.rows);
    NN_ASSERT(destination.colluns ==b.colluns);

    for (size_t i = 0; i < destination.rows; ++i) {
        for (size_t j = 0; j < destination.colluns; ++j) {
            MAT_AT(destination, i, j) = 0;
            for (size_t k = 0; k < n; ++k) {
                MAT_AT(destination, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

Mat mat_row(Mat m, size_t row)
{
    return (Mat) {
        .rows = 1,
        .colluns = m.colluns,
        .stride = m.stride,
        .elements_start = &MAT_AT(m, row, 0),
    };
}

void mat_copy(Mat destination, Mat src)
{
    NN_ASSERT(destination.rows == src.rows);
    NN_ASSERT(destination.colluns == src.colluns);
    for (size_t i = 0; i < destination.rows; ++i) {
        for (size_t j = 0; j < destination.colluns; ++j){
            MAT_AT(destination, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_sum(Mat destination, Mat a)
{
    NN_ASSERT(destination.rows == a.rows);
    NN_ASSERT(destination.colluns == a.colluns);

    for (size_t i = 0; i < destination.rows; ++i) {
        for (size_t j = 0; j < destination.colluns; ++j) {
            MAT_AT(destination, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i){
        for (size_t j = 0; j < m.colluns; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n",(int) padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int) padding, "");
        for (size_t j = 0; j < m.colluns; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int ) padding, "");
}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.colluns; ++j) {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.colluns; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
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

    nn.as[0] = mat_alloc(1, arch[0]);
    for (size_t i = 0; i < arch_count; ++i) {
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].colluns, arch[i]);
        nn.bs[i - 1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buffer[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count_layers; ++i) {
        snprintf(buffer, sizeof(buffer), "ws%zu", i);
        mat_print(nn.ws[i], buffer, 4);
        snprintf(buffer, sizeof(buffer), "bs%zu", i);
        mat_print(nn.bs[i], buffer, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count_layers; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn)
{
    // iterate in layers
    for (size_t i = 0; i < nn.count_layers; ++i) {
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}

float nn_cost(NN nn, Mat train_input, Mat train_output)
{
    NN_ASSERT(train_input.rows == train_output.rows);
    NN_ASSERT(train_output.colluns == NN_OUTPUT(nn).colluns);
    
    size_t n = train_input.rows;
    float cost = 0;

    for (size_t i = 0; i < n; ++i) {
        Mat x = mat_row(train_input, i);
        Mat y = mat_row(train_output, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);
        NN_OUTPUT(nn);

        size_t q = train_output.colluns;
        for (size_t j = 0; j < q; ++j) {
            float diference = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            cost = diference * diference;
        }
    }

    return cost / n;
}

void nn_finite_difference(NN nn, NN gate, float eps, Mat train_input, Mat train_output)
{
    float saved;
    float cost = nn_cost(nn, train_input, train_output);

    for (size_t i = 0; i < nn.count_layers; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].colluns; ++k) {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(gate.ws[i], j, k) = (nn_cost(nn, train_input, train_output) - cost) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].colluns; ++k) {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(gate.bs[i], j, k) = (nn_cost(nn, train_input, train_output) - cost) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_learn_rate(NN nn, NN gate, float rate)
{
    for (size_t i = 0; i < nn.count_layers; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].colluns; ++k) {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(gate.ws[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].colluns; ++k) {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(gate.bs[i], j, k);
            }
        }
    }
}

#endif // NN_IMPLEMENTATION
