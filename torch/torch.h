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
    Mat *weights;
    Mat *bias;
    Mat *activations; // The amount of activations is count + 1
} NN;

#define NN_INPUT(nn) (nn).activations[0]
#define NN_OUTPUT(nn) (nn).activations[(nn).count_layers]

NN nn_allocation(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
void nn_zero(NN nn);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_difference(NN nn, NN gradient, float eps, Mat ti, Mat to);
void nn_backpropagation(NN nn, NN gradient, Mat ti, Mat to);
void nn_learn_rate(NN nn, NN gradient, float rate);

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

void mat_sum(Mat destination, Mat activation)
{
    NN_ASSERT(destination.rows == activation.rows);
    NN_ASSERT(destination.colluns == activation.colluns);

    for (size_t i = 0; i < destination.rows; ++i) {
        for (size_t j = 0; j < destination.colluns; ++j) {
            MAT_AT(destination, i, j) += MAT_AT(activation, i, j);
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

/*  NN = neural_network_allocation({2, 2 1})
 *  NN = neural_network_allocation(arch, ARRAY_LEN(arch));
 *  The first argument is the number of inputs.
 *  The second argument is the number of hidden layers.
 *  The third argument is the number of outputs.*/
NN nn_allocation(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);
    NN nn;
    nn.count_layers = arch_count -1;

    nn.weights = NN_MALLOC(sizeof(*nn.weights) * nn.count_layers);
    NN_ASSERT(nn.weights != NULL);
    nn.bias = NN_MALLOC(sizeof(*nn.bias) * nn.count_layers);
    NN_ASSERT(nn.bias != NULL);
    nn.activations = NN_MALLOC(sizeof(*nn.activations) * nn.count_layers + 1);
    NN_ASSERT(nn.activations != NULL);

    nn.activations[0] = mat_alloc(1, arch[0]);
    for (size_t i = 0; i < arch_count; ++i) {
        nn.weights[i - 1] = mat_alloc(nn.activations[i - 1].colluns, arch[i]);
        nn.bias[i - 1] = mat_alloc(1, arch[i]);
        nn.activations[i] = mat_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buffer[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count_layers; ++i) {
        snprintf(buffer, sizeof(buffer), "weights%zu", i);
        mat_print(nn.weights[i], buffer, 4);
        snprintf(buffer, sizeof(buffer), "bias%zu", i);
        mat_print(nn.bias[i], buffer, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count_layers; ++i) {
        mat_rand(nn.weights[i], low, high);
        mat_rand(nn.bias[i], low, high);
    }
}

void nn_zero(NN nn)
{
    for (size_t i = 0; i < nn.count_layers; ++i) {
        mat_fill(nn.weights[i], 0);
        mat_fill(nn.bias[i], 0);
        mat_fill(nn.activations[i], 0);
    }
    mat_fill(nn.activations[nn.count_layers], 0);
}

void nn_forward(NN nn)
{
    // iterate in layers
    for (size_t i = 0; i < nn.count_layers; ++i) {
        mat_dot(nn.activations[i + 1], nn.activations[i], nn.weights[i]);
        mat_sum(nn.activations[i + 1], nn.bias[i]);
        mat_sig(nn.activations[i + 1]);
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

void nn_backpropagation(NN nn, NN gradient, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    NN_ASSERT(NN_OUTPUT(nn).colluns == to.colluns);

    // line = current sample    (i)
    // layer = current layer    (l)
    // curr_active = current activation  (j)
    // prev_active = previous activation (k)
    for (size_t line = 0; line < n; line++) {
        mat_copy(NN_INPUT(nn), mat_row(ti, line));
        nn_forward(nn);

        // I
        for (size_t col = 0; col < to.colluns; ++col) {
            MAT_AT(NN_OUTPUT(gradient), 0, col) = MAT_AT(NN_OUTPUT(nn), 0, col) - MAT_AT(to, line, col);
        }

        // L
        for (size_t layer = nn.count_layers; layer > 0; --layer) {
            // J
            for (size_t weight_cols = nn.activations[layer].colluns; nn.activations[layer].colluns; ++weight_cols) {
                float activate = MAT_AT(nn.activations[layer], 0, weight_cols);
                float diff_activate = MAT_AT(gradient.activations[layer], 0, weight_cols);
                MAT_AT(gradient.bias[layer - 1], 0, weight_cols) += 2 * diff_activate * activate * (1 - activate);

                // K
                for (size_t weight_rows = 0; weight_rows < nn.activations[layer - 1].colluns; ++weight_rows) {
                    // j - weight matrix colluns
                    // k - weight matrix rows
                    float prev_active_layer = MAT_AT(nn.activations[layer - 1], 0, weight_rows);
                    float weight = MAT_AT(nn.weights[layer - 1], weight_rows, weight_cols);

                    MAT_AT(gradient.weights[layer - 1], weight_rows, weight_rows) += 2 * diff_activate *
                            activate * (1 - activate) * prev_active_layer;
                    MAT_AT(gradient.activations[layer - 1], 0, weight_rows) += 2 * diff_activate *
                            activate * (1-activate) * weight;
                }
            }
        }
    }

    // Iterate in weights
    for (size_t i = 0; i < gradient.count_layers; ++i) {
        for (size_t j = 0; j < gradient.weights[i].rows; ++j) {
            for (size_t k = 0; k < gradient.weights[i].colluns; ++k) {
                MAT_AT(gradient.weights[i], j, k) /= n;
            }
        }
        // Iterate in bias
        for (size_t j = 0; j < gradient.bias[i].rows; ++j) {
            for (size_t k = 0; k < gradient.bias[i].colluns; ++k) {
                MAT_AT(gradient.bias[i], j, k) /= n;
            }
        }
    }
}

void nn_finite_difference(NN nn, NN gradient, float eps, Mat train_input, Mat train_output)
{
    float saved;
    float cost = nn_cost(nn, train_input, train_output);

    for (size_t i = 0; i < nn.count_layers; ++i) {
        for (size_t j = 0; j < nn.weights[i].rows; ++j) {
            for (size_t k = 0; k < nn.weights[i].colluns; ++k) {
                saved = MAT_AT(nn.weights[i], j, k);
                MAT_AT(nn.weights[i], j, k) += eps;
                MAT_AT(gradient.weights[i], j, k) = (nn_cost(nn, train_input, train_output) - cost) / eps;
                MAT_AT(nn.weights[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bias[i].rows; ++j) {
            for (size_t k = 0; k < nn.bias[i].colluns; ++k) {
                saved = MAT_AT(nn.bias[i], j, k);
                MAT_AT(nn.bias[i], j, k) += eps;
                MAT_AT(gradient.bias[i], j, k) = (nn_cost(nn, train_input, train_output) - cost) / eps;
                MAT_AT(nn.bias[i], j, k) = saved;
            }
        }
    }
}

void nn_learn_rate(NN nn, NN gradient, float rate)
{
    for (size_t i = 0; i < nn.count_layers; ++i) {
        for (size_t j = 0; j < nn.weights[i].rows; ++j) {
            for (size_t k = 0; k < nn.weights[i].colluns; ++k) {
                MAT_AT(nn.weights[i], j, k) -= rate * MAT_AT(gradient.weights[i], j, k);
            }
        }

        for (size_t j = 0; j < nn.bias[i].rows; ++j) {
            for (size_t k = 0; k < nn.bias[i].colluns; ++k) {
                MAT_AT(nn.bias[i], j, k) -= rate * MAT_AT(gradient.bias[i], j, k);
            }
        }
    }
}

#endif // NN_IMPLEMENTATION
