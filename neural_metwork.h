#ifndef NN_H_
#define NN_H_

typedef struct {
    size_t rows;
    size_t colluns;
    float *es;
} Math;

Math mathematical_allocation(size_t rows, size_t colluns);
void math_dot(Math distance, Math a, Math b);
void math_sum(Math distance, Math a);

#endif // NN_H_


#ifdef NN_IMPLEMENTATION

#endif // NN_IMPLEMENTATION
