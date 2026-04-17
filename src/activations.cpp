#include "activations.h"
#include <cmath>
#include <algorithm>
#include "NNMatrix.h"

using namespace std;

void apply_activation(NNMatrix* m, Activation type) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        if (type == RELU) {
            m->data[i] = max(0.0f, m->data[i]);
        } else if (type == SIGMOID) {
            m->data[i] = 1.0f / (1.0f + exp(-m->data[i]));
        }
    }
}

NNMatrix* apply_derivative(NNMatrix* m, Activation type) {
    NNMatrix* result = new NNMatrix(m->rows, m->cols);
    for (int i = 0; i < m->rows * m->cols; i++) {
        if (type == RELU) {
            result->data[i] = m->data[i] > 0 ? 1.0f : 0.0f;
        } else if (type == SIGMOID) {
            result->data[i] = m->data[i] * (1.0f - m->data[i]);
        }
    }
    return result;
}