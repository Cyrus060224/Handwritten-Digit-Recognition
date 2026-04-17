#pragma once
#include "NNMatrix.h" // 必须加上这一行！

enum Activation { RELU, SIGMOID };

void apply_activation(NNMatrix* m, Activation type);
NNMatrix* apply_derivative(NNMatrix* m, Activation type);