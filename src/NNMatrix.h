#pragma once

struct NNMatrix {
    int rows;
    int cols;
    float* data;

    // 构造函数 (替代 mat_create)
    NNMatrix(int r, int c);
    // 析构函数 (替代 mat_free)
    ~NNMatrix();
};

void mat_add(NNMatrix* m, NNMatrix* other);
NNMatrix* mat_multiply(NNMatrix* a, NNMatrix* b);
NNMatrix* mat_transpose(NNMatrix* m);