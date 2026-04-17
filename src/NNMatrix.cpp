#include "NNMatrix.h"
#include <iostream>

using namespace std;

NNMatrix::NNMatrix(int r, int c) {
    rows = r;
    cols = c;
    // 使用 new 分配内存，() 表示默认初始化为 0
    data = new float[r * c](); 
}

NNMatrix::~NNMatrix() {
    // 使用 delete 释放内存
    delete[] data; 
}

void mat_add(NNMatrix* m, NNMatrix* other) {
    if (m->rows != other->rows || m->cols != other->cols) {
        cout << "错误：矩阵加法维度不匹配！" << endl;
        return;
    }
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] += other->data[i];
    }
}

NNMatrix* mat_multiply(NNMatrix* a, NNMatrix* b) {
    if (a->cols != b->rows) {
        cout << "错误：矩阵乘法维度不匹配！" << endl;
        return nullptr;
    }
    NNMatrix* result = new NNMatrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
    return result;
}

NNMatrix* mat_transpose(NNMatrix* m) {
    NNMatrix* result = new NNMatrix(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[j * result->cols + i] = m->data[i * m->cols + j];
        }
    }
    return result;
}