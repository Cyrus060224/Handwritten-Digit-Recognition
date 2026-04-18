#include "NNMatrix.h"
#include <random>

using namespace std;

NNMatrix::NNMatrix(int r, int c) : rows(r), cols(c), data(r, vector<double>(c, 0.0)) {}

void NNMatrix::randomize() {
    static mt19937 gen(42); 
    normal_distribution<double> dist(0.0, 0.5); 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

NNMatrix NNMatrix::multiply(const NNMatrix& a, const NNMatrix& b) {
    NNMatrix result(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int k = 0; k < a.cols; k++) {
            for (int j = 0; j < b.cols; j++) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

void NNMatrix::add(const NNMatrix& other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] += other.data[i][j];
        }
    }
}

NNMatrix NNMatrix::subtract(const NNMatrix& a, const NNMatrix& b) {
    NNMatrix result(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return result;
}

void NNMatrix::multiply_elements(const NNMatrix& other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] *= other.data[i][j];
        }
    }
}

void NNMatrix::map(function<double(double)> func) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            this->data[i][j] = func(this->data[i][j]);
        }
    }
}

NNMatrix NNMatrix::transpose() const {
    NNMatrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = this->data[i][j];
        }
    }
    return result;
}