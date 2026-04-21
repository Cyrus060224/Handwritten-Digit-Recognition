#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <vector>
#include <functional>

using namespace std;

struct NNMatrix {
    int rows, cols;
    vector<vector<double>> data;

    NNMatrix(int r, int c);
    void randomize();

    static NNMatrix multiply(const NNMatrix& a, const NNMatrix& b);
    void add(const NNMatrix& other);
    static NNMatrix subtract(const NNMatrix& a, const NNMatrix& b);
    
    void multiply_elements(const NNMatrix& other);
    void map(function<double(double)> func);
    NNMatrix transpose() const;

    // 🌟 新增：生成 Inverted Dropout 所需的随机掩码矩阵
    static NNMatrix generate_dropout_mask(int r, int c, double keep_probability);
    
}; // <--- 注意这个分号绝对不能少！

#endif