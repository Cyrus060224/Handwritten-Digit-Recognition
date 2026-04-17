#pragma once
#include "NNMatrix.h"
#include <vector>
#include <string>

// 使用 C++ 的 vector 传递数组更加安全
void load_mnist_images(const std::string& filename, int count, std::vector<NNMatrix*>& images);
void load_mnist_labels(const std::string& filename, int count, std::vector<int>& labels);