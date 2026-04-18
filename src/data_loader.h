#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "NNMatrix.h"
#include <string>
#include <vector>

using namespace std;

struct MNISTData {
    vector<NNMatrix> images;
    vector<NNMatrix> labels;
}; // <--- 之前可能就是漏了这个分号导致你报错

class DataLoader {
public:
    static MNISTData load_mnist(string image_path, string label_path);
private:
    static int reverse_int(int i);
}; // <--- 还有这个分号

#endif