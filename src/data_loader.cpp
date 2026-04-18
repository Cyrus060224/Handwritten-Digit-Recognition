#include "data_loader.h"
#include <fstream>

using namespace std;

int DataLoader::reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

MNISTData DataLoader::load_mnist(string image_path, string label_path) {
    MNISTData data;
    ifstream img_file(image_path, ios::binary);
    ifstream lbl_file(label_path, ios::binary);

    if (img_file.is_open() && lbl_file.is_open()) {
        int magic = 0, count = 0, rows = 0, cols = 0;
        
        img_file.read((char*)&magic, 4);
        img_file.read((char*)&count, 4);
        count = reverse_int(count);
        img_file.read((char*)&rows, 4);
        img_file.read((char*)&cols, 4);

        lbl_file.read((char*)&magic, 4);
        lbl_file.read((char*)&magic, 4); 

        for (int i = 0; i < count; i++) {
            NNMatrix img(784, 1);
            for (int r = 0; r < 784; r++) {
                unsigned char pixel = 0;
                img_file.read((char*)&pixel, 1);
                img.data[r][0] = (double)pixel / 255.0; 
            }
            data.images.push_back(img);

            unsigned char label = 0;
            lbl_file.read((char*)&label, 1);
            NNMatrix lbl(10, 1);
            lbl.data[(int)label][0] = 1.0;
            data.labels.push_back(lbl);
        }
    }
    return data;
};