#include <iostream>
#include <string>
#include "raylib.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "neural_network.h"
#include "data_loader.h"
#include <sys/stat.h>

using namespace std;

bool file_exists(const string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

// 将用户的随意手写图像转化为标准 MNIST 格式
Image preprocess_image(Image original_img) {
    Color* pixels = LoadImageColors(original_img);
    int minX = original_img.width, minY = original_img.height;
    int maxX = 0, maxY = 0;
    bool hasInk = false;

    // 1. 扫描寻找包含数字的最小边界框 (Bounding Box)
    for (int y = 0; y < original_img.height; y++) {
        for (int x = 0; x < original_img.width; x++) {
            Color c = pixels[y * original_img.width + x];
            if (c.r < 128) { // 黑色墨水（画板是白底黑字）
                hasInk = true;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }
    UnloadImageColors(pixels);

    if (!hasInk) return GenImageColor(28, 28, WHITE);

    // 2. 裁剪出只有数字的矩形
    Rectangle bbox = { (float)minX, (float)minY, (float)(maxX - minX + 1), (float)(maxY - minY + 1) };
    Image crop_img = ImageCopy(original_img);
    ImageCrop(&crop_img, bbox);

    // 3. 等比缩放到最大边长为 20
    int new_w, new_h;
    if (crop_img.width > crop_img.height) {
        new_w = 20;
        new_h = (int)((20.0f / crop_img.width) * crop_img.height);
    } else {
        new_h = 20;
        new_w = (int)((20.0f / crop_img.height) * crop_img.width);
    }
    if (new_w == 0) new_w = 1;
    if (new_h == 0) new_h = 1;
    ImageResize(&crop_img, new_w, new_h);

    // 4. 居中放置到一个全新的 28x28 白色画布上
    Image final_img = GenImageColor(28, 28, WHITE);
    int offsetX = (28 - new_w) / 2;
    int offsetY = (28 - new_h) / 2;
    Rectangle srcRec = { 0, 0, (float)new_w, (float)new_h };
    Rectangle dstRec = { (float)offsetX, (float)offsetY, (float)new_w, (float)new_h };
    
    ImageDraw(&final_img, crop_img, srcRec, dstRec, WHITE);
    UnloadImage(crop_img);

    return final_img;
}

int main() {
    const int screenWidth = 1000;
    const int screenHeight = 620;
    InitWindow(screenWidth, screenHeight, "Mini-Digits - Neural Network");
    SetTargetFPS(60);
    
    // 使用默认 UI 样式，摒弃外部字体依赖带来的不确定性
    GuiSetStyle(DEFAULT, TEXT_SIZE, 16);

    vector<int> topology = {784, 128, 10};
    NeuralNetwork nn(topology, 0.05);
    bool isModelLoaded = false; 
    string model_path = "data/trained_model.txt";
    
    if (file_exists(model_path)) {
        nn.load_model(model_path);
        isModelLoaded = true;
    }

    RenderTexture2D canvas = LoadRenderTexture(400, 400); 
    BeginTextureMode(canvas);
    ClearBackground(RAYWHITE); 
    EndTextureMode();

    float brushSize = 15.0f;
    int recognizedDigit = -1;       
    bool isTraining = false;        
    Rectangle drawingArea = { 40, 100, 400, 400 }; 

    while (!WindowShouldClose()) {
        Vector2 mousePos = GetMousePosition();
        
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) && CheckCollisionPointRec(mousePos, drawingArea)) {
            BeginTextureMode(canvas);
            Vector2 drawPos = { mousePos.x - drawingArea.x, mousePos.y - drawingArea.y };
            DrawCircleV(drawPos, brushSize, BLACK);
            EndTextureMode();
        }

        BeginDrawing();
        ClearBackground(GetColor(GuiGetStyle(DEFAULT, BACKGROUND_COLOR)));

        DrawText("Mini-Digits Neural Network System", 30, 20, 20, DARKGRAY);

        // --- Left Panel ---
        Rectangle canvasGroup = { 25, 75, 430, 440 };
        GuiGroupBox(canvasGroup, "Drawing Canvas");
        
        Rectangle sourceRec = { 0.0f, 0.0f, (float)canvas.texture.width, -(float)canvas.texture.height };
        DrawTextureRec(canvas.texture, sourceRec, { drawingArea.x, drawingArea.y }, WHITE);
        DrawRectangleLines(drawingArea.x, drawingArea.y, drawingArea.width, drawingArea.height, GRAY);

        // --- Right Top Panel ---
        Rectangle modelGroup = { 480, 80, 480, 220 };
        GuiGroupBox(modelGroup, "Model Configuration");

        GuiLabel({ 500, 110, 80, 30 }, "Status:");
        GuiLabel({ 580, 110, 250, 30 }, isModelLoaded ? "Loaded (trained_model.txt)" : "Not Ready (Needs Training)");

        GuiLabel({ 500, 160, 100, 30 }, "LR Policy:");
        GuiLabel({ 600, 160, 80, 30 }, "Adaptive"); 
        
        GuiLabel({ 700, 160, 80, 30 }, "Topology:");
        GuiLabel({ 780, 160, 150, 30 }, "784-128-10");

        if (GuiButton({ 500, 230, 440, 40 }, "Start Training (Check Terminal)")) {
            if (!isTraining) {
                cout << "Check terminal for progress..." << endl;
                string train_img_path = "data/train-images.idx3-ubyte";
                string train_lbl_path = "data/train-labels.idx1-ubyte";
                MNISTData train_data = DataLoader::load_mnist(train_img_path, train_lbl_path);
                
                if (!train_data.images.empty()) {
                    int epochs = 5;
                    // 移除 min() 限制，使用完整的数据集
                    int total_samples = train_data.images.size();
                    
                    cout << "\n开始训练模型... (共 " << epochs << " 轮，每轮 " << total_samples << " 个样本)" << endl;

                    for (int epoch = 0; epoch < epochs; epoch++) { 
                        cout << "Epoch " << epoch + 1 << "/" << epochs << "  ";
                        
                        for (int i = 0; i < total_samples; i++) {
                            // 核心训练逻辑（完全保持你的原样）
                            nn.train(train_data.images[i], train_data.labels[i]);
                            
                            // 🌟 终端进度条逻辑 🌟
                            // 每 50 个样本刷新一次，避免频繁打印拖慢速度
                            if (i % 50 == 0 || i == total_samples - 1) {
                                float progress = (float)(i + 1) / total_samples;
                                int barWidth = 40;
                                
                                cout << "[";
                                int pos = barWidth * progress;
                                for (int j = 0; j < barWidth; ++j) {
                                    if (j < pos) cout << "=";
                                    else if (j == pos) cout << ">";
                                    else cout << " ";
                                }
                                cout << "] " << int(progress * 100.0) << " %\r";
                                cout.flush(); // 强制刷新终端输出
                            }
                        }
                        cout << endl; // 每一轮跑完换行
                    }
                    
                    nn.save_model(model_path);
                    isModelLoaded = true;
                    cout << "训练完成并已保存模型！" << endl;
                }
            }
        }

        // --- Right Bottom Panel ---
        Rectangle paintGroup = { 480, 320, 480, 180 };
        GuiGroupBox(paintGroup, "Controls");

        GuiLabel({ 500, 350, 80, 20 }, "Brush:");
        GuiSlider({ 580, 350, 160, 20 }, "Thin", "Thick", &brushSize, 5.0f, 40.0f);

        if (GuiButton({ 500, 400, 200, 40 }, "Clear Canvas")) {
            BeginTextureMode(canvas);
            ClearBackground(RAYWHITE);
            EndTextureMode();
            recognizedDigit = -1;
        }

        if (GuiButton({ 740, 400, 200, 40 }, "Recognize")) {
            if (isModelLoaded) {
                // 1. 获取画板的原始图像
                Image img = LoadImageFromTexture(canvas.texture);
                
                // 🌟 解决 Raylib 倒影 BUG 的关键：垂直翻转图像！
                ImageFlipVertical(&img); 
                
                // 2. 调用刚才写好的预处理流水线
                Image final_img = preprocess_image(img);

                // 3. 转换为神经网络输入向量 (784x1)
                NNMatrix input(784, 1);
                Color* final_pixels = LoadImageColors(final_img);
                for (int i = 0; i < 784; i++) {
                    // 白底黑字转为黑底白字（1.0代表纯黑，0.0代表纯白）
                    float brightness = (255.0f - final_pixels[i].r) / 255.0f; 
                    input.data[i][0] = brightness;
                }
                
                // 4. 清理内存
                UnloadImageColors(final_pixels);
                UnloadImage(final_img);
                UnloadImage(img);

                // 5. 喂给模型预测
                recognizedDigit = nn.predict(input);
                
            } else {
                cout << "请先训练或加载模型！" << endl;
            }
        }

        // --- Result Display ---
        DrawRectangle(480, 520, 480, 80, LIGHTGRAY);
        DrawRectangleLines(480, 520, 480, 80, GRAY);
        
        if (recognizedDigit != -1) {
            string resText = "Prediction: " + to_string(recognizedDigit);
            DrawText(resText.c_str(), 500, 535, 40, DARKBLUE);
        } else {
            DrawText("Waiting for input...", 500, 545, 30, DARKGRAY);
        }

        EndDrawing();
    }

    UnloadRenderTexture(canvas);
    CloseWindow();
    return 0;
}