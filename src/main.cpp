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
                    // 你原代码里限制了最多训练 1000 张图，我们以此为总数
                    int total_samples = min(1000, (int)train_data.images.size()); 
                    
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
                
                // 2. 扫描包围盒 (寻找黑色墨水的上下左右边界)
                Color* pixels = LoadImageColors(img);
                int min_x = img.width, min_y = img.height, max_x = 0, max_y = 0;
                for (int y = 0; y < img.height; y++) {
                    for (int x = 0; x < img.width; x++) {
                        if (pixels[y * img.width + x].r < 240) { // 像素点不是纯白
                            if (x < min_x) min_x = x;
                            if (x > max_x) max_x = x;
                            if (y < min_y) min_y = y;
                            if (y > max_y) max_y = y;
                        }
                    }
                }
                UnloadImageColors(pixels);

                // 3. 计算有效墨水区域的宽和高
                int crop_w = max_x - min_x + 1;
                int crop_h = max_y - min_y + 1;

                // 防止用户没画东西就点识别导致崩溃
                if (crop_w > 0 && crop_h > 0 && min_x <= max_x) {
                    
                    // 4. 裁剪出只有墨水的核心区域
                    ImageCrop(&img, { (float)min_x, (float)min_y, (float)crop_w, (float)crop_h });

                    // 5. 等比例缩放，让最长的一边恰好是 20 像素 (留出 MNIST 标准的 4 像素 Padding)
                    float scale = 20.0f / max(crop_w, crop_h);
                    ImageResize(&img, (int)(crop_w * scale), (int)(crop_h * scale));

                    // 6. 创建一张 28x28 的全白底图，把缩放后的数字居中贴上去
                    Image final_img = GenImageColor(28, 28, WHITE);
                    int offset_x = (28 - img.width) / 2;
                    int offset_y = (28 - img.height) / 2;
                    ImageDraw(&final_img, img, 
                              { 0, 0, (float)img.width, (float)img.height }, 
                              { (float)offset_x, (float)offset_y, (float)img.width, (float)img.height }, 
                              WHITE);

                    // 7. 转为神经网络输入向量 (784x1，并进行颜色反转)
                    NNMatrix input(784, 1);
                    Color* final_pixels = LoadImageColors(final_img);
                    for (int i = 0; i < 784; i++) {
                        float brightness = (255.0f - final_pixels[i].r) / 255.0f; 
                        input.data[i][0] = brightness;
                    }
                    
                    // 释放内存
                    UnloadImageColors(final_pixels);
                    UnloadImage(final_img);
                    UnloadImage(img);

                    // 8. 喂给模型预测！
                    recognizedDigit = nn.predict(input);
                } else {
                    UnloadImage(img);
                    cout << "画板是空的，没法识别！" << endl;
                }
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