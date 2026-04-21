# NoobNetwork - Handwritten Digit Recognition

> A pure C++ neural network framework with zero deep-learning framework dependencies, featuring an interactive GUI for real-time MNIST handwritten digit recognition and AutoML hyperparameter search.

---

## Highlights

| Feature | Detail |
|---|---|
| **Pure C++ Matrix Engine** | Entirely hand-written `NNMatrix` class -- no NumPy, no Eigen, no BLAS |
| **Zero Framework Dependency** | The neural network core has **zero** dependency on TensorFlow / PyTorch / any ML library |
| **Greedy Bold Driver** | Adaptive learning-rate strategy with 5% noise tolerance and automatic weight rollback |
| **AutoML** | Built-in K-Fold cross-validation with Grid / Random hyperparameter search |
| **Interactive GUI** | Real-time drawing canvas, live loss/accuracy curves, one-click recognition |

---

## Project Structure

```
Handwritten-Digit-Recognition/
├── CMakeLists.txt                  # Top-level CMake build configuration
├── README.md                       # This file
│
├── src/                            # Core source code
│   ├── main.cpp                    # Application entry point, GUI loop, training pipeline
│   ├── NNMatrix.h / NNMatrix.cpp   # Hand-written matrix engine (the heart of the project)
│   ├── neural_network.h / .cpp     # Neural network: forward pass, backprop, model I/O
│   ├── data_loader.h / .cpp        # MNIST IDX binary format parser
│   ├── activations.h               # Sigmoid, ReLU, Tanh and their derivatives
│   └── raygui.h                    # Single-header immediate-mode GUI library
│
├── data/                           # Dataset and resources
│   ├── train-images.idx3-ubyte     # MNIST training images (60,000 samples)
│   ├── train-labels.idx1-ubyte     # MNIST training labels
│   ├── t10k-images.idx3-ubyte      # MNIST test images (10,000 samples)
│   ├── t10k-labels.idx1-ubyte      # MNIST test labels
│   ├── ui_font.ttf                 # Custom HD anti-aliased font for the GUI
│   └── trained_model.txt           # Persisted model weights (generated after training)
│
└── raylib/                         # Local copy of raylib (graphics & windowing only)
```

---

## Pure C++ Handwritten Matrix Engine

The core of this project is the `NNMatrix` struct ([`NNMatrix.h`](file:///c:/Users/wxcyr/Desktop/Handwritten-Digit-Recognition/src/NNMatrix.h) / [`NNMatrix.cpp`](file:///c:/Users/wxcyr/Desktop/Handwritten-Digit-Recognition/src/NNMatrix.cpp)) -- a fully self-contained matrix computation engine written from scratch in standard C++11.

### What it provides

| Operation | Description |
|---|---|
| `multiply(a, b)` | Standard matrix multiplication (i-k-j loop order for cache efficiency) |
| `add(other)` | Element-wise in-place addition |
| `subtract(a, b)` | Element-wise subtraction, returns new matrix |
| `multiply_elements(other)` | Hadamard (element-wise) product |
| `map(func)` | Apply a scalar function to every element (activations, derivatives) |
| `transpose()` | Matrix transpose |
| `randomize()` | Xavier/Glorot initialization with variance scaled by fan-in |
| `generate_dropout_mask()` | Inverted Dropout mask generation with 1/p scaling |

### Why it matters

Most student-level neural network projects rely on NumPy (Python) or Eigen (C++). **NoobNetwork uses neither.** Every dot product, every gradient accumulation, every weight update goes through the custom `NNMatrix` engine backed only by `std::vector<std::vector<double>>`. This gives you:

- **Full transparency** -- you can trace every floating-point operation through the codebase
- **No external BLAS/LAPACK linking** -- the project compiles with a standard C++ compiler alone
- **Educational value** -- serves as a reference implementation for understanding how frameworks like PyTorch work under the hood

---

## Zero Framework Dependency

The neural network training pipeline -- including data loading, forward propagation, backpropagation, gradient accumulation, weight updates, model serialization, and the AutoML search engine -- depends on **nothing** beyond the C++ Standard Library:

```
Neural Network Core Dependencies:
  +-- <vector>       (std::vector for matrix storage)
  +-- <cmath>        (exp, tanh, sqrt)
  +-- <fstream>      (model save/load)
  +-- <random>       (Xavier init, dropout, data shuffling)
  +-- <functional>   (std::function for activation map)

NOT used:  TensorFlow, PyTorch, Eigen, OpenBLAS, cuDNN, etc.
```

The only third-party library in the entire project is **raylib**, which is used **exclusively** for window creation, canvas rendering, and GUI widgets. It has zero involvement in the neural network computation.

---

## Neural Network Architecture

### Supported Topology

The network accepts an arbitrary layer topology vector. For MNIST, the default is:

```
784 (input) → H1 (configurable 10–512) → [H2 (optional)] → 10 (output)
```

### Activation Functions ([`activations.h`](file:///c:/Users/wxcyr/Desktop/Handwritten-Digit-Recognition/src/activations.h))

- **Sigmoid** -- default, used on the output layer
- **ReLU** -- recommended for hidden layers
- **Tanh** -- alternative for hidden layers

### Optimizers

| Mode | Batch Size | Description |
|---|---|---|
| SGD | 1 | Stochastic gradient descent, per-sample updates |
| Mini-Batch | 128 | Gradient accumulation over 128 samples before update |
| BGD | Full dataset | Batch gradient descent over entire training set |

### Regularization

- **Inverted Dropout** -- configurable keep rate (0.5–1.0), applied during training only
- **Early Stopping** -- monitors smoothed loss with patience of 3 epochs

### Model Persistence

Models are saved as plain-text weight matrices to `data/trained_model.txt` and can be reloaded on startup for instant recognition without retraining.

---

## Greedy Bold Driver Adaptive Learning Rate (Requirement 10)

### Overview

The **Greedy Bold Driver** is an aggressive yet safe adaptive learning-rate strategy implemented directly in the training loop of [`main.cpp`](file:///c:/Users/wxcyr/Desktop/Handwritten-Digit-Recognition/src/main.cpp#L283-L327). It automatically adjusts the learning rate at every weight-update step based on whether the loss improved or worsened, eliminating the need for manual learning-rate scheduling.

### Algorithm Flow

```
Before each batch update:
  1. Save a full checkpoint of all layer weights  (backup_layers = layers)

After applying gradients:
  2. Compute smoothed loss:  S = 0.05 * current_batch_loss + 0.95 * S_prev

  3. Compare with historical best:
     
     IF  S <= previous_loss * 1.05    (loss decreased or within 5% tolerance):
         -- Accept the new weights (already applied)
         -- Increase learning rate:  lr *= 1.05  (with Greedy enabled)
                                    lr *= 1.01  (without Greedy)
         -- Update previous_loss = S
     
     ELSE  (loss increased beyond 5% tolerance -- "we took a bad step"):
         -- ROLLBACK weights from checkpoint:  layers = backup_layers
         -- Halve learning rate:  lr *= 0.5
         -- Restore smoothed loss:  S = previous_loss (undo the corrupted metric)
```

### Key Design Decisions

#### 5% Noise Tolerance (`* 1.05`)

The condition `smoothed_loss <= previous_loss * 1.05` allows the loss to fluctuate upward by up to **5%** before triggering a rollback. This is critical because:

- Mini-batch gradients are inherently noisy -- a slight loss increase does not mean the step was bad
- Without this tolerance, the algorithm would rollback far too frequently, preventing convergence
- The 5% threshold strikes a balance between aggressive exploration and stability

#### Weight Rollback Mechanism

The rollback is the defining feature of the "Bold" in Bold Driver:

```cpp
// Snapshot before update (main.cpp line 285-286)
if (enableGreedy && !isAutoSearching) {
    nn.save_checkpoint();   // backup_layers = layers (deep copy via vector assignment)
}

nn.apply_gradients();       // Apply the gradient step

// If loss worsened beyond tolerance (main.cpp line 311-315)
if (enableGreedy) {
    nn.load_checkpoint();       // layers = backup_layers (undo the weights entirely)
    nn.learningRate *= 0.5;     // Reduce step size for next attempt
    smoothed_loss = previous_loss;  // Undo the corrupted loss metric too
}
```

This means the network **never permanently accepts a destructive weight update**. The worst-case scenario is a halved learning rate with the previous safe weights intact.

#### Learning Rate Bounds

To prevent runaway values, the learning rate is clamped:

```
Lower bound: 0.0001
Upper bound: 0.1
```

#### Toggle Control

The strategy can be toggled on/off via the GUI checkbox "Greedy Bold Driver (Req.10)". When disabled, a conservative fallback strategy is used (`lr *= 1.01` on improvement, `lr *= 0.99` on degradation, no rollback).

---

## AutoML Hyperparameter Search

### Features

The project includes a built-in AutoML engine that automatically searches for the best hyperparameter configuration using **K-Fold Cross-Validation**.

### Search Strategies

| Strategy | Description |
|---|---|
| **Grid Search** | Evaluates the corner points of the search space: `(lr_min, h_min)` and `(lr_max, h_max)` |
| **Random Search** | Samples 3 random configurations within the specified ranges |

### Configurable Search Space

| Parameter | Range | Default |
|---|---|---|
| Learning Rate | LR Min -- LR Max (integers, divided by 100) | 0.01 -- 0.05 |
| Hidden Layer 1 Nodes | H Min -- H Max | 64 -- 256 |
| K-Fold Splits | 2 -- 10 | 5 |

### How It Works

1. **Space Generation** -- Grid or Random configurations are generated within user-specified ranges
2. **K-Fold Split** -- The 60,000 training samples are shuffled and partitioned into K folds (seeded with `mt19937(1337)`)
3. **Per-Configuration Training** -- Each configuration trains for 1 epoch on each fold's training split
4. **Validation Scoring** -- After each fold, the configuration is evaluated on the validation split and the accuracy is accumulated
5. **Best Selection** -- After all folds and all configurations complete, the best-performing configuration is reported

### UI Integration

The AutoML search runs asynchronously within the main render loop, with real-time progress display:

```
Running: [Grid LR:0.01 H:64] | Fold: 3/5
AUTOML PROGRESS: 12 / 15 FOLDS COMPLETED
```

---

## Image Preprocessing Pipeline

When recognizing a hand-drawn digit from the canvas, the image goes through a multi-stage preprocessing pipeline ([`main.cpp`](file:///c:/Users/wxcyr/Desktop/Handwritten-Digit-Recognition/src/main.cpp#L25-L83)) designed to match the MNIST data distribution:

1. **Bounding Box Detection** -- Scans for the minimum rectangle containing all ink pixels (threshold: R < 128)
2. **Aspect-Ratio-Preserving Resize** -- Scales the cropped digit so the larger dimension becomes 20px
3. **Center Padding** -- Places the resized digit onto a 28x28 white canvas, centered
4. **Normalization** -- Converts to grayscale float values in `[0, 1]` range (inverted: 255 - R)

This pipeline ensures that digits drawn at any position or size on the canvas are normalized to match the 28x28 MNIST format the network was trained on.

---

## Build Instructions

### Prerequisites

| Requirement | Version |
|---|---|
| C++ Compiler | MSVC (Visual Studio 2019+), MinGW-w64, or GCC/Clang on Linux/macOS |
| CMake | 3.10 or higher |

No other dependencies need to be installed -- raylib is included as a local subdirectory.

### Windows (MSVC / Visual Studio)

```powershell
# Create build directory
mkdir build
cd build

# Configure with CMake (UTF-8 encoding is enforced for MSVC)
cmake .. -G "Visual Studio 17 2022"

# Build (Release mode is the default)
cmake --build . --config Release

# Run
cd Release
.\NoobNetwork.exe
```

### Windows (MinGW-w64)

```powershell
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
.\NoobNetwork.exe
```

### Linux / macOS

```bash
mkdir build && cd build
cmake ..
cmake --build .
./NoobNetwork
```

### Build Notes

- **Release mode is enforced by default** in `CMakeLists.txt` (`set(CMAKE_BUILD_TYPE Release)`) -- no need to manually specify it
- The `data/` folder is automatically copied next to the executable during the build process via a `POST_BUILD` custom command, ensuring the dataset is always accessible at runtime
- C++11 standard is required (`set(CMAKE_CXX_STANDARD 11)`)

---

## Usage Guide

### Manual Training

1. Launch the application
2. Configure hyperparameters in the right panel:
   - **Activation**: Sigmoid / ReLU / Tanh
   - **Optimizer**: SGD / Mini-Batch / BGD
   - **Hidden L1**: Number of neurons in the first hidden layer (10--512)
   - **Hidden L2**: Optional second hidden layer (0 = disabled)
   - **Dropout**: Regularization rate (0.0 = off, up to 0.5)
   - **Max Epochs**: Training epochs per run
   - **Early Stop**: Automatically halts training if loss plateaus
   - **Greedy Bold Driver**: Enable the adaptive learning rate strategy
3. Click **START MANUAL TRAINING**
4. Watch the real-time loss (maroon) and accuracy (blue) curves at the bottom
5. Click **RUN PERFORMANCE TEST (10K)** to evaluate on the test set

### Handwritten Digit Recognition

1. After training (or if a pre-trained model exists at `data/trained_model.txt`), draw a digit on the canvas
2. Click **RECOGNIZE DIGIT**
3. The recognized digit appears in the status bar
4. Click **CLEAR CANVAS** to reset

### AutoML Search

1. Configure the search space:
   - **LR Range**: Min and Max learning rate (as integers, e.g. 1--5 = 0.01--0.05)
   - **H1 Range**: Min and Max hidden layer size
   - **Strategy**: Grid or Random search
   - **K-Fold**: Number of cross-validation folds
2. Click **RUN AUTO SEARCH**
3. Monitor progress in the status line
4. Best configuration is displayed upon completion

---

## Technical Details

### Backpropagation

The network implements standard backpropagation with MSE loss:

1. **Forward pass** -- computes activations layer by layer, storing `z` (pre-activation) and `a` (post-activation) for each layer
2. **Output error** -- `error = target - output`
3. **Backward pass** -- propagates errors backward through transposed weights, applying the appropriate activation derivative at each layer
4. **Gradient accumulation** -- accumulates weight and bias gradients across samples in a batch before applying the update

### Gradient Accumulation

The `accumulate_gradients()` / `apply_gradients()` pattern supports all three optimizer modes:
- **SGD**: `accumulated_samples` reaches 1 immediately, update after every sample
- **Mini-Batch**: Accumulates 128 samples, then averages and applies
- **BGD**: Accumulates all samples, applies once per epoch

### Data Loading

MNIST IDX binary files are parsed with proper big-endian byte order conversion ([`data_loader.cpp`](file:///c:/Users/wxcyr/Desktop/Handwritten-Digit-Recognition/src/data_loader.cpp#L6-L10)). Pixel values are normalized to `[0, 1]` and labels are one-hot encoded as 10-dimensional vectors.
