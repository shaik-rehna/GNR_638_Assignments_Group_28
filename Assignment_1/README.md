# Custom Deep Learning Framework (C++ Backend + Python Frontend)

## Overview

This project implements a **custom deep learning framework from scratch** and uses it to train and evaluate a Convolutional Neural Network (CNN) for multiclass image classification.

The framework includes:

* Tensor abstraction with automatic differentiation
* Convolutional layers
* Activation functions
* Pooling layers
* Fully connected layers
* Cross-entropy loss
* SGD optimizer
* Model complexity analysis (Parameters, MACs, FLOPs)

The backend is implemented in **C++ (C++17)** and exposed to Python via **pybind11**.
All training and evaluation scripts use only this custom framework.

---

## Project Structure

```
my_framework/
│
├── backend/
│   │
│   ├── core/
│   │   ├── tensor.h
│   │   ├── tensor.cpp
│   │   ├── autograd.h
│   │   └── autograd.cpp
│   │
│   ├── layers/
│   │   ├── conv2d.h
│   │   ├── conv2d.cpp
│   │   ├── linear.h
│   │   ├── linear.cpp
│   │   ├── maxpool2d.h
│   │   └── maxpool2d.cpp
│   │
│   ├── ops/
│   │   ├── activation.h
│   │   ├── activation.cpp
│   │   ├── flatten.h
│   │   └── flatten.cpp
│   │
│   ├── loss/
│   │   ├── cross_entropy.h
│   │   └── cross_entropy.cpp
│   │
│   ├── optim/
│   │   ├── sgd.h
│   │   └── sgd.cpp
│   │
│   ├── bindings.cpp
│   ├── CMakeLists.txt
│   │
│   └── build/
│       └── my_framework.cpython-312-*.so
│
├── frontend/
│   │
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── dataloader.py
│   │
│   ├── configs/
│   │   ├── config_data_1.json
│   │   └── config_data_2.json
│
├── outputs/
│   │
│   ├── data_1/
│   │   ├── weights/
│   │   │   └── model_weights.json
│   │   ├── logs/
│   │   │   └── training_logs.json
│   │   └── console/
│   │       └── training_output.txt
│   │
│   └── data_2/
│       ├── weights/
│       │   └── model_weights.json
│       ├── logs/
│       │   └── training_logs.json
│       └── console/
│           └── training_output.txt
│
├── requirements.txt
└── .gitignore
```

## Model Architecture

The CNN architecture dynamically adapts based on the number of classes:

- ≤ 10 classes:
  - 2 Convolution layers (4 → 8 channels)
  - 1 MaxPool layer
  - Fully connected layer (32 hidden units)

- greater than 10 classes:
  - 2 Convolution layers (8 → 16 channels)
  - 1 MaxPool layer
  - Fully connected layer (64 hidden units)

All inputs are resized to 32×32 and use valid convolutions
(kernel = 3, stride = 1, no padding).


## Development Environment

This project was developed and tested on:

  * macOS (Apple Silicon / ARM64)

  * Apple Clang 17

  * CMake 4.x

* Python Version: 3.12.12  
* pybind11 Version: 3.0.1  
* OpenCV Version: 4.13.0


No external deep learning libraries (PyTorch, TensorFlow, NumPy, etc.) are used.

---

# Setup and Execution Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/shaik-rehna/GNR_638_Assignments_Group_28.git
cd GNR_638_Assignments_Group_28/Assignment_1
```

The project structure:

```
GNR_638_Assignments_Group_28/
└── Assignment_1/
    └── my_framework/
```

---

# 2. Python Setup

## Check Python 3.12

### Windows

```bash
py -3.12 --version
```

### macOS / Linux

```bash
python3.12 --version
```

If not installed, download from:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

(Ensure “Add Python to PATH” is enabled on Windows.)

---

## Create Virtual Environment

### Windows

```bash
py -3.12 -m venv gnr638_venv
```

### macOS / Linux

```bash
python3.12 -m venv gnr638_venv
```

---

## Activate Virtual Environment

### Windows (Command Prompt)

```bash
gnr638_venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
gnr638_venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source gnr638_venv/bin/activate
```

You should now see:

```
(gnr638_venv)
```

---

# 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 4. Build Backend (C++ Compilation) 

The precompiled `build/` directory included in this project was generated on macOS.

If running on a different operating system (e.g., Windows or Linux), or with a different Python version, the backend must be rebuilt.

* Rebuilding is required only if:

  * Backend source code is modified
  
  * Python version changes
  
  * Operating system changes (The existing `build/` directory was created on macOS)
  
  * Compiler/toolchain changes

If none of the above apply, the existing `build/` directory can be used directly.

Navigate to backend:

```bash
cd my_framework/backend
```

## Clean Previous Builds

### macOS / Linux

```bash
rm -rf build
mkdir build
cd build
```

### Windows (PowerShell)

```powershell
Remove-Item -Recurse -Force build
mkdir build
cd build
```

---

## Configure with CMake

```bash
cmake -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" ..
```

---

## Build

### macOS / Linux

```bash
make -j
```

### Windows

```powershell
cmake --build . --config Release
```

This generates:

* macOS/Linux:

```
my_framework/backend/build/my_framework.cpython-312-*.so
```

* Windows:

```
my_framework/backend/build/Release/my_framework.cp312-win_amd64.pyd
```

---

# 5. Train the Model

Navigate to the frontend directory:

```bash
cd my_framework/frontend
```

---

## Configuration Files

Configuration files for both datasets are available in:

```
my_framework/frontend/configs/
```

* `config_data_1.json`
* `config_data_2.json`

Each configuration file defines:

* Number of epochs
* Batch size
* Learning rate
* Validation split
* Random seed

---

## Training 

### Data 1
* macOS/Linux:

```
python -u train.py \
  --data_path /path/to/data_1 \
  --config_path configs/config_data_1.json | tee training_output_data_1.txt
```

* Windows(PowerShell):

```
python -u train.py `
  --data_path /path/to/data_1 `
  --config_path configs/config_data_1.json | Tee-Object training_output_data_1.txt
```
### Data 2

* macOS/Linux:

```
python -u train.py \
  --data_path /path/to/data_2 \
  --config_path configs/config_data_2.json | tee training_output_data_2.txt
```

* Windows(PowerShell):

```
python -u train.py `
  --data_path /path/to/data_2 `
  --config_path configs/config_data_2.json | Tee-Object training_output_data_2.txt
```
---

## During Training, the Script Displays

* Dataset loading time
* Model architecture configuration
* Total trainable parameters
* MACs per forward pass
* FLOPs per forward pass
* Training loss and accuracy
* Validation accuracy
* Epoch time
* Throughput (samples/sec)

---

# 6. Training Outputs

All outputs are saved inside:

```
my_framework/outputs/
```

### Data 1 Outputs

```
my_framework/outputs/data_1/
│
├── weights/
│   └── model_weights_data1.json
│
├── logs/
│   └── training_logs.json
│
└── console/
    └── training_output.txt
```

---

### Data 2 Outputs

```
my_framework/outputs/data_2/
│
├── weights/
│   └── model_weights_data2.json
│
├── logs/
│   └── training_logs.json
│
└── console/
    └── training_output.txt
```

Each dataset directory contains:

* Trained model weights
* Full training logs
* Console output of the training run

---

# 7. Evaluate the Model

Navigate to the frontend directory:

```bash
cd my_framework/frontend
```

---

## Evaluation 

### Data 1

```bash
python evaluate.py \
  --data_path /path/to/test_dataset \
  --weights_path ../outputs/data_1/weights/model_weights_data1.json \
  --batch_size 32
```

### Data 2

```bash
python evaluate.py \
  --data_path /path/to/test_dataset \
  --weights_path ../outputs/data_2/weights/model_weights_data2.json \
  --batch_size 32
```



# Pre-trained Model Weights

The model is trained on the given two datasets(MNIST and CIFAR100) and the trained model weights are included in the below folders:

* **Data 1**

  ```
  my_framework/outputs/data_1/weights/model_weights_data1.json
  ```

* **Data 2**

  ```
  my_framework/outputs/data_2/weights/model_weights_data2.json
  ```

## Sources Used

The following resources were referred for this assignment:

1. **Official Documentation**

   * Python Documentation: [https://docs.python.org/3/](https://docs.python.org/3/)
   * C++ Reference: [https://en.cppreference.com/](https://en.cppreference.com/)
   * pybind11 Documentation: [https://pybind11.readthedocs.io/](https://pybind11.readthedocs.io/)
   * CMake Documentation: [https://cmake.org/documentation/](https://cmake.org/documentation/)
   * OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)

2. **General Programming References**

   * Stack Overflow (for debugging build and environment issues)
   * Git documentation (for repository management)

3. **Academic Knowledge**

   * Lecture slides and course materials provided in GNR 638, IIT Bombay

4. **AI Assistance**

   * ChatGPT (OpenAI) — used for debugging support, conceptual clarification, and documentation refinement.
