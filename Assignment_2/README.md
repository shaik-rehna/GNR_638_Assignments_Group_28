# Pre-trained CNN Representation Transfer and Robustness Analysis

### GNR 638 – Assignment 2

### Group 28

This project analyzes how **ImageNet-pretrained convolutional neural networks transfer to aerial scene classification** using the **Aerial Image Dataset (AID)** under multiple experimental scenarios.

The experiments investigate:

* Transferability of pretrained representations
* Fine-tuning strategies
* Data efficiency under few-shot learning
* Robustness to input corruptions
* Layer-wise semantic abstraction in CNNs

---

# Implemented CNN Architectures

The following pretrained backbones are evaluated:

* **ResNet50**
* **DenseNet121**
* **EfficientNet-B0**

All models use **ImageNet pretrained weights**.

---

# Experimental Scenarios

The repository implements all assignment scenarios.

---

## 4.1 Linear Probe Transfer

* Backbone frozen
* Only the classifier layer is trained

Evaluates **how well pretrained ImageNet features transfer** to aerial scene classification.

Outputs:

* Training / Validation accuracy curves
* Confusion matrix
* PCA and t-SNE feature visualization

---

## 4.2 Fine-Tuning Strategies

Comparison of multiple transfer learning strategies:

* Linear probe
* Last block fine-tuning
* Selective 20% parameter unfreezing
* Full fine-tuning

Metrics:

* Accuracy vs percentage of unfrozen parameters
* Convergence stability
* Gradient norm statistics

---

## 4.3 Few-Shot Learning

Training models with limited data:

* 100% training data
* 20% training data
* 5% training data

Metrics:

* Validation accuracy
* Relative performance drop
* Training–validation gap

---

## 4.4 Corruption Robustness

Evaluation under controlled distribution shifts:

* Gaussian noise
* Motion blur
* Brightness shift

Metrics:

* Corruption error
* Relative robustness

---

## 4.5 Layer-Wise Feature Probing

Analyzing **semantic abstraction across network depth**.

Features are extracted from:

* Early layers
* Middle layers
* Final layers

Metrics:

* Probe accuracy vs depth
* Feature norm statistics
* PCA visualization

---

# Repository Structure

```
Assignment_2
│
├── src/
│   └── cnn_transfer/
│       ├── dataset.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       ├── corruption.py
│       ├── feature_probe.py
│       └── utils.py
│
├── notebooks/
│   ├── experiments_resnet50.ipynb
│   ├── experiments_densenet121.ipynb
│   ├── experiments_efficientnet_b0.ipynb
│   └── evaluation_script.ipynb
│
├── checkpoints/
│   ├── resnet50/
│   ├── densenet121/
│   └── efficientnet_b0/
│
├── logs/
│   ├── resnet50/
│   ├── densenet121/
│   └── efficientnet_b0/
│
├── plots/
│   ├── resnet50/
│   ├── densenet121/
│   └── efficientnet_b0/
│
├── results/
│   ├── resnet50/
│   ├── densenet121/
│   └── efficientnet_b0/
│
├── experimental_results/
│   ├── plots/
│   └── results/
│
├── REPORT_GNR638_ASSIGN_2_GROUP28
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# Setup Instructions

## 1 Clone Repository

```
git clone https://github.com/shaik-rehna/GNR_638_Assignments_Group_28.git
cd GNR_638_Assignments_Group_28/Assignment_2
```

---

# Python Setup

## Check Python Version

Windows

```
py -3.12 --version
```

macOS / Linux

```
python3.12 --version
```

If Python is not installed:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

---

# Create Virtual Environment

Windows

```
py -3.12 -m venv gnr638_venv
```

macOS / Linux

```
python3.12 -m venv gnr638_venv
```

---

# Activate Virtual Environment

Windows

```
gnr638_venv\Scripts\activate
```

macOS / Linux

```
source gnr638_venv/bin/activate
```

---

# Install Dependencies

```
pip install -r requirements.txt
pip install -e .
pip install ipykernel
```

---

# Jupyter Notebook Setup (VS Code)

To run notebooks inside the virtual environment, install Jupyter kernel support.

---

## Register the Environment as a Kernel

```
python -m ipykernel install --user --name=gnr638_venv --display-name "Python (gnr638_venv)"
```

---

## Reload VS Code

Press:

```
Cmd + Shift + P
```

Run:

```
Developer: Reload Window
```

---

## Select Kernel in Notebook

Open 

```
notebooks/evaluation_script.ipynb
```

Then select the kernel:

```
Kernel → Python (gnr638_venv)
```

---

## If VS Code Does Not Detect the Interpreter

Manually select the interpreter.

1. Press

```
Cmd + Shift + P
```

2. Run

```
Python: Select Interpreter
```

3. Choose

```
Enter interpreter path → Find
```

4. Navigate to the environment:

macOS / Linux

```
Assignment_2/gnr638_venv/bin/python
```

Windows

```
Assignment_2\gnr638_venv\Scripts\python.exe
```

---

# Dataset Setup

Train dataset (**Aerial Image Dataset (AID)**) is inside dataset folder:

```
dataset/train_data/
```

Example:

```
dataset/train_data/
├── airport
├── bareland
├── baseballfield
├── beach
└── ...
```

To evalaute on test data place it inside the dataset folder:

```
dataset/test_data/
```

---

# Running Experiments

Training notebooks are provided for each architecture:

```
notebooks/experiments_resnet50.ipynb
notebooks/experiments_densenet121.ipynb
notebooks/experiments_efficientnet_b0.ipynb
```

These notebooks perform:

* model training
* experiment analysis
* checkpoint saving
* logging metrics

Outputs are stored in:

```
checkpoints/
logs/
plots/
results/
```

---

# Evaluation Notebook

A universal evaluation notebook is provided:

```
notebooks/evaluation_script.ipynb
```

This notebook **reproduces evaluation results without retraining models**.

---

# Switching Models

To evaluate another architecture, change one line:

```
MODEL_NAME = "resnet50"
```

Options:

```
MODEL_NAME = "resnet50"
MODEL_NAME = "densenet121"
MODEL_NAME = "efficientnet_b0"
```

The notebook automatically loads:

```
checkpoints/<model_name>/
logs/<model_name>/
```

---

# Validation vs Test Evaluation

Choose dataset split:

```
EVAL_SPLIT = "val"
```

Options:

```
EVAL_SPLIT = "val"
EVAL_SPLIT = "test"
```

If using test evaluation, place dataset in:

```
dataset/test_data/
```

---

# Generated Outputs

Running the evaluation notebook populates:

```
plots/
results/
```

Example:

```
plots/resnet50/
results/resnet50/
```

Outputs from the already performed experiments are stored in:

```
experimental_results/
```

---

# Reproducibility

All experiments use fixed random seeds:

```
seed = 42
```

Randomness controlled for:

* NumPy
* PyTorch
* DataLoader workers

---

# Computational Environment

Experiments were run using:

* Python 3.12
* PyTorch
* NVIDIA GPU (Google Colab Tesla T4)


