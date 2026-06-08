# 🧠 Enhanced MobileNet-Based Image Classification using Custom Focal Loss on CIFAR-10

> A lightweight, efficient image classification system built with **MobileNet** + **Custom Focal Loss** on the **CIFAR-10** dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Team Members](#team-members)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Custom Loss Function](#custom-loss-function)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
- [Evaluation Metrics](#evaluation-metrics)
- [Generated Outputs](#generated-outputs)
- [Learning Outcomes](#learning-outcomes)
- [Future Scope](#future-scope)
- [Acknowledgements](#acknowledgements)

---

## 📌 Overview

This project presents an efficient image classification system developed using the **CIFAR-10** dataset and the **MobileNet** deep learning architecture. The primary goal is to classify images into ten distinct categories while boosting model performance through a custom **Focal Loss** function.

The system leverages **Transfer Learning** with MobileNet and evaluates performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix analysis. The entire implementation is built using **TensorFlow** and **Keras**, and executed on GPU-enabled environments such as **Google Colab**.

---

## 👥 Team Members

| S.No | Name | Role |
|------|------|------|
| 1 | **Thrilokesh Reddy** | Project Lead & Model Development |
| 2 | **Eswarnadh** | Data Engineering & System Integration |
| 3 | **Arun** | Evaluation, Documentation & Visualization |

---

## 🎯 Objectives

- Develop an image classification system using **MobileNet** with Transfer Learning
- Implement a custom **Focal Loss** function to improve model performance
- Perform image classification on the **CIFAR-10** dataset (10 classes)
- Evaluate the model using multiple performance metrics
- Generate graphical visualizations for performance analysis
- Save and prepare the trained model for deployment

---

## 📦 Dataset

### CIFAR-10

The CIFAR-10 dataset consists of **60,000 color images** across **10 classes**.

| Attribute | Value |
|-----------|-------|
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Total Classes | 10 |
| Image Size | 32 × 32 × 3 (RGB) |

### Classes

| # | Class | # | Class |
|---|-------|---|-------|
| 0 | ✈️ Airplane | 5 | 🐕 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐎 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚚 Truck |

---

## 🏗️ Model Architecture

### MobileNet (Transfer Learning)

MobileNet is a lightweight convolutional neural network that uses **depthwise separable convolutions** to reduce computational cost while preserving classification accuracy — ideal for resource-constrained environments.

```
Input (32×32×3)
    ↓
MobileNet Base Model  (pretrained weights, feature extractor)
    ↓
Global Average Pooling Layer
    ↓
Dropout Layer          (regularization)
    ↓
Dense Output Layer     (10 units)
    ↓
Softmax Activation     (probability distribution over 10 classes)
```

---

## ⚙️ Custom Loss Function — Focal Loss

A custom **Focal Loss** function is implemented to improve learning by concentrating training effort on **hard-to-classify samples**.

### Formula

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

Where:
- `p_t` — model's estimated probability for the true class
- `γ (gamma)` — focusing parameter (down-weights easy examples)
- `α (alpha)` — class weighting factor

### Benefits

| Benefit | Description |
|---------|-------------|
| 🎯 Hard sample focus | Concentrates learning on misclassified or ambiguous samples |
| ⚖️ Easy sample suppression | Reduces influence of well-classified examples |
| 🔒 Robustness | Improves overall classification robustness |
| 🛠️ Customizability | Demonstrates flexible deep learning workflow design |

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|----------|-------------------|
| **Language** | Python 3.x |
| **Deep Learning** | TensorFlow, Keras |
| **Data Processing** | NumPy, Pickle |
| **Visualization** | Matplotlib |
| **Evaluation** | Scikit-Learn |
| **Environment** | Google Colab (GPU), VS Code |
| **Version Control** | GitHub |

---

## 📁 Project Structure

```
ICSPROJECT/
│
├── Train.py                   # Main training script
├── custom_loss.py             # Custom Focal Loss implementation
├── utils.py                   # Utility/helper functions
├── evaluate.py                # Evaluation and metrics generation
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore
│
├── cifar-10-batches-py/       # CIFAR-10 raw dataset
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── test_batch
│   └── batches.meta
│
├── results/                   # Generated output files
│   ├── accuracy.png           # Training/validation accuracy graph
│   ├── loss.png               # Training/validation loss graph
│   ├── confusion_matrix.png   # Confusion matrix visualization
│   ├── metrics.txt            # Numeric performance metrics
│   └── classification_report.txt
│
└── saved_model/
    └── mobilenet_cifar10.keras  # Trained and saved model
```

---

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/asodithrilokeshreddy-hue/ICSPROJECT.git
cd ICSPROJECT
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** It is recommended to use a virtual environment or run on **Google Colab** with GPU acceleration enabled for faster training.

---

## ▶️ Running the Project

Execute the main training script:

```bash
python Train.py
```

The pipeline will automatically:

1. **Load** the CIFAR-10 dataset from `cifar-10-batches-py/`
2. **Preprocess** images (normalize, resize)
3. **Build** the MobileNet model with Transfer Learning
4. **Apply** the custom Focal Loss function
5. **Train** the network on 50,000 images
6. **Evaluate** performance on 10,000 test images
7. **Generate** metrics and visualizations in `results/`
8. **Save** the trained model to `saved_model/`

---

## 📊 Evaluation Metrics

The model is evaluated using the following metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions / total predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Per-class prediction breakdown (10×10 matrix) |
| **Classification Report** | Full per-class metrics summary |

---

## 📈 Generated Outputs

After training, the following files are saved in the `results/` directory:

| File | Description |
|------|-------------|
| `accuracy.png` | Training vs. validation accuracy over epochs |
| `loss.png` | Training vs. validation loss over epochs |
| `confusion_matrix.png` | Heatmap of predictions vs. true labels |
| `metrics.txt` | Accuracy, Precision, Recall, F1-Score values |
| `classification_report.txt` | Per-class detailed metrics report |

The trained model is saved as:
```
saved_model/mobilenet_cifar10.keras
```

---

## 🎓 Learning Outcomes

Through this project, the team gained hands-on experience in:

- Deep Learning model design and training
- Transfer Learning with pretrained MobileNet weights
- Custom Loss Function implementation (Focal Loss)
- Image preprocessing and data pipeline development
- Model evaluation using multiple performance metrics
- GPU-based model training via Google Colab
- GitHub collaboration and version control
- Research documentation and report writing

---

## 🔭 Future Scope

- [ ] Fine-tune MobileNet's upper layers for domain-specific improvements
- [ ] Apply advanced data augmentation (random flips, cutout, mixup)
- [ ] Explore alternative architectures: **EfficientNet**, **ResNet50**, **Vision Transformer**
- [ ] Perform systematic hyperparameter optimization (learning rate, gamma, batch size)
- [ ] Deploy the trained model as a **web application** (Flask / FastAPI)
- [ ] Build a **real-time image classification** pipeline using a webcam feed

---

## 🙏 Acknowledgements

We sincerely thank:

- Our **faculty members and institution** for their continuous guidance and support
- The **TensorFlow / Keras** development team for the powerful open-source framework
- **CIFAR-10 dataset creators** (Alex Krizhevsky, Vinod Nair, Geoffrey Hinton) for the benchmark dataset
- The broader **open-source community** for tools, tutorials, and resources that made this project possible

---

## 👨‍💻 Contributors

<table>
  <tr>
    <td align="center"><b>Thrilokesh Reddy</b><br/><sub>Project Lead & Model Development</sub></td>
    <td align="center"><b>Eswarnadh</b><br/><sub>Data Engineering & Integration</sub></td>
    <td align="center"><b>Arun</b><br/><sub>Evaluation, Docs & Visualization</sub></td>
  </tr>
</table>

---

<p align="center">
  Made with ❤️ using TensorFlow, Keras, and MobileNet
</p>
