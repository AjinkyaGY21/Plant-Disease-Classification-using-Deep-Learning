# 🌱 Plant Disease Classification Using Deep Learning  

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Enabled-blue)  
![Python](https://img.shields.io/badge/Python-3.7%2B-green)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  

This project implements a deep learning solution for detecting plant diseases using multiple CNN architectures, including a custom CNN, ResNet50, and VGG16. The system processes plant images in color, grayscale, and segmented formats to classify plant diseases.

---

## 🚀 Features

- 🕼️ **Multi-Mode Image Processing**: color, grayscale, segmented
- 🧠 **CNN Architectures**:
  - Custom CNN
  - ResNet50
  - VGG16
- 🔄 **Data Augmentation** for robust training
- 📊 **Training History Visualization**
- 🔗 **Automated Dataset Splitting** into train/test sets

---

## 🗈 Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for faster training)

### Required Libraries

Install the dependencies via pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn Pillow
```

---

## 📂 Dataset Structure

The project expects the dataset to follow this structure:

```
plantvillage dataset/
├── color/
│   ├── disease_class_1/
│   │   ├── image1.jpg
│   │   └── ...
│   └── disease_class_2/
├── grayscale/
│   └── ...
└── segmented/
    └── ...
```

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/AjinkyaGY21/Plant-Disease-Classification-using-Deep-Learning.git
cd Plant-Disease-Classification-using-Deep-Learning
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Modify the constants in `plant_disease_detection.py` to suit your dataset and requirements:

```python
BASE_DIR = "path/to/your/dataset"
TARGET_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
```

---

## ▶️ Usage

1. Prepare your dataset according to the structure mentioned above.

2. Run the script:

```bash
python plant_disease_detection.py
```

### Output:

- Trains models using different architectures and modes
- Displays training and validation plots
- Prints test accuracy for each configuration

---

## 🧠 Model Architectures

### Custom CNN

- 3 Convolutional blocks (filters: 32, 64, 128)
- Batch normalization after each convolution
- MaxPooling layers
- Dense layers with dropout for classification

### Pre-trained Models

- ResNet50 and VGG16 with transfer learning
- Frozen base layers
- Custom top layers for final classification

---

## 📊 Results Visualization

During training, the system generates:

- Accuracy and loss plots for training vs validation
- Test accuracy for each configuration (mode + model type)

### 🌟 CNN Performance Plots  

- **Live Demo**:  
  View Live: [https://claude.site/artifacts/46734783-75d3-4d86-97e0-4a0070f96eae](https://claude.site/artifacts/46734783-75d3-4d86-97e0-4a0070f96eae)

### 🌟 ResNet and VGGNet Performance Plots

- **Live Demo**:  
  View Live: [https://claude.site/artifacts/0a7199ee-8f86-489b-addb-d7ac75dfe8b6](https://claude.site/artifacts/0a7199ee-8f86-489b-addb-d7ac75dfe8b6)


---

## 🌟 Acknowledgments

- [Plant Village Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data)
