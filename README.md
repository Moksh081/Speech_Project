# 🎙️ Speaker Age & Gender Estimation from Voice

A Machine Learning and Deep Learning-based project to estimate the **age group** and **gender** of a speaker using **voice signals**, developed as part of the UCS749: Conversational AI course.

## 📌 Project Overview

This project explores speech signal processing and integrates it with traditional and deep learning models to predict speaker demographics. By extracting features such as **MFCCs**, **pitch**, and **formants**, we aim to build a system that can classify:

- 👤 **Gender** (Male / Female)
- 🧓 **Age Groups** (e.g., Teen, Adult, Senior)

---

## 📂 Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [References](#references)

---

## 📊 Dataset

We used a dataset containing labeled audio samples with corresponding gender and age annotations. The dataset was preprocessed using:
- Label mapping for age and gender
- Feature extraction using **Librosa**
- Feature scaling and encoding

---

## ⚙️ Methodology

1. **Import Libraries**
2. **Load and Clean Data**
3. **Feature Extraction**
   - MFCCs
   - Pitch
   - Formants
4. **Label Encoding & Feature Scaling**
5. **Train-Test Split (80/20)**
6. **Model Building**
   - Neural Network with Dense Layers, Dropout, and BatchNorm
7. **Model Training & Evaluation**
8. **Accuracy/Loss Visualization**

---

## 🧠 Model Architecture

- **Input Layer:** Audio features (MFCCs, etc.)
- **Hidden Layers:** Dense layers with ReLU activation, Dropout, Batch Normalization
- **Output Layer:** Softmax for classification (gender / age group)

---
