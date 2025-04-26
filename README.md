# ☀️ Lacuna Solar Survey Challenge - Solar Panel & Boiler Detection

## 🏆 Project Overview

This repository contains our solution for the [Lacuna Solar Survey Challenge](https://zindi.africa/competitions/lacuna-solar-survey-challenge) hosted on Zindi.  
The goal was to build a robust machine learning model capable of accurately detecting and counting **solar panels** and **solar boilers** from **satellite and drone imagery** of Madagascar.

A scalable detection system like this can help:
- Governments
- NGOs
- Energy providers  
...to better plan, monitor, and optimize renewable energy deployment across Africa.

---

## 📜 Competition Description

> Access to reliable and sustainable energy is a critical issue in Madagascar and across Africa, where a significant portion of the population lacks electricity or relies on costly, environmentally harmful energy sources. Solar energy presents a viable alternative, but mapping the distribution of solar panels and boilers is challenging over vast and remote areas.

Participants were challenged to:
- Detect and **count** the number of **solar panels** and **solar boilers** per image.
- Build scalable, efficient, and accurate machine learning solutions.

🏆 Winning solutions were announced at the **Global AI Summit for Africa**, held in **Kigali, Rwanda** in **April 2025**.

---

## 📂 Project Structure

├── GODS.ipynb # Main training and inference notebook ├── README.md # Project overview (this file)

## 🛠️ Approach

### 📦 Data Preprocessing
- Combined `title` and `content` columns into a unified `text` field.
- Filled missing values with empty strings.
- Mapped metadata fields (`placement`, `img_origin`) and created full image paths.
- Applied **Stratified K-Fold** (7 folds) cross-validation based on combined panel and boiler counts.

### 🎨 Data Augmentation
- **Albumentations** transformations:
  - Resizing, flips, brightness/contrast adjustment, elastic/grid distortions, normalization.
- **Custom Torch transforms**:
  - Random sharpness and random blur added for better generalization.
- **Combined transforms**: Seamlessly merged Albumentations + TorchVision transformations into one pipeline.

### 🏗️ Model Architecture
- **Backbone**: Pretrained `InceptionV3` model (via TIMM library).
- **Head**: Final fully connected layer outputs two values:
  - Number of **boilers** and number of **solar panels**.
- **Loss Function**: Mean Absolute Error (**L1 Loss**).

### ⚙️ Training Strategy
- **Optimizer**: Adam with an initial learning rate of `1e-4`.
- **Scheduler**: ReduceLROnPlateau (adaptive learning rate).
- **Early Stopping**: Stops training after 10 epochs without validation improvement.
- **TensorBoard**: Training and validation loss curves monitored in real time.

### 📈 Evaluation
- Main validation metric: **Mean Absolute Error (MAE)**.
- Best model checkpoint saved automatically.

---

## 🚀 Inference & Submission

- Predictions generated on the test set.
- Outputs formatted with one row per ID per target (`_boil` and `_pan` suffix).
- Final predictions clipped between **0** and **1000**.
- Saved to `Submission.csv` for competition submission.

---

## 📈 Results

| Metric        | Value         |
|---------------|---------------|
|Validation MAE | 1.32         |

---

## 📚 Key Learnings

- Designing hybrid augmentation pipelines with Albumentations + TorchVision.
- Adapting pretrained models for small object detection and counting tasks.
- Handling imbalanced multi-target regression problems with stratified data splitting.
- Efficient training monitoring with early stopping and learning rate scheduling.

---

## 🙏 Acknowledgements

- Thanks to **Zindi Africa** and **Lacuna** for organizing this important and impactful challenge.
- Thanks to the **Global AI Summit for Africa** for showcasing and supporting African innovation.

---
