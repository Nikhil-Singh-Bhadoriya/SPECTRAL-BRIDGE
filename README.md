
# Spectral Bridge – Few‑Shot Signal In‑Painting

## Overview
This project solves the **Spectral Bridge Challenge**, which focuses on reconstructing missing portions of degraded analog audio recordings.

Due to **Sticky Shed Syndrome**, portions of magnetic recording tapes have deteriorated, resulting in missing waveform data. The dataset contains partial signal values (context points) and missing values (target points). The goal is to train a machine learning model capable of reconstructing these missing signals accurately.

This implementation uses a **Hybrid 1D Convolution + Transformer Cross‑Attention architecture** to learn both local temporal patterns and long‑range dependencies in the signal.

---

# Problem Description

Each audio sample contains:

• Context points → known signal values  
• Target points → missing values to predict  

Each **Sample_ID represents an independent acoustic event**.  
Therefore, the model must infer the waveform pattern only using context points from that specific sample.

Steps the model performs:

1. Observe context points
2. Infer spectral structure of the signal
3. Predict missing waveform values

---

# Dataset Description

The dataset contains approximately **80,000 audio sequences**, each lasting **100 milliseconds**.

Sampling Rate: **1 kHz**  
Total rows: **~8 million**

## Dataset Columns

| Column | Description |
|------|-------------|
| Sample_ID | Unique identifier for each audio clip |
| Time_ms | Timestamp (1–100 ms) |
| Is_Context | 1 = known signal, 0 = missing signal |
| Value | Normalized voltage amplitude |

### Context vs Target

Is_Context = 1 → Known value used for training  
Is_Context = 0 → Missing value to predict

---

# Objective

Predict missing signal values using context information.

Mathematically:

f(x_context, y_context) → y_target

Where:

x_context = time positions of known points  
y_context = known signal values  
y_target = predicted signal values

---

# Evaluation Metric

Performance is evaluated using **Mean Squared Error (MSE)** calculated only on missing points.

MSE = (1/N) Σ (y_true − y_pred)²

Lower MSE indicates better predictions.

---

# Model Architecture

The solution uses a **Hybrid Conv‑Transformer architecture**.

Pipeline:

Input Signal  
↓  
1D Convolution Feature Extractor  
↓  
Transformer Encoder with Cross Attention  
↓  
Feedforward Decoder  
↓  
Predicted Signal Values

### Components

**1D Convolution Blocks**
• Extract local temporal features
• Capture waveform patterns

**Transformer Attention**
• Learns relationships between context points
• Captures long‑range dependencies

**Decoder**
• Generates predicted signal values

---

# Training Strategy

Training pipeline:

Load dataset  
↓  
Group rows by Sample_ID  
↓  
Separate context and target points  
↓  
Train hybrid model  
↓  
Evaluate using cross‑validation  
↓  
Save best model

### Cross Validation

The project uses **3‑Fold Cross Validation** to improve model generalization and stability.

---

# Model Configuration

Best configuration:

d_model = 192  
num_conv_blocks = 4  
num_heads = 6  
num_decoder_layers = 4  
ff_mult = 4  
num_bands = 16  
dropout = 0.15  
learning_rate = 0.0003  
batch_size = 128  
epochs = 80  

---

# Training Pipeline

Load Training Dataset  
↓  
Preprocess Data  
↓  
Generate Context / Target Pairs  
↓  
Train Model (3‑Fold CV)  
↓  
Select Best Model  
↓  
Save Model Checkpoint

Saved model file:

final_best_model_3fold.pt

---

# Inference Pipeline

During evaluation:

Load trained model  
↓  
Load test dataset  
↓  
Extract context points  
↓  
Predict missing signal values  
↓  
Generate submission file

Output file:

submission_Test.csv

---

# Submission Format

CSV format:

Sample_ID,Time_ms,Predicted_Value

Example:

10001,32,0.4521  
10001,33,0.4489  
10001,34,0.4401  

Only rows with missing values are evaluated.

---

# Project Structure

SpectralBridge/

final_best_model_3fold.pt  
README.md  
Round1.ipynb  
Round2_Using_all_round1_config.ipynb  
search_results_3fold.csv  
submission_Test.csv  
test_features.csv 

---

# How to Run

## Install dependencies

pip install torch numpy pandas scikit-learn

## Train the model

Run:

Round1.ipynb

This notebook trains the model and saves the best checkpoint.

## Run inference

Run:

Round2_Using_all_round1_config.ipynb

This loads the saved model and generates predictions for the test dataset.

---

# Innovations

• Hybrid Conv + Transformer architecture  
• Few‑shot signal reconstruction  
• Efficient amortized inference  
• Cross‑validation training

---

# Challenges

• Large dataset (~8 million rows)  
• Highly non‑stationary audio signals  
• Sparse context information  
• Efficient training with limited compute

---
