# ECG-Based Seizure Detection

This repository contains the code and resources for our project **“ECG-Based Seizure Detection: Evaluating Modern Anomaly Detection Methods”**, conducted as part of the **Advanced Seminar Information Management (ASIM SS25)** at the University of Cologne.

## 🧠 Project Overview

Epileptic seizures can cause serious medical and psychosocial consequences. While EEG is the clinical gold standard for seizure detection, wearable ECG devices (e.g., smartwatches) offer a more practical alternative for daily use.

In this project, we evaluate the performance of three modern anomaly detection algorithms on ECG data for the purpose of seizure detection:

- **MERLIN++**
- **TimeVQVAE-AD**
- **Matrix Profile**

We aim to identify approaches that achieve a favorable balance between **sensitivity** and **false alarm rate** (FAR), using the [SeizeIT2 dataset](https://doi.org/10.48550/arXiv.2502.01224).

## 📊 Research Question

> To what extent can modern anomaly detection algorithms detect epileptic seizures from ECG signals in terms of sensitivity and false alarm rate?

## 🧪 Methodology

We follow a structured analysis approach based on the **CRISP-DM** model. Key steps:

- **Exploratory Data Analysis (EDA)**  
  Signal quality check, artifact detection, seizure episode distribution  
- **Feature Engineering**  
  Includes heart rate (HR), heart rate variability (HRV)-based features, and Cardiac Sympathetic Index (CSI)  
- **Model Implementation**  
  Evaluation of three anomaly detection algorithms (MERLIN++, TimeVQVAE-AD, Matrix Profile)  
- **Baseline**  
  A 1-class SVM is used for comparison  
- **Tracking & Evaluation**  
  Using [Weights & Biases](https://wandb.ai/) for experiment tracking  
  Metrics: Sensitivity, FAR, AUC, Specificity, PPV, Accuracy  

## 📁 Project Structure
ecg-seizure-detection
├── data/ # Data loaders and preprocessed ECG files
├── notebooks/ # Jupyter notebooks for exploration and prototyping
├── models/ # Implementation of MERLIN++, TimeVQVAE-AD, etc.
├── scripts/ # Scripts for training, evaluation, and metrics
├── results/ # Evaluation outputs and plots
├── README.md # This file
└── requirements.txt # Dependencies


## 📂 Dataset

We use the **SeizeIT2 dataset**, which includes:

- ECG recordings from 125 patients
- Over 11,000 hours of data sampled at 250Hz
- 886 annotated focal seizures

➡️ Download instructions will be added soon.

## ⚙️ Installation

```bash
git clone https://github.com/creintjes/ecg-seizure-detection.git
cd ecg-seizure-detection
pip install -r requirements.txt

