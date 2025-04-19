
# 👕👟 T-shirt vs Sneaker Classifier with PCA & Bayesian Decision Theory

A dimensionality-reduction and classification pipeline built from scratch to classify Fashion-MNIST images (T-shirts vs Sneakers) using **Principal Component Analysis (PCA)**, **Gaussian density estimation**, and **Bayesian minimum-error classification**.  
Designed for explainability and performance using core statistical learning techniques — without relying on prebuilt ML libraries.

---

## 📌 Problem Overview

The project addresses the challenge of classifying high-dimensional image data from Fashion-MNIST using:
- PCA for dimensionality reduction
- 2D Gaussian modeling for class-conditional densities
- Bayesian decision theory for optimal classification

The original data contains grayscale 28×28 images of T-shirts and sneakers. Each image is reshaped into a 784-dimensional vector. Due to the curse of dimensionality, PCA is applied before classification.

---

## 📁 Dataset Description

- Source: **Modified Fashion MNIST (.mat format)**
- Classes: T-shirt (Label 0), Sneaker (Label 1)
- Each image: 28×28 pixels → reshaped to 784-dimensional vector

| Dataset      | T-shirts | Sneakers |
|--------------|----------|----------|
| Training Set | 6000     | 6000     |
| Testing Set  | 1000     | 1000     |

---

## 🧪 Project Workflow

### ✅ Task 1: Feature Normalization
- Standardized each feature (pixel) using the mean and standard deviation from the training set
- Normalization applied to both train and test sets

### ✅ Task 2: Principal Component Analysis (PCA)
- Computed covariance matrix and eigen decomposition manually (no `sklearn`)
- Identified principal components and sorted by descending eigenvalues

### ✅ Task 3: 2D Projection & Visualization
- Projected data onto the first two principal components
- Visualized T-shirt and Sneaker clusters in 2D space

### ✅ Task 4: Density Estimation
- Modeled class-conditional distributions as **2D multivariate Gaussians**
- Estimated mean and covariance matrix for each class

### ✅ Task 5: Bayesian Decision Theory
- Used Bayes' optimal rule to assign class labels based on Gaussian likelihoods
- Assumed equal class priors: P(T-shirt) = P(Sneaker) = 0.5

### 🎯 Classification Accuracy
```
Training Accuracy: 99.50%
Testing Accuracy: 99.90%
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/tshirt-sneaker-bayes-classifier.git
cd tshirt-sneaker-bayes-classifier

# 2. Place the following dataset files in the same directory:
#    - train_data.mat
#    - test_data.mat

# 3. Run the classification pipeline
python pca_bayes_classifier.py
```

> 🔎 *No ML libraries like scikit-learn used for PCA or classification — this project implements everything from scratch for interpretability and clarity.*

---

## 🧠 Key Skills Demonstrated

- 📉 **Dimensionality Reduction** with PCA
- 📐 **Multivariate Gaussian Modeling**
- 🤖 **Bayesian Decision Theory**
- 📊 **Data Normalization & Preprocessing**
- 🧪 **From-scratch Implementation of Core ML Concepts**
