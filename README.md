# Multi-Class Prediction of Mobile Game Installation

## Overview
This project tackles a **multi-class classification problem** on real-world mobile game data.
The objective is to predict **which game a user will install in the following month**, among four possible outcomes:

- Class 1: No game installed  
- Class 2: Game A  
- Class 3: Game B  
- Class 4: Game C  

The dataset contains **5,000 labeled training observations** and **55,000 test observations**.
The task was part of a **Kaggle competition** used for academic evaluation at **HEC Montréal**.

This project achieved **2nd place on the Kaggle leaderboard**, demonstrating strong predictive performance and robust generalization.

---

## Dataset
The dataset consists of tabular user-level data describing:

- **User activity** (sessions, playtime, score)
- **Monetization behavior** (purchases, spending trends)
- **Skill levels** (skill1, skill2)
- **Temporal and behavioral trends**
- **Categorical context variables** (country, operating system, acquisition channel)

Both **training and test datasets are included** in this repository for full reproducibility.

---

## Methodology

### 1. Baseline: One-vs-All Random Forest
A **Random Forest One-vs-All (OvA)** classifier was implemented to explore binary decomposition approaches for multi-class classification.

This baseline achieved approximately **70% accuracy**, serving as a reference point.

---

### 2. Feature Engineering
Significant feature engineering was performed to capture behavioral intensity and interaction effects:

- **Intensity metrics**  
  (e.g. score per session, playtime per day, purchases per session)

- **Behavioral interactions**  
  (e.g. purchases × sessions, skill × score)

- **Threshold effects** via quantile binning  
  (e.g. low vs high spenders)

These engineered features consistently improved performance and were validated using SHAP analysis.

---

### 3. Gradient Boosting Models
Three tree-based boosting models were trained and compared:

- **LightGBM**
- **XGBoost**
- **CatBoost**

Each model was optimized using **Bayesian hyperparameter optimization (Optuna)** with stratified cross-validation.

---

### 4. Ensemble Learning: Stacking
To exploit model complementarity, a **stacking ensemble** was implemented:

1. Out-of-fold class probabilities generated for each base model.
2. Probabilities concatenated into a meta-feature matrix.
3. A **regularized LightGBM** model trained as a meta-learner with early stopping.

This ensemble produced the best overall performance and was used for final Kaggle submissions.

---

### 5. Interpretability (SHAP)
Model interpretability was addressed using **SHAP values**:

- Comparison between models trained on **raw features** vs **engineered features**
- Confirmation that monetization and engagement intensity variables drive predictions

This ensures the model is not only performant but also explainable.

---

## Results

| Model                        | Accuracy | F1-Macro | ROC-AUC |
|-----------------------------|----------|----------|---------|
| Random Forest (OvA)         | ~0.70    | ~0.70    | —       |
| LightGBM                    | ~0.77    | ~0.77    | ~0.95   |
| XGBoost                     | ~0.77    | ~0.77    | ~0.94   |
| CatBoost                    | ~0.78    | ~0.79    | ~0.95   |
| **Stacking (Meta LightGBM)**| **~0.82**| **~0.82**| **~0.96** |

**Kaggle leaderboard:**  
- 🥈 **2nd place overall**
- ~84% accuracy on the public leaderboard

---

## Technologies Used
- Python
- NumPy, Pandas
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Optuna (Bayesian optimization)
- SHAP (model interpretability)
- Matplotlib

---

## Reproducibility
The repository includes:
- Full training and test datasets
- Feature engineering pipeline
- Hyperparameter optimization code
- Stacking ensemble implementation
- Final submission generation

All experiments can be reproduced end-to-end.

---




