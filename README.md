# 📊 Bank Marketing Campaign (LightGBM + SMOTE)

This repository contains my solution for the [SIGNATE "Bank Marketing Campaign"](https://signate.jp/competitions/###) binary classification competition, where the goal is to predict whether a customer will subscribe to a term deposit product.

---

## 🔍 Problem Overview

- **Type**: Binary Classification  
- **Target variable**: `y` (1 = subscribed, 0 = not subscribed)  
- **Challenge**: Highly imbalanced dataset (~11.7% positives)  
- **Goal**: Maximize ROC-AUC while improving recall for the minority class (subscribers)

---

## 📁 Files in this Repository

| File                         | Description                                   |
|------------------------------|-----------------------------------------------|
| `submit_lightgbm_final.csv`  | Final prediction file (ready for submission) |
| `README.md`                  | This file                                    |

---

## ⚙️ Approach Summary

1. **Data Preprocessing**
   - Dropped `duration` column (future leak)
   - Label Encoding for categorical features
   - Standardization via `StandardScaler` (for logistic model)

2. **Imbalanced Data Handling**
   - Used `SMOTE` to oversample the minority class
   - Balanced train set (50% positive, 50% negative)

3. **Modeling**
   - Baseline: RandomForestClassifier
   - Final Model: LightGBMClassifier
   - Tuned with default settings for interpretability and speed

4. **Evaluation**
   - Train/Valid split (80/20)
   - Metrics: `ROC-AUC`, `Recall`, `Precision`, `F1-score`

| Model         | ROC-AUC | Recall (1) | F1 (1) |
|---------------|---------|------------|--------|
| Random Forest | 0.760   | 0.44       | 0.41   |
| LightGBM      | 0.775   | 0.50       | 0.43   |

---
```python
submit = pd.read_csv("submit_sample.csv", header=None)
submit[1] = y_pred_proba
submit.to_csv("submission.csv", index=False, header=False)
