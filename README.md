# Credit Risk Prediction with Random Forest

This project predicts whether a bank customer will default on a loan using the German Credit Data (from the UCI Machine Learning Repository).  
It includes a full machine learning pipeline: data preprocessing, training, evaluation, and interpretation.

## 📊 Dataset
- Source: UCI ML Repo – German Credit Data (1000 samples)
- Binary classification:
  - `1` = Good credit
  - `0` = Bad credit

## 🔧 Workflow
1. **Load and explore data** via `ucimlrepo`
2. **Rename columns** for clarity
3. **One-hot encoding** for categorical features
4. **MinMax scaling** for numerical features
5. **Train/test split (80/20)**
6. **Model training** using `RandomForestClassifier` with `class_weight='balanced'`
7. **Evaluation** using:
   - Accuracy
   - Confusion Matrix
   - Classification Report
   - ROC Curve & AUC Score
8. **Feature Importance** plot

## 📈 Results
- AUC score: ~0.79
- Accuracy: ~73%
- Top features: `credit_amount`, `duration_in_month`, and checking account status

## 🧰 Tools & Libraries
- Python, pandas, scikit-learn, seaborn, matplotlib
- `ucimlrepo` for dataset loading

## 💡 Motivation
Predicting credit risk can help banks and lenders make more informed, data-driven decisions to reduce default rates.

## 🚀 Run it on Google Colab
Just open the Jupyter notebook and run it cell by cell — the first cell installs any missing packages.

## 📂 File
- `final_credit_risk_project.ipynb` — main notebook
