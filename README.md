# Credit Risk Prediction with Random Forest

This project predicts whether a bank customer will default on a loan using the German Credit Data from the UCI Machine Learning Repository.  
It includes a full machine learning pipeline with data preprocessing, model training, evaluation, and interpretation.

## Dataset
- Source: UCI ML Repository – German Credit Data (1000 samples)
- Binary classification:
  - 1 = Good credit
  - 0 = Bad credit

## Workflow
1. Load and explore the dataset using `ucimlrepo`
2. Rename columns for clarity
3. Apply one-hot encoding to categorical features
4. Apply MinMax scaling to numerical features
5. Split the data into training and testing sets (80/20)
6. Train a RandomForestClassifier with balanced class weights
7. Evaluate the model using:
   - Accuracy
   - Confusion Matrix
   - Classification Report
   - ROC Curve and AUC Score
8. Plot the most important features

## Results
- AUC score: approximately 0.79
- Accuracy: approximately 73%
- Most important features: credit amount, duration in months, checking account status

## Tools and Libraries
- Python, pandas, scikit-learn, seaborn, matplotlib
- ucimlrepo (for loading the dataset)

## How to Run
Open the Jupyter notebook and run the cells one by one. The first cell installs any missing packages.

## Files
- final_credit_risk_project.ipynb — main notebook
