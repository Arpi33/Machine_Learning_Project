<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <h2>Insurance Claim Prediction – ML Pipeline</h2>
</head>
<body>
  <h1>Insurance Claim Prediction – Machine Learning Pipeline</h1>
  <h3>A comparative study of imbalance-aware ML models for car-insurance claim prediction</h3>

  <h2>Overview</h2>
  <p>
    This project implements an end-to-end machine-learning pipeline to predict car insurance
    claim occurrence using structured policyholder, vehicle, and geographic data.
    The workflow covers data preprocessing, handling severe class imbalance, feature
    engineering, mutual-information–based feature selection, and comparative evaluation
    of multiple supervised learning models.
  </p>

  <h2>Key Features</h2>
  <ul>
    <li>End-to-end ML workflow: EDA → preprocessing → resampling → feature selection → modeling → evaluation</li>
    <li>Imbalance-aware training using SMOTE and decision-threshold optimization</li>
    <li>Comparison of Logistic Regression, LDA, Decision Tree, Random Forest, SVM, and XGBoost</li>
    <li>Mutual-information ranking for selecting top-k informative features</li>
    <li>Evaluation with metrics suited for imbalanced data (ROC-AUC, precision, recall, F1-score, confusion matrices)</li>
  </ul>

  <h2>Methods</h2>
  <ul>
    <li>Outlier detection and removal using the IQR rule on key numerical features</li>
    <li>Numeric parsing for mixed-format attributes (e.g., torque and power fields)</li>
    <li>Label encoding for categorical attributes and Min–Max scaling for numerical features</li>
    <li>Feature selection via mutual information to identify the most informative predictors</li>
    <li>Hyperparameter tuning with <code>GridSearchCV</code> (5-fold) for Decision Tree, Random Forest, and XGBoost</li>
    <li>Threshold tuning on predicted probabilities to improve minority-class recall</li>
  </ul>

  <h2>Results (Summary)</h2>
  <ul>
    <li>XGBoost achieved the best overall performance in terms of ROC-AUC and F1-score.</li>
    <li>SVM achieved the highest recall but with more false positives.</li>
    <li>Mutual-information–based feature subsets improved stability for linear models (LDA, Logistic Regression).</li>
    <li>Ensemble models (Random Forest, XGBoost) handled imbalance and multicollinearity more effectively.</li>
  </ul>

  <h2>Repository Structure</h2>
  <pre>
/notebooks
    ├── data_preprocessing.ipynb
    ├── EDA.ipynb
    ├── model_training_comparisons.ipynb

  </pre>

  <h2>Requirements</h2>
  <ul>
    <li>Python 3.10+</li>
    <li>pandas, numpy</li>
    <li>scikit-learn</li>
    <li>imbalanced-learn</li>
    <li>xgboost</li>
    <li>matplotlib (and/or seaborn)</li>
  </ul>

  <p>Install dependencies:</p>
  <pre><code>pip install -r requirements.txt</code></pre>

  <h2>How to Run</h2>
  <ol>
    <li>Open <code>data_preprocessing.ipynb</code> and run all cells to clean and transform the dataset.</li>
    <li>Open <code>model_training_comparisons.ipynb</code> to train and evaluate the models.</li>
    <li>Inspect the generated metrics and plots to compare model performance.</li>
  </ol>

</body>
</html>
