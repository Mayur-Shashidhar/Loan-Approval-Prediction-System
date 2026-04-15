# 🧠 Loan Approval Prediction System

## 📌 Overview

This project implements an end-to-end machine learning pipeline to predict loan approval status based on applicant financial and demographic data. The system includes preprocessing, feature engineering, model training, evaluation, and prediction on unseen data.

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib
* Scikit-learn

---

## Dataset

This project uses a dataset from Kaggle.

Note: Dataset is subject to its own license (CC0 / CC BY).

---

## 📂 Project Workflow (Notebook Explanation)

---

### 🔹 1. Import Libraries & Load Dataset

We import required libraries and load the dataset into a DataFrame.
This allows us to perform data analysis and preprocessing.
Initial rows are displayed to understand the structure.

```python
import pandas as pd

df = pd.read_csv("loan.csv")
df.head()
```

---

### 🔹 2. Data Inspection

We inspect data types, structure, and missing values.
This step helps identify issues such as null values and categorical variables.
Understanding the dataset is crucial before preprocessing.

```python
df.info()
df.isnull().sum()
```

---

### 🔹 3. Handling Missing Values

Missing values are handled using appropriate statistical methods.
Median is used for numerical features to reduce outlier impact.
Mode is used for categorical features to preserve distribution.

```python
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
```

---

### 🔹 4. Feature Engineering

New features are created to improve model performance.
These features capture financial relationships more effectively.
They help the model learn better patterns.

```python
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
df['DTI'] = df['LoanAmount'] / df['Total_Income']
```

---

### 🔹 5. Dropping Irrelevant Features

The `Loan_ID` column is removed as it has no predictive value.
Keeping irrelevant features can reduce model performance.
This step ensures cleaner input data.

```python
df = df.drop('Loan_ID', axis=1)
```

---

### 🔹 6. Target Encoding

The target variable is converted into numerical form.
Machine learning models require numeric inputs.
This allows classification algorithms to process the target.

```python
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
```

---

### 🔹 7. Feature & Target Separation

The dataset is divided into input features and output labels.
This separation is required for supervised learning.
Models learn patterns from features to predict the target.

```python
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
```

---

### 🔹 8. Categorical Encoding

Categorical variables are converted into numerical format.
One-hot encoding creates binary columns for each category.
This prevents models from misinterpreting categorical values.

```python
X = pd.get_dummies(X, drop_first=True)
```

---

### 🔹 9. Train-Test Split

The dataset is split into training and testing sets.
Training data is used to build the model.
Testing data evaluates how well the model generalizes.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 🔹 10. Feature Scaling

Feature scaling standardizes the range of variables.
This is important for models like SVM that depend on distance.
It ensures all features contribute equally.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 🔹 11. Model Training

Multiple models are trained to compare performance.
Different algorithms capture patterns differently.
This helps in selecting the best-performing model.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

lr = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
```

---

### 🔹 12. Model Evaluation

Models are evaluated using accuracy score.
This helps compare performance across algorithms.
The best model is selected based on results.

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, lr.predict(X_test)))
print(accuracy_score(y_test, rf.predict(X_test)))
print(accuracy_score(y_test, svm.predict(X_test)))
```

---

### 🔹 13. Confusion Matrix & Classification Report

Detailed evaluation metrics are computed.
Precision, recall, and F1-score provide deeper insights.
This helps understand model strengths and weaknesses.

```python
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, svm.predict(X_test)))
print(classification_report(y_test, svm.predict(X_test)))
```

---

### 🔹 14. ROC Curve

ROC curve evaluates model performance at different thresholds.
It shows trade-off between true positive and false positive rates.
AUC score indicates overall model quality.

```python
from sklearn.metrics import roc_curve, auc
```

---

### 🔹 15. Hyperparameter Tuning

GridSearchCV is used to optimize model parameters.
This improves model performance and generalization.
It helps find the best combination of parameters.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
```

---

### 🔹 16. Feature Importance (Explainable AI)

Feature importance identifies key influencing factors.
This improves interpretability of the model.
It helps understand decision-making.

```python
import pandas as pd

importances = best_rf.feature_importances_
features = X.columns

pd.Series(importances, index=features).nlargest(10)
```

---

### 🔹 17. Cross-Validation

Cross-validation checks model stability across folds.
It reduces risk of overfitting.
Provides a more reliable performance estimate.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(best_rf, X, y, cv=5)
print(scores.mean())
```

---

### 🔹 18. Test Dataset Preprocessing

The unseen dataset is processed similarly to training data.
Consistency ensures correct model predictions.
Feature alignment is crucial.

```python
test_df = pd.get_dummies(test_df, drop_first=True)
test_df = test_df.reindex(columns=X.columns, fill_value=0)
```

---

### 🔹 19. Predictions on Unseen Data

The trained model is applied to new data.
This simulates real-world deployment.
Predictions are generated for each sample.

```python
predictions = best_rf.predict(test_df)
```

---

### 🔹 20. Output Generation

Predictions are saved in CSV format.
This allows submission or further analysis.
Output follows required structure.

```python
output.to_csv("submission.csv", index=False)
```

---

### 🔹 21. Final Results Display

Final accuracy is displayed on test data.
Prediction distribution can also be analyzed.
This summarizes overall performance.

```python
print("Accuracy:", accuracy_score(y_test, best_rf.predict(X_test)))
```

---

## 📊 Key Insights

* SVM achieved the highest accuracy
* Random Forest provided better interpretability
* Credit history and income are key influencing factors
* Model performs better in predicting approvals than rejections

---

## 🎯 Conclusion

This project demonstrates a complete machine learning workflow including preprocessing, feature engineering, model comparison, tuning, and explainability. The system is capable of predicting loan approval and simulating real-world deployment.

---

## 🚀 Future Improvements

* Improve class imbalance handling
* Deploy using Streamlit
* Add real-time prediction system

---

## License
Code is licensed under Apache 2.0.
Dataset belongs to original source and is not redistributed.
