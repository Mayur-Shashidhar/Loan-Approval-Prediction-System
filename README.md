# 🧠 Loan Approval Prediction System

## 📌 Overview

This project implements an end-to-end machine learning pipeline to predict loan approval status based on applicant financial and demographic data. The system includes data preprocessing, feature engineering, model training, evaluation, and prediction on unseen data.

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib
* Scikit-learn

---

## 📂 Project Workflow (Notebook Explanation)

### 🔹 1. Import Libraries & Load Dataset

The dataset is loaded using Pandas, and essential libraries for data processing and machine learning are imported.

---

### 🔹 2. Data Inspection

Initial exploration is performed using:

* `head()` → view sample data
* `info()` → check data types
* `isnull()` → identify missing values

This step helps understand dataset structure and quality.

---

### 🔹 3. Handling Missing Values

Missing values are handled using:

* **Median** for numerical features (robust to outliers)
* **Mode** for categorical features

This ensures no data loss and maintains distribution.

---

### 🔹 4. Feature Engineering

New features are created to improve model performance:

* **Total Income** = Applicant + Coapplicant income
* **EMI** = LoanAmount / Loan Term
* **Debt-to-Income Ratio (DTI)**

These features provide better financial insights.

---

### 🔹 5. Dropping Irrelevant Features

The `Loan_ID` column is removed as it does not contribute to prediction.

---

### 🔹 6. Target Encoding

The target variable `Loan_Status` is converted:

* `Y → 1`
* `N → 0`

This enables numerical processing by ML models.

---

### 🔹 7. Feature & Target Separation

The dataset is split into:

* Features (`X`)
* Target (`y`)

---

### 🔹 8. Categorical Encoding

Categorical variables are transformed using **One-Hot Encoding** to convert them into numerical format.

---

### 🔹 9. Train-Test Split

The dataset is split into:

* 80% Training data
* 20% Testing data

This allows proper model evaluation.

---

### 🔹 10. Feature Scaling

Standardization is applied using `StandardScaler` to normalize feature values, especially important for models like SVM.

---

### 🔹 11. Model Training

Three machine learning models are trained:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)

---

### 🔹 12. Model Evaluation

Models are evaluated using **accuracy score**, and performance is compared to select the best model.

---

### 🔹 13. Confusion Matrix & Classification Report

Detailed evaluation is performed using:

* Confusion Matrix
* Precision, Recall, F1-score

This provides deeper insight beyond accuracy.

---

### 🔹 14. ROC Curve

The ROC curve is plotted to evaluate model performance across different thresholds, and AUC is calculated.

---

### 🔹 15. Hyperparameter Tuning

Random Forest is optimized using **GridSearchCV** to improve performance and generalization.

---

### 🔹 16. Feature Importance (Explainable AI)

Feature importance is visualized to understand which variables influence predictions the most.

---

### 🔹 17. Cross-Validation

K-Fold cross-validation is used to ensure model stability and avoid overfitting.

---

### 🔹 18. Test Dataset Preprocessing

The unseen test dataset is processed using the same steps:

* Missing value handling
* Feature engineering
* Encoding
* Scaling

---

### 🔹 19. Predictions on Unseen Data

The trained model is used to generate predictions on new data.

---

### 🔹 20. Output Generation

Predictions are saved into a CSV file (`submission.csv`) in the required format.

---

### 🔹 21. Final Results Display

* Accuracy on test split is displayed
* Prediction distribution is analyzed for unseen data

---

## 📊 Key Insights

* SVM achieved the highest accuracy
* Random Forest provided better interpretability
* Credit history and income are key influencing factors
* Model performs better in predicting approvals than rejections

---

## 🎯 Conclusion

This project demonstrates a complete machine learning workflow, including preprocessing, model comparison, tuning, and explainability. The system is capable of predicting loan approval and can be extended for real-world financial applications.

---

## 🚀 Future Improvements

* Improve class imbalance handling
* Deploy using Streamlit
* Integrate real-time prediction system

---

## 👨‍💻 Author

Mayur
