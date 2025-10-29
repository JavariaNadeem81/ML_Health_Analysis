# 🧠 Health Disease Prediction Models

This repository contains two end-to-end **Machine Learning projects** focused on early disease prediction using health-related datasets — **Diabetes** and **Breast Cancer**.  
Both projects demonstrate complete ML workflows: from data cleaning and EDA to model training, evaluation, and saving trained models for reuse.

---

## 📁 Projects Included

### 1️⃣ Diabetes Prediction
- **Dataset:** Pima Indians Diabetes Dataset  
- **Goal:** Predict the likelihood of diabetes based on features like Glucose, BMI, Insulin, and Age.  
- **Techniques Used:**
  - Data Cleaning & Exploratory Data Analysis (EDA)
  - Model Comparison (Logistic Regression, Decision Tree, Random Forest, KNN, SVM)
  - Performance Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
  - Visualization: Correlation Heatmap, ROC Curve, Feature Importance
- **Output:** Trained model saved as `RandomForest_model.pkl`

---

### 2️⃣ Breast Cancer Prediction
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset *(from sklearn.datasets)*  
- **Goal:** Classify tumors as **Benign** or **Malignant** based on cell nucleus features.  
- **Techniques Used:**
  - Data Preprocessing & EDA
  - Model Training & Comparison
  - Evaluation using Confusion Matrix, ROC Curve, and Classification Report
- **Output:** Trained model saved as `BreastCancer_model.pkl`

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib  

---

## 💾 Model Reuse Example
Easily reuse trained models without retraining:

```python
import joblib
model = joblib.load("RandomForest_model.pkl")
prediction = model.predict(new_data)

# 🌟 Future Work
Integrate models into a Streamlit web app  
Add patient-level prediction dashboard  
Expand with additional health datasets (Heart Disease, Liver Disease, etc.)

# 👩‍💻 Author
Javaria Nadeem  
Machine Learning Enthusiast | Python Developer  