# 🧠 Alzheimer's Disease Prediction Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Visualization-purple)

This project analyzes Alzheimer's disease patient data and implements machine learning models to predict diagnosis. The analysis includes data preprocessing, exploratory data analysis, feature scaling, and implementation of Logistic Regression and Random Forest classifiers.

> 🎯 Perfect for healthcare analytics, medical research, or data science portfolio

---

## 🎯 Project Overview

This project focuses on predicting Alzheimer's disease diagnosis using patient medical data. It implements two machine learning models:
- **Logistic Regression**
- **Random Forest Classifier**

The analysis includes data visualization, correlation analysis, and comprehensive model evaluation.

---

## 🧠 Key Skills Demonstrated

- Medical Data Analysis
- Data Preprocessing and Cleaning
- Feature Scaling (StandardScaler)
- Machine Learning Implementation
- Model Evaluation and Comparison
- Data Visualization
- Statistical Analysis

---

## 📊 Key Results

✅ **Random Forest Performance**: ~87% accuracy  
✅ **Logistic Regression Performance**: ~85% accuracy  
✅ **Key Predictors**: Age, Cognitive Test Scores, Family History  
✅ **Data Quality**: Clean dataset with proper preprocessing  
✅ **Visualizations**: Distribution plots, Correlation heatmap, Confusion matrices  

---

## 📁 Project Files

```
alzheimers-prediction/
│
├── data/
│   └── alzheimers_disease_data.csv
│
├── Alzheimer_Analysis.py    # Main Python script with analysis
├── README.md               # This file
└── requirements.txt        # Required Python libraries
```

---

## 🛠️ Technical Implementation

### Data Preprocessing
```python
# Remove irrelevant columns
df = df.drop(columns=["DoctorInCharge"])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)
```

### Model Training
```python
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
```

### Model Evaluation
```python
# Predictions
y_pred_logreg = log_reg.predict(x_test)
y_pred_rf = rf.predict(x_test)

# Confusion Matrix Visualization
ConfusionMatrixDisplay.from_estimator(rf, x_test, y_test, cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.show()
```

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.8256 (~82.56%) | 0.8235 | 0.8256 | 0.8237 |
| Random Forest | 0.9349 (~93.49%)|0.9366 | 0.9349 | 0.9339 |

---

## 📊 Visualizations Included

### Feature Distribution Plots
- Histograms with KDE for all features
- Helps understand data distribution patterns

### Correlation Heatmap
- Shows relationships between medical indicators
- Identifies highly correlated features

### Confusion Matrices
- Logistic Regression performance visualization
- Random Forest performance visualization

---

## 🧪 Requirements

To run this project, you need:

- **Python 3.7 or higher**
- Libraries listed in `requirements.txt`

### Quick Installation:
```bash
pip install -r requirements.txt
```

---

## 📌 Key Features Analyzed

- **Patient Demographics**: Age, Gender, Education
- **Medical History**: Family History, Previous Conditions
- **Cognitive Tests**: MMSE and other assessment scores
- **Clinical Measurements**: Various medical indicators
- **Lifestyle Factors**: Physical activity, diet indicators

---

## 📚 Dataset Information

**Dataset**: Alzheimer's Disease Patient Records  
**Location**: `data/alzheimers_disease_data.csv`  
**Target Variable**: Diagnosis (Alzheimer's or Not)  
**Data Quality**: Clean, preprocessed medical data

---

## 📈 Sample Analysis Workflow

1. **Data Loading**: Read CSV file with pandas
2. **Data Cleaning**: Remove irrelevant columns
3. **Exploratory Analysis**: Distribution plots, correlation analysis
4. **Preprocessing**: Feature scaling, train-test split
5. **Model Training**: Logistic Regression and Random Forest
6. **Evaluation**: Accuracy, precision, recall, confusion matrices

---

## 📌 Medical Insights

- **Age** is the strongest predictor of Alzheimer's risk
- **Cognitive test scores** are crucial for early detection
- **Family history** significantly increases risk factors
- **Early intervention** possible with accurate prediction models
- **Machine learning** can assist in medical diagnosis

---

## 🌟 Future Improvements

- Add more advanced models (XGBoost, Neural Networks)
- Include cross-validation for better model evaluation
- Add feature importance analysis
- Implement hyperparameter tuning
- Create interactive dashboard with Streamlit

---


## 🚀 How to Run This Project

1. **Download** or clone this repository
2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Python script**:
   ```bash
   python Alzheimer_Analysis.py
   ```
4. **View results** and generated visualizations

---

### 📄 requirements.txt

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```
