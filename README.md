# WEATHER-PREDICTION-USING-MACHINE-LEARNING-ALGORITHMS

# 🌦️ Weather Prediction Using Machine Learning Algorithms

This project focuses on predicting weather conditions using various machine learning algorithms by analyzing historical weather data. It includes complete steps from data cleaning to model evaluation using classification and regression techniques.

---

## 📁 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Workflow Pipeline](#workflow-pipeline)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Results & Visualization](#results--visualization)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [License](#license)

---

## 📝 Overview

Weather forecasting is a crucial task that supports agriculture, disaster management, transportation, and more. This project builds a robust prediction system using machine learning models such as:

- Linear Regression
- Decision Tree
- Naive Bayes
- Random Forest
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)

We predict various weather metrics such as **temperature**, **humidity**, and **rainfall occurrence** using real-world datasets.

---

## 🧰 Tech Stack

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebook / Google Colab
- Machine Learning Models
- Data Visualization Libraries

---

## 🔄 Workflow Pipeline

1. **Importing the Dataset**
2. **Exploratory Data Analysis (EDA)**
3. **Data Cleaning**
4. **Data Transformation**
5. **Feature Engineering**
6. **Feature Selection**
7. **Model Training**
8. **Hyperparameter Tuning**
9. **Evaluation**
10. **Visualization of Results**

---

## 📊 Dataset

- **Source**: [Kaggle/OpenWeather/historical-data.csv]
- **Fields**: Date, Temperature, Humidity, Wind Speed, Precipitation, Pressure, Cloud Cover, Weather Condition (label)

---

## 🧹 Data Preprocessing

### ✅ Data Cleaning
- Removed null/missing values
- Filtered out extreme/invalid weather values
- Normalized column names and data types

### 🔄 Data Transformation
- Scaled numerical features using StandardScaler/MinMaxScaler
- Converted categorical data using LabelEncoder / OneHotEncoding

### 🧠 Feature Engineering
- Extracted features like day, month, season from the date
- Created weather severity flags based on thresholds (e.g., high humidity, heavy rainfall)

### 📉 Feature Selection
- Correlation matrix to eliminate highly correlated features
- Feature importance ranking using Random Forest and Mutual Information

---

## 🤖 Model Building

Applied and compared multiple ML algorithms:

### ✅ Regression Models
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

### ✅ Classification Models
- **Naive Bayes**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**
- **Logistic Regression**

---

## 📈 Model Evaluation

Used various metrics to evaluate model performance:

### 📊 Regression Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### 🧪 Classification Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

---

## 📉 Results & Visualization

- **Comparative performance plots** for each algorithm
- **Feature importance plots**
- **Confusion matrices and classification reports**
- **Scatter plots, heatmaps, and trend lines** for data insights

---

## ✅ Conclusion

- **Random Forest** and **Decision Tree** performed best for classification tasks like predicting rainfall.
- **Linear Regression** showed strong performance in temperature prediction.
- Proper **feature selection** and **data transformation** significantly boosted accuracy.

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/weather-prediction-ml.git
cd weather-prediction-ml

# Install dependencies
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook Weather_Prediction.ipynb
