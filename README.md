# SCADA_Optimization_AI
This repository contains an AI-driven approach to optimize SCADA-based production processes. It identifies input-output relationships, determines optimal production values, and enhances efficiency using machine learning models like Random Forest and optimization techniques such as L-BFGS-B. Key insights include feature importance, correlation analysis, and predictive modeling for optimal production.
I'll analyze the provided Jupyter notebook and extract relevant details to generate the GitHub repository name, description, README file, and important code snippets. I'll also include visual plots that should be placed in the README. Let me process the notebook now.

# Production Quality Optimization using Machine Learning

## 📌 Objective:
- Identify which inputs influence which outputs and how.
- Determine the optimal values of the outputs.
- Find out which outputs significantly impact production quality.

## 📊 Methodology:

### **1️⃣ Data Preprocessing**
- Read the dataset from an Excel file.
- Convert column names to strings for better readability.
- Handle missing values using median imputation.
- Convert irregular time formats to datetime using regex.
- Convert numerical features to float.
- Separate input (`X`) and output (`Y`) features.

```python
import pandas as pd

# Load Data
file_path = "Data Prediction Data - Final - Eng.xlsx"
df = pd.read_excel(file_path)

# Convert columns to string for readability
df.columns = df.columns.map(str)

# Handle missing values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert numerical columns to float
df = df.astype(float, errors='ignore')
```

---

### **2️⃣ Correlation Analysis**

- Removed features with zero correlation.
- Selected features with high correlation for analysis.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Compute Correlation Matrix
correlation_matrix = df.corr()

# Plot Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

**📌 Output:**  
![Simplified Correlation](https://github.com/user-attachments/assets/f610194c-157d-4868-ae8c-e2e22f7a9300)

---

### **3️⃣ LSTM Model for Predicting Outputs**
- Applied LSTM for time-series prediction of outputs.
- Achieved **Average Test Huber Loss: 0.0173** and **MSE: 0.0346**.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Compile and Train Model
model.compile(optimizer="adam", loss="huber")
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
```
![image](https://github.com/user-attachments/assets/020ec2d0-0cfd-4513-8bcb-e36a4b2f97b6)

---

### **4️⃣ Feature Importance using Random Forest**
- Applied Random Forest Regressor to determine the most influential input features.
- Plotted feature importance scores.

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Train Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get Feature Importance Scores
feature_importances = rf.feature_importances_

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=X.columns)
plt.xlabel("Feature Importance Score")
plt.ylabel("Input Features")
plt.title("Feature Importance via Random Forest")
plt.show()
```

**📌 Output:**  
![image](https://github.com/user-attachments/assets/7c1c8367-ff44-4dcd-851e-fd4e2f9b0a60)

---

### **5️⃣ Output Optimization using L-BFGS-B**
- Applied `L-BFGS-B` to minimize error and optimize output values.
- Saved the best input values corresponding to optimized outputs in `results.csv`.

```python
from scipy.optimize import minimize

def objective_function(inputs):
    predictions = model.predict(inputs.reshape(1, -1))
    return np.sum((predictions - desired_output)**2)

# Apply Optimization
result = minimize(objective_function, x0=initial_inputs, method='L-BFGS-B')
optimal_inputs = result.x

# Save Results
optimal_df = pd.DataFrame({"Optimized Inputs": optimal_inputs})
optimal_df.to_csv("results.csv", index=False)
```

---

### **6️⃣ Plotting Optimized Output Features**
- Plotted optimized output values as vertical bar plots.

```python
plt.figure(figsize=(10, 5))
plt.bar(optimal_df.index, optimal_df["Optimized Inputs"], color="skyblue")
plt.xlabel("Features")
plt.ylabel("Optimized Values")
plt.title("Optimized Output Features")
plt.xticks(rotation=45)
plt.show()
```

**📌 Output:**  
![image](https://github.com/user-attachments/assets/dcb72dc4-b77f-497b-9c0d-0ddf31cd15ae)

---

## 📄 Results:
- Successfully identified **key input features** influencing production quality.
- Used **LSTM to predict outputs** with high accuracy.
- Determined **most important input variables** using Random Forest.
- **Optimized production outputs** using `L-BFGS-B` optimization.

---
