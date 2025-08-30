# 🙏 Predicting Crowd Levels in Religious Places

This project uses **Machine Learning** (Classification + Regression) to **predict crowd levels** and **visitor counts** in religious places based on factors such as country, place, holiday, and other conditions.  
The project is implemented in both **Jupyter Notebook (`main.ipynb`)** and a standalone **Python script (`main.py`)**.

---

## 📌 Objective
The main goal of this project is to:
- Predict **crowd levels** (e.g., low, medium, high) at religious places.  
- Estimate the **visitor count** using regression models.  
- Help with **management and planning** (e.g., security, facilities, traffic control).  

---

## 📊 Dataset
File: **`main.2 - Sheet1.csv`**  

Features include:
- **Country**
- **Holiday**
- **Place**
- **Crowd Level** (categorical: e.g., Low, Medium, High)
- **Visitor Count** (numeric)  

---

## 🔧 Technologies Used
- Python 🐍  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib (for saving models/encoders)  
- Matplotlib & Seaborn (for data visualization)  
- Jupyter Notebook  

---

## 🚀 Steps in the Project
1. **Data Preprocessing**  
   - Handled categorical features using **Label Encoding**.  
   - Cleaned and structured dataset.  

2. **Model Building**  
   - **Classification Model** → Predicts crowd levels.  
   - **Regression Model** → Predicts visitor counts.  

3. **Model Saving**  
   - Stored trained models (`classification_model.joblib`, `regression_model.joblib`).  
   - Saved label encoders for each categorical column.  

4. **Prediction**  
   - Models can be reused to make predictions on new data.  

---

## 📂 Project Structure
