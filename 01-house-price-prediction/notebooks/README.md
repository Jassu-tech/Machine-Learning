# 🏠 Project 1: House Price Prediction App

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jassu-tech/Machine-Learning/blob/main/01-house-price-prediction/notebooks/house_price_prediction.ipynb)

An end-to-end Machine Learning web application that predicts California housing prices based on socio-economic and geographical features using an ensemble Random Forest model.

---

## 📌 Project Overview
* **Goal:** Predict median house value for California districts.
* **Dataset:** Scikit-Learn California Housing Dataset (20,640 samples, 8 features).
* **Model Architecture:** Scaled Random Forest Regressor (`n_estimators=100`).
* **Deployment Interface:** Interactive Streamlit GUI.

---

## 📊 Feature Definitions
| Feature | Description |
| :--- | :--- |
| `MedInc` | Median income in block group ($10,000s) |
| `HouseAge` | Median house age in block group (Years) |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Total block group population |
| `AveOccup` | Average number of household members |
| `Latitude` / `Longitude` | Geographical location coordinates |

---

## 🚀 How to Run the Web App Locally

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn streamlit joblib
