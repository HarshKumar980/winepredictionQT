# App UI

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Wine Quality Prediction", layout="wide")
st.title("🍷 Wine Quality Prediction App (KNN)")
st.write("This app predicts wine quality based on physicochemical features using KNN algorithm.")

import pandas as pd
import joblib

data = pd.read_csv("WineQT.csv")
st.write("### Raw Dataset", data.head())

# Preprocessing
X = data.drop(['quality', 'Id'], axis=1)
y = data['quality']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sidebar for K selection
k = st.sidebar.slider("Select number of neighbors (K)", 1, 20, 5)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
st.write("Model and scaler have been saved as 'knn_model.pkl' and 'scaler.pkl'.")

# Accuracy
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {acc:.2f}")

# Predict from user input
st.write("### Try predicting your own wine!")

def user_input_features():
    features = {}
    for col in X.columns:
        features[col] = st.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    return pd.DataFrame([features])

input_df = user_input_features()
input_scaled = scaler.transform(input_df)
prediction = knn.predict(input_scaled)
st.write("### Predicted Wine Quality:", int(prediction[0]))



