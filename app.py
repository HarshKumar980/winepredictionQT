import gradio as gr
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Define the prediction function
def predict_quality(*features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return round(prediction, 2)

# Define feature names (based on WineQT.csv)
feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

# Create inputs for Gradio interface
inputs = [gr.Number(label=feature) for feature in feature_names]
output = gr.Number(label="Predicted Quality")

# Launch the interface
demo = gr.Interface(fn=predict_quality, inputs=inputs, outputs=output, title="Wine Quality Predictor")
demo.launch()
