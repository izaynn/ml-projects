import streamlit as st
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
@st.cache_resource
def load_models():
    lr = joblib.load('models/lr_model.pkl')
    knn = joblib.load('models/knn_model.pkl')
    dt = joblib.load('models/dt_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    return lr, knn, dt,scaler,encoder

lr,knn,dt,scaler,encoder = load_models()


st.set_page_config(page_title='Iris Classifier',layout="centered")

st.title("🌿 Iris Flower Species Predictor")

st.markdown("### Enter the Flower measurements below to classify species using **Logistic Regression, KNN and Decision Trees**.")

st.sidebar.header("📏 Input Measurements")
sepal_lenght = st.slider("Sepal Lenght",4.0,8.0,5.8,0.1)
sepal_width = st.slider("Sepal Width",2.0,4.5,3.0,0.1)
petal_lenght = st.slider("Petal Lenght",1.0,7.0,3.8,0.1)
petal_width = st.slider("Petal Width",0.1,2.5,1.2,0.1)

input_df = pd.DataFrame(
    [[sepal_lenght, sepal_width, petal_lenght, petal_width]],
    columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
)
scaled_input = scaler.transform(input_df)

st.subheader("🤖 Model Predictions")

col1,col2,col3 = st.columns(3)

lr_pred = lr.predict(scaled_input)[0]
knn_pred = knn.predict(scaled_input)[0]
dt_pred = dt.predict(input_df)[0]

probs_lr = lr.predict_proba(scaled_input)[0]
probs_knn = knn.predict_proba(scaled_input)[0]
probs_dt = dt.predict_proba(input_df)[0]
with col1:
    st.metric("Logistic Regression",encoder.inverse_transform([lr_pred])[0],
              delta=f"{np.max(probs_lr)*100:.1f}% Confidence.")

with col2:
    st.metric("KNN",encoder.inverse_transform([knn_pred])[0],
              delta=f"{np.max(probs_knn)*100:.1f}% Confidence.")

with col3:
    st.metric("Decision Tree",encoder.inverse_transform([dt_pred])[0],
              delta=f"{np.max(probs_dt)*100:.1f}% Confidence.")