# -*- coding: utf-8 -*-
"""Medical Cost Prediction App"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf

# --------------------
# Page Config
# --------------------
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="wide")
st.title("💰 Medical Insurance Cost Predictor")

# --------------------
# Load Dataset
# --------------------
@st.cache_data
def load_data():
    url = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
    df = pd.read_csv(url)
    return df

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------
# Data Preprocessing
# --------------------
X = df.drop(columns=["expenses"])
y = np.log(df["expenses"])  # log-transform target

numFeat = ["age", "bmi", "children"]
catFeat = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numFeat),
        ("cat", OneHotEncoder(), catFeat),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# --------------------
# Model Definition
# --------------------
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mae",
        metrics=["mae"],
    )
    return model

# --------------------
# Cache Trained Model (train once on demand)
# --------------------
@st.cache_resource
def get_trained_model():
    model = create_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model

# --------------------
# Prediction UI
# --------------------
st.subheader("🔮 Try Your Own Prediction")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 18, 64, 30)
    bmi = st.slider("BMI", 15.0, 40.0, 25.0)
with col2:
    children = st.number_input("Children", 0, 5, 1)
    sex = st.selectbox("Sex", ["male", "female"])
with col3:
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", df["region"].unique())

if st.button("Predict"):
    # Train model only once
    model = get_trained_model()

    # Prepare input
    input_df = pd.DataFrame([[age, bmi, children, sex, smoker, region]],
                    columns=["age", "bmi", "children", "sex", "smoker", "region"])
    input_transformed = preprocessor.transform(input_df)

    # Predict
    y_log = model.predict(input_transformed).flatten()[0]
    pred = np.exp(y_log)

    st.success(f"💰 Estimated Medical Expense: **${pred:,.2f}**")

    # --------------------
    # Plot distribution with prediction
    # --------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of expenses
    ax.hist(df['expenses'], bins=40, alpha=0.7, color="lightgray", edgecolor="black")
    ax.axvline(pred, color="red", linestyle="--", linewidth=2, label=f"Prediction: ${pred:,.0f}")

    # Calculate percentage of observations around prediction (±10%)
    lower, upper = pred * 0.9, pred * 1.1
    pct = ((df['expenses'] >= lower) & (df['expenses'] <= upper)).mean() * 100

    ax.axvspan(lower, upper, color="skyblue", alpha=0.3, label=f"±10% Range ({pct:.1f}% of data)")
    ax.set_xlabel("Expenses ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Medical Expenses & Predicted Value")
    ax.legend()

    st.pyplot(fig)
