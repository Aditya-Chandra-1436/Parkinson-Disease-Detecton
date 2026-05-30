import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------

st.markdown(
    """
    <style>
    .main-title{
        text-align:center;
        color:#1565c0;
        font-size:42px;
        font-weight:bold;
    }

    .result-positive{
        background-color:#ffebee;
        color:#c62828;
        padding:20px;
        border-radius:10px;
        font-size:28px;
        font-weight:bold;
        text-align:center;
    }

    .result-negative{
        background-color:#e8f5e9;
        color:#2e7d32;
        padding:20px;
        border-radius:10px;
        font-size:28px;
        font-weight:bold;
        text-align:center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("parkinsons.data")
    return df


# ---------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------

@st.cache_resource
def train_all_models():

    df = load_data()

    X = df.drop(columns=["name", "status"])
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "Gaussian NB": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "predictions": pred
        }

    return (
        df,
        scaler,
        X_test,
        y_test,
        results
    )


# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

def show_confusion_matrix(y_true, y_pred, model_name):

    cm = confusion_matrix(y_true, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(
            x="Predicted",
            y="Actual",
            color="Count"
        ),
        x=["Healthy", "Parkinson"],
        y=["Healthy", "Parkinson"],
        title=f"Confusion Matrix - {model_name}"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    st.markdown(
        '<p class="main-title">🧠 Parkinson Disease Detection System</p>',
        unsafe_allow_html=True
    )

    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Home",
            "Model Performance",
            "Disease Prediction",
            "About"
        ]
    )

    (
        df,
        scaler,
        X_test,
        y_test,
        results
    ) = train_all_models()

    # ---------------------------------------------------
    # HOME
    # ---------------------------------------------------

    if page == "Home":

        st.header("Welcome")

        st.write(
            """
            This application uses machine learning models
            to detect Parkinson's Disease using voice features.

            Dataset: UCI Parkinson's Dataset

            Models Used:
            - Logistic Regression
            - KNN
            - SVM
            - Gaussian NB
            - Decision Tree
            - Random Forest
            """
        )

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write(f"Total Records: {len(df)}")

    # ---------------------------------------------------
    # MODEL PERFORMANCE
    # ---------------------------------------------------

    elif page == "Model Performance":

        st.header("Model Performance")

        metrics_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": [
                results[m]["accuracy"]
                for m in results
            ],
            "Precision": [
                results[m]["precision"]
                for m in results
            ],
            "Recall": [
                results[m]["recall"]
                for m in results
            ],
            "F1 Score": [
                results[m]["f1"]
                for m in results
            ]
        })

        st.dataframe(metrics_df)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=metrics_df["Model"],
                y=metrics_df["Accuracy"],
                name="Accuracy"
            )
        )

        fig.update_layout(
            title="Accuracy Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        best_model = metrics_df.loc[
            metrics_df["Accuracy"].idxmax(),
            "Model"
        ]

        st.success(
            f"Best Model: {best_model}"
        )

        selected_model = st.selectbox(
            "Select Model",
            list(results.keys())
        )

        show_confusion_matrix(
            y_test,
            results[selected_model]["predictions"],
            selected_model
        )

    # ---------------------------------------------------
    # DISEASE PREDICTION
    # ---------------------------------------------------

    elif page == "Disease Prediction":

        st.header("Disease Prediction Using Dataset Row")

        row_number = st.number_input(
            "Enter Row No. for Specific Patient Disease Detection" ,
            min_value=1,
            max_value=len(df),
            value=1,
            
        )

        if st.button("Predict"):

            selected_row = df.iloc[row_number -1]

            patient_name = selected_row["name"]

            actual_status = selected_row["status"]

            input_data = (
                selected_row
                .drop(labels=["name", "status"])
                .values
                .reshape(1, -1)
            )

            input_scaled = scaler.transform(input_data)

            st.subheader("Selected Record")

            st.write(
                f"Patient ID: {patient_name}"
            )

            feature_df = pd.DataFrame(
                {
                    "Feature":
                    df.drop(
                        columns=["name", "status"]
                    ).columns,
                    "Value":
                    input_data.flatten()
                }
            )

            st.dataframe(feature_df)

            st.subheader(
                "Individual Model Predictions"
            )

            votes = []

            prediction_table = []

            for model_name, data in results.items():

                model = data["model"]

                prediction = model.predict(
                    input_scaled
                )[0]

                votes.append(prediction)

                prediction_table.append(
                    [
                        model_name,
                        "Parkinson Detected"
                        if prediction == 1
                        else "Parkinson Not Detected"
                    ]
                )

            prediction_df = pd.DataFrame(
                prediction_table,
                columns=[
                    "Model",
                    "Prediction"
                ]
            )

            st.dataframe(prediction_df)

            majority_prediction = (
                1
                if sum(votes)
                > len(votes) / 2
                else 0
            )

            st.subheader("Final Result")

            col1, col2 = st.columns(2)

            with col1:

                actual_text = (
                    "Parkinson"
                    if actual_status == 1
                    else "Healthy"
                )

                st.info(
                    f"Actual Dataset Label: {actual_text}"
                )

            with col2:

                predicted_text = (
                    "Parkinson"
                    if majority_prediction == 1
                    else "Healthy"
                )

                st.info(
                    f"Predicted Label: {predicted_text}"
                )

            if majority_prediction == 1:

                st.markdown(
                    """
                    <div class="result-positive">
                    ⚠️ Parkinson's Disease Detected
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result-negative">
                    ✅ Healthy: "Parkinson's Disease Not Detected"
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if majority_prediction == actual_status:

                st.success(
                    "Prediction matches actual dataset label."
                )

            else:

                st.warning(
                    "Prediction does not match actual dataset label."
                )

    # ---------------------------------------------------
    # ABOUT
    # ---------------------------------------------------

    elif page == "About":

        st.header("About")

        st.write(
            """
            Parkinson's Disease Detection System

            Dataset:
            UCI Machine Learning Repository
            Parkinson's Dataset

            Features:
            22 biomedical voice measurements

            Models:
            Logistic Regression,
            KNN,
            SVM,
            Gaussian NB,
            Decision Tree,
            Random Forest

            This project is intended for
            educational and research purposes.
            """
        )


if __name__ == "__main__":
    main()