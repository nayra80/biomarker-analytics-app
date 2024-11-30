# File: app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# App Header
st.title("Biomarker Data Analytics POC")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload File", "Explore Data", "Predictive Analytics"])

# Shared Session State
if "data" not in st.session_state:
    st.session_state["data"] = None

# Page 1: Home
if page == "Home":
    st.write("""
    ### Welcome to the Biomarker Data Analytics Proof of Concept!
    This app demonstrates key competencies in data visualization and predictive modeling for clinical sample operations.

    **How to Use This App:**
    1. Navigate to "Upload File" to upload your dataset (CSV format).
    2. Use "Explore Data" to visualize and analyze your data.
    3. Apply "Predictive Analytics" to uncover machine learning insights.

    **Need Help?** Download a [sample dataset](https://github.com/nayra80/biomarker-analytics-app/blob/main/sample_data/biomarker_data.csvm) to get started.
    """)

# Page 2: Upload File
elif page == "Upload File":
    st.header("Upload Your Dataset")
    st.write("""
    **Instructions:**
    - Upload a CSV file.
    - Ensure the dataset includes columns such as `Sample Type`, `Biomarker_1`, `Biomarker_2`, etc.
    - If you don’t have a dataset, download a [sample file](https://your-sample-dataset-link.com).
    """)

    uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state["data"] = data
            st.write("### File Preview")
            st.dataframe(data.head())
            st.success("File uploaded successfully! Navigate to 'Explore Data' to continue.")
        except Exception as e:
            st.error(f"Error uploading file: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

# Page 3: Explore Data
elif page == "Explore Data":
    st.header("Explore Your Dataset")
    if st.session_state["data"] is not None:
        data = st.session_state["data"]
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        categorical_cols = data.select_dtypes(include="object").columns.tolist()

        # Numeric Column Distribution
        st.write("### Numeric Column Distribution")
        num_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
        if num_col:
            fig = px.histogram(data, x=num_col, title=f"Distribution of {num_col}")
            st.plotly_chart(fig)

        # Relationship Visualization
        st.write("### Scatter Plot")
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("X-axis", numeric_cols)
            y_axis = st.selectbox("Y-axis", numeric_cols)
            if x_axis and y_axis:
                fig = px.scatter(data, x=x_axis, y=y_axis, title=f"Relationship: {x_axis} vs {y_axis}")
                st.plotly_chart(fig)
    else:
        st.warning("Please upload a file in 'Upload File' first.")

# Page 4: Predictive Analytics
elif page == "Predictive Analytics":
    st.header("Predictive Analytics")
    if st.session_state["data"] is not None:
        data = st.session_state["data"]
        st.write("### Select Features and Target")
        predictors = st.multiselect("Select Predictor Columns", data.columns)
        target = st.selectbox("Select Target Column", data.columns)

        if predictors and target:
            # Preprocess Data
            processed_data = data.copy()
            for col in processed_data.select_dtypes(include="object").columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col])

            X = processed_data[predictors]
            y = processed_data[target]

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Model
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Model Performance
            mse = mean_squared_error(y_test, predictions)
            st.write(f"### Model Performance")
            st.write(f"Mean Squared Error: {mse}")

            # Results Visualization
            results_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
            st.write("### Actual vs Predicted")
            st.dataframe(results_df)
            fig = px.scatter(results_df, x="Actual", y="Predicted", title="Actual vs Predicted")
            st.plotly_chart(fig)
    else:
        st.warning("Please upload a file in 'Upload File' first.")
