import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# App Configuration
st.set_page_config(page_title="Biomarker Analytics POC", layout="wide")

# Set Base Directory (starting from where the app script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "sample_data/sample_biomarker_data.csv")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Navigate", ["Home", "Upload or Use Sample Data", "Explore Insights", "Predict Trends"])

# Load Sample Data Function
def load_sample_data():
    if os.path.exists(SAMPLE_DATA_PATH):
        data = pd.read_csv(SAMPLE_DATA_PATH)
        st.success("Sample dataset loaded successfully!")
        return data
    else:
        st.error(f"Sample dataset file not found at: {SAMPLE_DATA_PATH}")
        return None

# Home Page
if page == "Home":
    st.title("Biomarker Data Analytics POC")
    st.markdown("""
    This application demonstrates data analytics and predictive modeling for biomarker datasets.

    **Features**:
    - Seamless demo experience using preloaded sample data.
    - Optional file upload for real-world applications.
    - Interactive visualizations and predictive analytics.

    **Get Started**:
    - Navigate to **Upload or Use Sample Data** to load a dataset.
    - Explore features in **Explore Insights** and **Predict Trends**.
    """)

# Upload or Use Sample Data Page
elif page == "Upload or Use Sample Data":
    st.header("📂 Upload Your Dataset or Use Sample Data")
    st.write("""
    **Options**:
    - Use the preloaded sample dataset for a quick demo.
    - Upload your own CSV file to see the app's flexibility.

    **Dataset Requirements**:
    - File format: CSV
    - Required Columns:
        - `Sample Type` (e.g., Blood, Urine)
        - `Biomarker_1`, `Biomarker_2`, `Biomarker_3` (numeric values)
        - `Date Collected` (YYYY-MM-DD format)
    """)

    # Sample Data Button
    if st.button("Use Sample Data"):
        st.session_state["data"] = load_sample_data()

    # File Upload Option
    uploaded_file = st.file_uploader("Or, upload your CSV file", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state["data"] = data
            st.success("File uploaded successfully!")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error: {e}")

# Explore Insights Page
elif page == "Explore Insights":
    st.header("📊 Explore Data")
    data = st.session_state.get("data")
    if data is not None:
        numeric_cols = data.select_dtypes(include="number").columns.tolist()

        st.write("### Numeric Column Visualization")
        selected_col = st.selectbox("Select a column to visualize", numeric_cols)
        if selected_col:
            fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig)

        st.write("### Scatter Plot")
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("Select X-axis", numeric_cols, key="x_axis")
            y_axis = st.selectbox("Select Y-axis", numeric_cols, key="y_axis")
            if x_axis and y_axis:
                fig = px.scatter(data, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}")
                st.plotly_chart(fig)
    else:
        st.warning("Please load a dataset first.")

# Predict Trends Page
elif page == "Predict Trends":
    st.header("📈 Predictive Analytics")
    data = st.session_state.get("data")
    if data is not None:
        predictors = st.multiselect("Select Predictor Columns", data.columns)
        target = st.selectbox("Select Target Column", data.columns)

        if predictors and target:
            # Encode categorical data
            processed_data = data.copy()
            for col in processed_data.select_dtypes(include="object").columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col])

            X = processed_data[predictors]
            y = processed_data[target]

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Selection
            model_choice = st.radio("Choose a Model", ["Linear Regression", "Decision Tree", "Random Forest"])
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor()
            else:
                model = RandomForestRegressor()

            # Train the model
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Performance Metrics
            mse = mean_squared_error(y_test, predictions)
            st.write(f"### Model: {model_choice}")
            st.write(f"Mean Squared Error: {mse}")

            # Results Visualization
            results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
            st.write("### Prediction Results")
            st.dataframe(results.head())
            fig = px.scatter(results, x="Actual", y="Predicted", title="Actual vs Predicted")
            st.plotly_chart(fig)
    else:
        st.warning("Please load a dataset first.")
