import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# App Configuration
st.set_page_config(page_title="Biomarker Analytics POC", layout="wide")

# Generate Enhanced Sample Dataset
def generate_large_sample_data():
    np.random.seed(42)
    rows = 1000
    data = pd.DataFrame({
        "Sample ID": [f"SAMPLE_{i}" for i in range(1, rows + 1)],
        "Sample Type": np.random.choice(["Blood", "Urine", "Tissue", "Saliva"], size=rows),
        "Biomarker_1": np.random.normal(loc=50, scale=10, size=rows).round(2),
        "Biomarker_2": np.random.normal(loc=100, scale=20, size=rows).round(2),
        "Biomarker_3": np.random.normal(loc=200, scale=30, size=rows).round(2),
        "Date Collected": pd.date_range("2023-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
        "Patient Age": np.random.randint(20, 80, size=rows),
        "Patient Sex": np.random.choice(["Male", "Female"], size=rows),
    })
    return data

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Navigate", ["Home", "Explore Insights", "Predict Trends"])

# Load Sample Data Automatically
if "data" not in st.session_state:
    st.session_state["data"] = generate_large_sample_data()

# Home Page
if page == "Home":
    st.title("Biomarker Data Analytics POC")
    st.markdown("""
    This application demonstrates data analytics and predictive modeling for biomarker datasets.

    **Features**:
    - Uses a large, realistic sample dataset with 1000 rows.
    - Interactive visualizations and predictive analytics.
    
    **Get Started**:
    - Navigate to **Explore Insights** or **Predict Trends** to see the app in action.
    """)

# Explore Insights Page
elif page == "Explore Insights":
    st.header("📊 Explore Data")
    data = st.session_state["data"]
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    categorical_cols = data.select_dtypes(include="object").columns.tolist()

    st.write("### Numeric Column Visualization")
    selected_col = st.selectbox("Select a numeric column to visualize", numeric_cols)
    if selected_col:
        fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig)

    st.write("### Categorical Column Distribution")
    selected_cat_col = st.selectbox("Select a categorical column to visualize", categorical_cols)
    if selected_cat_col:
        # Create a value count DataFrame with proper columns
        cat_data = data[selected_cat_col].value_counts().reset_index()
        cat_data.columns = [selected_cat_col, "Count"]  # Rename for clarity
        fig = px.bar(cat_data,
                     x=selected_cat_col,
                     y="Count",
                     labels={selected_cat_col: "Category", "Count": "Count"},
                     title=f"Distribution of {selected_cat_col}")
        st.plotly_chart(fig)

    st.write("### Scatter Plot")
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Select X-axis", numeric_cols, key="x_axis")
        y_axis = st.selectbox("Select Y-axis", numeric_cols, key="y_axis")
        if x_axis and y_axis:
            fig = px.scatter(data, x=x_axis, y=y_axis, color="Sample Type",
                             title=f"Scatter Plot: {x_axis} vs {y_axis}")
            st.plotly_chart(fig)

# Predict Trends Page
elif page == "Predict Trends":
    st.header("📈 Predictive Analytics")
    data = st.session_state["data"]
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
