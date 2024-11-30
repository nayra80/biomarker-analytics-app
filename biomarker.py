# File: enhanced_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Helper Functions
def preprocess_data(data):
    """Preprocess data by encoding categorical variables and handling missing values."""
    data = data.copy()
    for col in data.select_dtypes(include='object').columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    return data

def plot_relationship(data, x_col, y_col):
    """Plot scatter plot for relationships between features."""
    fig = px.scatter(data, x=x_col, y=y_col, trendline="ols", title=f"Relationship: {x_col} vs {y_col}")
    st.plotly_chart(fig)

# Page Title
st.title("Enhanced Sample Operations & Biomarker Data Analytics App")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Upload", "Exploratory Dashboard", "Predictive Analytics"])

if page == "Home":
    st.write("""
    ## Welcome!
    This app showcases sample operations and biomarker data analytics capabilities:
    - Upload your dataset.
    - Explore and visualize the data.
    - Apply predictive analytics to generate insights.
    Use the sidebar to navigate between features.
    """)

elif page == "Data Upload":
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Upload your CSV/XLSX file", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(data)
        st.session_state['data'] = data

elif page == "Exploratory Dashboard":
    st.header("Data Exploration Dashboard")
    if 'data' in st.session_state:
        data = st.session_state['data']
        numeric_columns = data.select_dtypes(include='number').columns.tolist()
        categorical_columns = data.select_dtypes(include='object').columns.tolist()

        st.write("### Summary Statistics")
        st.write(data.describe())

        st.write("### Numeric Column Distribution")
        num_col = st.selectbox("Select Numeric Column", numeric_columns)
        if num_col:
            fig = px.histogram(data, x=num_col, title=f"Distribution of {num_col}")
            st.plotly_chart(fig)

        st.write("### Relationship Visualization")
        if len(numeric_columns) >= 2:
            x_col = st.selectbox("X-Axis Column", numeric_columns)
            y_col = st.selectbox("Y-Axis Column", numeric_columns)
            if x_col and y_col:
                plot_relationship(data, x_col, y_col)
    else:
        st.warning("Please upload data first.")

elif page == "Predictive Analytics":
    st.header("Predictive Analytics")
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("### Select Features for Prediction")
        predictors = st.multiselect("Predictor Columns", data.columns)
        target = st.selectbox("Target Column", data.columns)

        if predictors and target:
            # Preprocess data
            processed_data = preprocess_data(data)
            X = processed_data[predictors]
            y = processed_data[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Selection
            st.write("### Select Model")
            model_name = st.radio("Model", ["Linear Regression", "Decision Tree", "Random Forest"])

            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_name == "Random Forest":
                model = RandomForestRegressor()

            # Train and Evaluate Model
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            st.write(f"### Model Performance: {model_name}")
            st.write(f"Mean Squared Error: {mse}")

            st.write("### Prediction vs Actuals")
            results_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
            st.write(results_df.head())
            fig = px.scatter(results_df, x="Actual", y="Predicted", title="Actual vs Predicted")
            st.plotly_chart(fig)

    else:
        st.warning("Please upload data first.")
