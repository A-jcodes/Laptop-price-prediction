import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.utils import load_data, save_data
from src.train_model import LaptopPriceModel
import os

# Page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">💻 Laptop Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict laptop prices using Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a page:", ["Predict Price", "Dataset Overview", "Model Training", "About"])

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = LaptopPriceModel()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Load or create data
if st.session_state.data is None:
    data_path = 'data/laptop_data.csv'
    if os.path.exists(data_path):
        st.session_state.data = pd.read_csv(data_path)
    else:
        st.session_state.data = load_data()
        os.makedirs('data', exist_ok=True)
        save_data(st.session_state.data, data_path)

# PAGE 1: Predict Price
if page == "Predict Price":
    st.header("🔮 Predict Laptop Price")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Brand", ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', 'Other'])
        processor = st.selectbox("Processor", [
            'Intel Core i3 10th Gen', 'Intel Core i5 11th Gen', 'Intel Core i7 11th Gen',
            'Intel Core i9 12th Gen', 'AMD Ryzen 5 5000', 'AMD Ryzen 7 5000',
            'Apple M1', 'Apple M2'
        ])
        ram = st.selectbox("RAM", [4, 8, 16, 32, 64])
        storage = st.selectbox("Storage (GB)", [256, 512, 1024, 2048])
    
    with col2:
        screen_size = st.selectbox("Screen Size (inches)", [13.3, 14.0, 15.6, 17.3])
        os = st.selectbox("Operating System", ['Windows 11', 'Windows 10', 'macOS', 'Ubuntu', 'Linux'])
        gpu = st.selectbox("GPU", [
            'Intel Integrated', 'NVIDIA GTX 1650', 'NVIDIA RTX 3050',
            'NVIDIA RTX 3060', 'NVIDIA RTX 4060', 'AMD Radeon', 'Apple GPU'
        ])
    
    if st.button("Predict Price", type="primary"):
        # Check if model is trained
        model_path = 'models/laptop_price_model.pkl'
        if os.path.exists(model_path) or st.session_state.model_trained:
            try:
                if not st.session_state.model_trained:
                    st.session_state.model.load_model(model_path)
                
                # Create input dataframe
                input_data = pd.DataFrame([{
                    'laptop_name': f"{brand} Laptop",
                    'brand': brand,
                    'processor': processor,
                    'ram': f"{ram}GB",
                    'storage': f"{storage}GB" if storage < 1024 else f"{storage//1024}TB",
                    'screen_size': f"{screen_size} inches",
                    'os': os,
                    'gpu': gpu
                }])
                
                # Preprocess and predict
                X, _, _ = st.session_state.model.prepare_data(input_data, target_col=None)
                prediction = st.session_state.model.predict(X)[0]
                
                # Display prediction
                st.success(f"### Predicted Price: ${prediction:,.2f}")
                
                # Show price range
                st.info(f"Price Range: ${prediction*0.9:,.2f} - ${prediction*1.1:,.2f}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please train the model first from the 'Model Training' page.")
        else:
            st.warning("⚠️ No trained model found. Please train the model first from the 'Model Training' page.")

# PAGE 2: Dataset Overview
elif page == "Dataset Overview":
    st.header("📊 Dataset Overview")
    
    df = st.session_state.data
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Laptops", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Avg Price", f"${df['price'].mean():,.2f}")
    
    # Display dataframe
    st.subheader("Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Visualizations
    st.subheader("Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Brand Analysis", "Specifications"])
    
    with tab1:
        fig = px.histogram(df, x='price', nbins=30, title='Price Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        brand_avg = df.groupby('brand')['price'].mean().sort_values(ascending=False)
        fig = px.bar(x=brand_avg.index, y=brand_avg.values, 
                     title='Average Price by Brand', labels={'x': 'Brand', 'y': 'Average Price'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            ram_avg = df.groupby('ram')['price'].mean().sort_values(ascending=False)
            fig = px.bar(x=ram_avg.index, y=ram_avg.values,
                        title='Average Price by RAM', labels={'x': 'RAM', 'y': 'Average Price'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            processor_avg = df.groupby('processor')['price'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=processor_avg.index, y=processor_avg.values,
                        title='Top 10 Processors by Price', labels={'x': 'Processor', 'y': 'Average Price'})
            st.plotly_chart(fig, use_container_width=True)

# PAGE 3: Model Training
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    df = st.session_state.data
    
    st.subheader("Training Configuration")
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models... This may take a moment."):
            try:
                # Prepare data
                X, y, feature_names = st.session_state.model.prepare_data(df)
                
                # Train models
                results, (X_train, X_test, y_train, y_test) = st.session_state.model.train_models(X, y, test_size=test_size)
                
                # Save model
                os.makedirs('models', exist_ok=True)
                st.session_state.model.save_model()
                st.session_state.model_trained = True
                
                # Display results
                st.success("✅ Models trained successfully!")
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'RMSE': [r['rmse'] for r in results.values()],
                    'MAE': [r['mae'] for r in results.values()],
                    'R²': [r['r2'] for r in results.values()],
                    'CV RMSE': [r['cv_rmse'] for r in results.values()]
                })
                
                st.subheader("Model Performance Comparison")
                st.dataframe(results_df, use_container_width=True)
                
                # Visualize model comparison
                fig = px.bar(results_df, x='Model', y='R²', title='Model R² Score Comparison')
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                feature_imp = st.session_state.model.get_feature_importance(feature_names)
                if feature_imp is not None:
                    st.subheader("Feature Importance")
                    fig = px.bar(feature_imp, x='importance', y='feature', orientation='h',
                                title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")

# PAGE 4: About
elif page == "About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### Laptop Price Predictor
    
    This is a machine learning application that predicts laptop prices based on various specifications.
    
    #### Features:
    - **Data Preprocessing**: Robust feature engineering to extract meaningful information
    - **Multiple ML Models**: Comparison of Random Forest, Gradient Boosting, XGBoost, Ridge, and Lasso
    - **Interactive UI**: Clean Streamlit interface for easy interaction
    - **Visualizations**: Comprehensive data analysis and insights
    
    #### Tech Stack:
    - Python 3.8+
    - Streamlit for UI
    - Scikit-learn for ML models
    - Pandas & NumPy for data processing
    - Plotly for visualizations
    
    #### How to Use:
    1. **Dataset Overview**: Explore the laptop dataset and visualizations
    2. **Model Training**: Train ML models on the dataset
    3. **Predict Price**: Enter laptop specifications to get price predictions
    
    #### Model Performance:
    The application trains multiple regression models and automatically selects the best performing one
    based on R² score. Feature importance analysis helps understand which specifications most influence price.
    
    ---
    
    **Developed by**: A-jcodes  
    **GitHub**: [Laptop-price-prediction](https://github.com/A-jcodes/Laptop-price-prediction)
    """)
    
    # Add some statistics
    st.subheader("Quick Stats")
    if st.session_state.data is not None:
        df = st.session_state.data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Laptops", len(df))
        with col2:
            st.metric("Brands", df['brand'].nunique())
        with col3:
            st.metric("Min Price", f"${df['price'].min():,.2f}")
        with col4:
            st.metric("Max Price", f"${df['price'].max():,.2f}")
