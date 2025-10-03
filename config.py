# Configuration file for Laptop Price Prediction

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Random Forest Configuration
RF_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'max_depth': None,
    'min_samples_split': 2
}

# Gradient Boosting Configuration
GB_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'learning_rate': 0.1,
    'max_depth': 3
}

# XGBoost Configuration
XGB_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'learning_rate': 0.1,
    'max_depth': 6
}

# Ridge Regression Configuration
RIDGE_CONFIG = {
    'alpha': 1.0
}

# Lasso Regression Configuration
LASSO_CONFIG = {
    'alpha': 1.0
}

# Data Configuration
DATA_CONFIG = {
    'sample_size': 100,
    'data_path': 'data/laptop_data.csv',
    'model_path': 'models/laptop_price_model.pkl'
}

# Feature Configuration
FEATURES = {
    'numeric': ['ram_gb', 'storage_gb', 'screen_inches', 'processor_gen'],
    'categorical': ['brand', 'processor_type', 'os', 'gpu']
}

# Brand List
BRANDS = ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', 
          'Microsoft', 'Razer', 'Samsung', 'Huawei', 'Xiaomi']

# Processor Types
PROCESSORS = [
    'Intel Core i3 10th Gen', 'Intel Core i5 11th Gen', 'Intel Core i7 11th Gen',
    'Intel Core i9 12th Gen', 'AMD Ryzen 5 5000', 'AMD Ryzen 7 5000',
    'AMD Ryzen 9 5000', 'Apple M1', 'Apple M2'
]

# Operating Systems
OPERATING_SYSTEMS = ['Windows 11', 'Windows 10', 'macOS', 'Ubuntu', 'Linux']

# GPU List
GPUS = [
    'Intel Integrated', 'NVIDIA GTX 1650', 'NVIDIA RTX 3050', 
    'NVIDIA RTX 3060', 'NVIDIA RTX 4060', 'AMD Radeon', 'Apple GPU'
]

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'page_title': 'Laptop Price Predictor',
    'page_icon': '💻',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}
