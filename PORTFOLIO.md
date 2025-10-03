# Project Overview

## 📁 Complete Portfolio Structure

```
Laptop-price-prediction/
│
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 🚀 run.sh                       # Quick start script
│
├── 💻 app.py                       # Main Streamlit application
├── 🔧 train.py                     # Model training script
│
├── 📂 src/                         # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data preprocessing & feature engineering
│   ├── train_model.py              # ML model training & evaluation
│   └── utils.py                    # Utility functions & data generation
│
├── 📂 data/                        # Dataset directory
│   ├── README.md
│   └── laptop_data.csv             # Sample laptop dataset
│
├── 📂 models/                      # Saved ML models
│   └── README.md
│
└── 📂 notebooks/                   # Jupyter notebooks
    ├── README.md
    └── EDA.ipynb                   # Exploratory Data Analysis
```

## 🎯 Key Components

### 1. **Streamlit Application (`app.py`)**
   - Interactive web interface
   - 4 main pages: Predict, Dataset Overview, Model Training, About
   - Real-time price predictions
   - Data visualizations with Plotly
   - Model performance metrics

### 2. **Data Processing (`src/data_preprocessing.py`)**
   - Brand extraction from laptop names
   - RAM/Storage size parsing (GB/TB)
   - Screen size extraction (inches)
   - Processor type and generation detection
   - Label encoding for categorical features
   - Standardized feature engineering pipeline

### 3. **Model Training (`src/train_model.py`)**
   - Multiple ML algorithms:
     * Random Forest Regressor
     * Gradient Boosting Regressor
     * XGBoost Regressor
     * Ridge Regression
     * Lasso Regression
   - Cross-validation
   - Model comparison and selection
   - Feature importance analysis
   - Model persistence (joblib)

### 4. **Utilities (`src/utils.py`)**
   - Sample dataset generation
   - Data loading and saving
   - Configurable data creation

### 5. **Training Script (`train.py`)**
   - Command-line model training
   - Dataset preparation
   - Model evaluation
   - Feature importance display

### 6. **Exploratory Analysis (`notebooks/EDA.ipynb`)**
   - Comprehensive data exploration
   - Price distribution analysis
   - Brand and specification comparisons
   - Correlation analysis
   - Visual insights

## 🔄 Workflow

```
1. Data Input → 2. Preprocessing → 3. Feature Engineering
                                          ↓
6. Deployment ← 5. Model Selection ← 4. Model Training
```

## 📊 Features Extracted

| Original Feature | Extracted Features |
|-----------------|-------------------|
| laptop_name | brand |
| ram | ram_gb (numeric) |
| storage | storage_gb (numeric) |
| screen_size | screen_inches (numeric) |
| processor | processor_type, processor_gen |
| os, gpu | encoded categorical values |

## 🚀 Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/A-jcodes/Laptop-price-prediction.git
cd Laptop-price-prediction
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Or use the run script
./run.sh

# Train model via CLI
python train.py
```

## 📈 Model Performance Metrics

The application evaluates models using:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of Determination
- **CV RMSE**: Cross-Validation RMSE

## 🎨 UI Pages

1. **Predict Price**: Enter specs → Get instant price prediction
2. **Dataset Overview**: Explore data with interactive charts
3. **Model Training**: Train and compare ML models
4. **About**: Project info and quick stats

## 🔧 Technologies

- **Backend**: Python, scikit-learn, XGBoost
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Persistence**: Joblib

## 📝 Dataset Schema

```
laptop_name  : String - Full laptop name
brand        : String - Manufacturer brand
processor    : String - CPU specifications
ram          : String - RAM size (e.g., "8GB")
storage      : String - Storage capacity (e.g., "512GB")
screen_size  : String - Display size (e.g., "15.6 inches")
os           : String - Operating system
gpu          : String - Graphics card
price        : Float - Price in USD
```

## 🎓 Learning Outcomes

This portfolio demonstrates:
- ✅ End-to-end ML project development
- ✅ Feature engineering techniques
- ✅ Model comparison and selection
- ✅ Web application development
- ✅ Data visualization
- ✅ Code organization and documentation
- ✅ Version control best practices
