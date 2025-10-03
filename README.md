# 💻 Laptop Price Prediction

A machine learning portfolio project that predicts laptop prices based on specifications using classical ML algorithms with robust feature engineering and a clean Streamlit UI.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: PEP8](https://img.shields.io/badge/code%20style-PEP8-blue.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/A-jcodes/Laptop-price-prediction/graphs/commit-activity)

## 🌟 Features

- **Interactive Web Application**: Clean and intuitive Streamlit UI for price predictions
- **Multiple ML Models**: Comparison of Random Forest, Gradient Boosting, XGBoost, Ridge, and Lasso regressors
- **Robust Feature Engineering**: Extract meaningful features from laptop specifications
- **Data Visualization**: Comprehensive charts and graphs for data analysis
- **Model Evaluation**: Detailed performance metrics and feature importance analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/A-jcodes/Laptop-price-prediction.git
cd Laptop-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📚 Documentation

- **[Setup Guide](SETUP.md)** - Detailed installation and setup instructions
- **[Portfolio Overview](PORTFOLIO.md)** - Complete project structure and workflow
- **[Features Documentation](FEATURES.md)** - Comprehensive feature list
- **[API Reference](API.md)** - Complete API documentation for all modules
- **[Demo Guide](DEMO.md)** - Screenshots and usage walkthrough
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project

## 📊 Dataset

The project includes a sample dataset generator that creates synthetic laptop data with realistic specifications:
- **Brands**: Dell, HP, Lenovo, Asus, Acer, Apple, MSI
- **Processors**: Intel Core series, AMD Ryzen, Apple Silicon
- **RAM**: 4GB to 64GB
- **Storage**: 256GB to 2TB
- **Screen Sizes**: 13.3" to 17.3"
- **GPU**: Integrated and Dedicated graphics cards
- **Operating Systems**: Windows, macOS, Linux

You can also use your own dataset by placing it in the `data/` directory.

## 🏗️ Project Structure

```
Laptop-price-prediction/
│
├── app.py                      # Main Streamlit application
├── train.py                    # Model training script
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data preprocessing and feature engineering
│   ├── train_model.py          # Model training and evaluation
│   └── utils.py                # Utility functions
│
├── data/                       # Data directory
│   └── laptop_data.csv         # Dataset (generated/provided)
│
├── models/                     # Saved models
│   └── laptop_price_model.pkl  # Trained model
│
└── notebooks/                  # Jupyter notebooks
    └── EDA.ipynb               # Exploratory Data Analysis
```

## 🔧 Usage

### Web Application

The Streamlit app provides four main pages:

1. **Predict Price**: Enter laptop specifications to get price predictions
2. **Dataset Overview**: Explore the dataset with interactive visualizations
3. **Model Training**: Train and compare multiple ML models
4. **About**: Project information and documentation

### Command Line Training

Train the model using the command line:

```bash
python train.py
```

This will:
- Load or generate the dataset
- Preprocess the data
- Train multiple ML models
- Save the best performing model
- Display performance metrics and feature importance

### Jupyter Notebook

Explore the data using the provided Jupyter notebook:

```bash
jupyter notebook notebooks/EDA.ipynb
```

## 🤖 Models

The project implements and compares multiple regression algorithms:

- **Random Forest Regressor**: Ensemble learning method using decision trees
- **Gradient Boosting Regressor**: Sequential ensemble technique
- **XGBoost Regressor**: Optimized gradient boosting implementation
- **Ridge Regression**: Linear regression with L2 regularization
- **Lasso Regression**: Linear regression with L1 regularization

The best model is automatically selected based on R² score.

## 📈 Performance Metrics

Models are evaluated using:
- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average magnitude of errors
- **R² Score**: Proportion of variance explained
- **Cross-Validation RMSE**: 5-fold CV for robust evaluation

## 🔍 Feature Engineering

The preprocessing pipeline extracts and creates features:
- Brand extraction from laptop names
- RAM size (in GB)
- Storage capacity (in GB)
- Screen size (in inches)
- Processor type and generation
- Label encoding for categorical variables

## 📝 API Reference

### DataPreprocessor Class

```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df_processed = preprocessor.preprocess_data(df)
```

### LaptopPriceModel Class

```python
from src.train_model import LaptopPriceModel

model = LaptopPriceModel()
X, y, features = model.prepare_data(df)
results, data = model.train_models(X, y)
model.save_model('models/my_model.pkl')
```

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Statistical data visualization
- **Joblib**: Model serialization

## 📊 Visualizations

The application provides various visualizations:
- Price distribution histograms
- Brand-wise price comparisons
- Specification impact analysis
- Feature importance charts
- Model performance comparisons
- Correlation heatmaps

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**A-jcodes**

- GitHub: [@A-jcodes](https://github.com/A-jcodes)

## 🙏 Acknowledgments

- Scikit-learn documentation
- Streamlit community
- XGBoost developers
- Open-source ML community

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ Star this repository if you find it helpful!
