# Setup Guide

Complete step-by-step guide to set up and run the Laptop Price Prediction portfolio project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Training Models](#training-models)
- [Using Jupyter Notebooks](#using-jupyter-notebooks)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** - Python package installer (usually comes with Python)
- **Git** - [Download Git](https://git-scm.com/downloads)
- **Virtual Environment** (recommended) - `python -m venv` or `virtualenv`

### Check Your Installation

```bash
python --version   # Should show Python 3.8+
pip --version      # Should show pip version
git --version      # Should show git version
```

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/A-jcodes/Laptop-price-prediction.git
cd Laptop-price-prediction
```

### Step 2: Create Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when activated.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- joblib
- xgboost

### Step 4: Verify Installation

```bash
python -c "import streamlit; import pandas; import sklearn; print('All packages installed successfully!')"
```

## Running the Application

### Method 1: Using the Run Script (macOS/Linux)

```bash
chmod +x run.sh  # Make script executable (first time only)
./run.sh
```

### Method 2: Direct Streamlit Command

```bash
streamlit run app.py
```

### Method 3: Python Module

```bash
python -m streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Application Features

Once running, you can:

1. **Predict Prices**: 
   - Navigate to "Predict Price" page
   - Enter laptop specifications
   - Get instant price predictions

2. **Explore Data**:
   - Navigate to "Dataset Overview"
   - View interactive charts and statistics
   - Analyze price distributions

3. **Train Models**:
   - Navigate to "Model Training"
   - Adjust test size
   - Train and compare multiple models

4. **Learn More**:
   - Navigate to "About" page
   - View project information and quick stats

## Training Models

### Using the Training Script

```bash
python train.py
```

This will:
1. Load or create the dataset
2. Preprocess the data
3. Train multiple ML models
4. Display performance metrics
5. Save the best model to `models/laptop_price_model.pkl`

### Training Output

You'll see:
- Dataset information
- Model training progress
- Performance metrics (RMSE, MAE, R²)
- Feature importance
- Best model selection

## Using Jupyter Notebooks

### Step 1: Install Jupyter (if not already installed)

```bash
pip install jupyter
```

### Step 2: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 3: Open the EDA Notebook

Navigate to `notebooks/EDA.ipynb` in the Jupyter interface

### Step 4: Run Cells

- Click "Cell" → "Run All" to execute all cells
- Or run cells individually with Shift+Enter

## Project Structure Overview

```
Laptop-price-prediction/
├── app.py              # Main Streamlit application
├── train.py            # CLI training script
├── config.py           # Configuration settings
├── requirements.txt    # Dependencies
├── src/                # Source modules
├── data/               # Datasets
├── models/             # Saved models
└── notebooks/          # Jupyter notebooks
```

## Troubleshooting

### Issue: Module Not Found

**Solution:**
```bash
pip install -r requirements.txt
# or install specific package
pip install package-name
```

### Issue: Port Already in Use

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Issue: Virtual Environment Not Activating

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### Issue: Permission Denied (run.sh)

**Solution:**
```bash
chmod +x run.sh
./run.sh
```

### Issue: Streamlit Not Opening Browser

**Solution:**
```bash
# Manually open browser and go to:
http://localhost:8501
```

### Issue: Model File Not Found

**Solution:**
```bash
# Train the model first
python train.py
```

### Issue: Dataset Not Found

The application will automatically create a sample dataset if none exists. To manually create:

```bash
python -c "from src.utils import load_data, save_data; import os; os.makedirs('data', exist_ok=True); df = load_data(); save_data(df, 'data/laptop_data.csv')"
```

## Custom Dataset

To use your own dataset:

1. Create a CSV file with these columns:
   - laptop_name
   - brand
   - processor
   - ram (e.g., "8GB")
   - storage (e.g., "512GB")
   - screen_size (e.g., "15.6 inches")
   - os
   - gpu
   - price

2. Place it in the `data/` directory as `laptop_data.csv`

3. Run the application or training script

## Advanced Configuration

### Modify Model Parameters

Edit `config.py` to change:
- Model hyperparameters
- Train/test split ratio
- Cross-validation folds
- Feature lists

### Add New Features

1. Update `src/data_preprocessing.py`
2. Add feature extraction logic
3. Update feature list in config

### Add New Models

1. Update `src/train_model.py`
2. Add model to `self.models` dictionary
3. Train and evaluate

## Updating the Project

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/A-jcodes/Laptop-price-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/A-jcodes/Laptop-price-prediction/discussions)
- **Documentation**: See [PORTFOLIO.md](PORTFOLIO.md)

## Next Steps

1. ✅ Set up the project
2. ✅ Run the Streamlit app
3. ✅ Explore the dataset
4. ✅ Train models
5. ✅ Make predictions
6. 📚 Read [PORTFOLIO.md](PORTFOLIO.md) for detailed project overview
7. 🤝 Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

Happy coding! 🚀
