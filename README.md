# Laptop-price-prediction

This is a machine learning project that predicts laptop prices based on specifications.  
The project covers data preprocessing, feature engineering, model training, and deployment using a Streamlit app.

---

##  Project Overview
The price of a laptop depends on multiple features such as brand, RAM, storage, display resolution, CPU, and GPU.  
This project uses **machine learning** to predict laptop prices with accuracy and provide a deployable application for users.

**Goal:**  
- Analyze laptop specifications and their effect on price.  
- Train predictive models.  
- Deploy an interactive app to estimate price ranges.  

---

## Dataset
- **File:** `laptop_data.csv`  
- **Shape:** ~1,300 rows × 12 columns  
- **Features:** Brand, Type, RAM, Weight, Touchscreen, IPS, Screen Size, Resolution, CPU, HDD, SSD, GPU, OS.  
- **Target:** Price (in USD).  

---

## Exploratory Data Analysis (EDA)
Some highlights from the analysis:
- Higher **RAM** and **SSD** significantly increase price.  
- Premium brands (Apple, MSI, Razer) are priced higher.  
- **IPS panels** and higher **PPI** values are associated with higher prices.  

**Example Plots:**  
- Price distribution histogram.  
- Brand vs. Average Price bar chart.  
- Correlation heatmap.  

---

##  Feature Engineering
- Extracted **CPU brand** and **GPU brand**.  
- Derived **PPI (pixels per inch)** from resolution and screen size.  
- One-Hot Encoding for categorical features.  
- Numerical scaling for continuous variables.  

---

##  Model Training
Models tested:
- Linear Regression  
- Decision Trees  
- Random Forest  
- Gradient Boosting  

**Best Model:**  
- **Random Forest (v2)** pipeline performed best.  
- Evaluation: MAE ≈ 0.21, meaning the model predicts prices within ±0.21 of the log price scale.  

---

##  Streamlit App
Interactive app allows users to input specs and predict laptop prices.  

**Features:**
- User selects Brand, Type, RAM, CPU, GPU, Storage, etc.  
- Predicts price range with ±$1000 margin.

Results

Strong relationship between specs and price.

Example: A laptop with 16GB RAM, SSD storage, and IPS panel will predict much higher price compared to base models.

Future Work

Update dataset with recent laptop models.

Add feature importance visualization.

Deploy on Heroku/Render for live access.

Run locally:
```bash
streamlit run app.py


