# API Reference

Documentation for the core modules and functions in the Laptop Price Prediction project.

## Table of Contents
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Utilities](#utilities)
- [Configuration](#configuration)

---

## Data Preprocessing

### Module: `src.data_preprocessing`

#### Class: `DataPreprocessor`

Main class for data preprocessing and feature engineering.

##### Constructor

```python
DataPreprocessor()
```

Initializes the preprocessor with empty label encoders and a standard scaler.

##### Methods

###### `extract_brand(laptop_name: str) -> str`

Extracts the brand name from a laptop name.

**Parameters:**
- `laptop_name` (str): Full laptop name

**Returns:**
- str: Extracted brand name or 'Unknown'

**Example:**
```python
preprocessor = DataPreprocessor()
brand = preprocessor.extract_brand("Dell Inspiron 15")  # Returns: "Dell"
```

---

###### `extract_ram(ram_str: str) -> int`

Extracts RAM size in GB from string format.

**Parameters:**
- `ram_str` (str): RAM string (e.g., "8GB", "16GB DDR4")

**Returns:**
- int: RAM size in GB

**Example:**
```python
ram_gb = preprocessor.extract_ram("16GB DDR4")  # Returns: 16
```

---

###### `extract_storage(storage_str: str) -> int`

Extracts storage size in GB.

**Parameters:**
- `storage_str` (str): Storage string (e.g., "512GB", "1TB SSD")

**Returns:**
- int: Storage size in GB

**Example:**
```python
storage_gb = preprocessor.extract_storage("1TB SSD")  # Returns: 1024
```

---

###### `extract_screen_size(screen_str: str) -> float`

Extracts screen size in inches.

**Parameters:**
- `screen_str` (str): Screen size string (e.g., "15.6 inches")

**Returns:**
- float: Screen size in inches

**Example:**
```python
size = preprocessor.extract_screen_size("15.6 inches")  # Returns: 15.6
```

---

###### `extract_processor_info(processor_str: str) -> tuple`

Extracts processor type and generation.

**Parameters:**
- `processor_str` (str): Processor description

**Returns:**
- tuple: (processor_type: str, generation: int)

**Example:**
```python
proc_type, gen = preprocessor.extract_processor_info("Intel Core i7 11th Gen")
# Returns: ("i7", 11)
```

---

###### `preprocess_data(df: pd.DataFrame) -> pd.DataFrame`

Main preprocessing function that applies all transformations.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe with raw laptop data

**Returns:**
- pd.DataFrame: Processed dataframe with engineered features

**Example:**
```python
df_processed = preprocessor.preprocess_data(df_raw)
```

---

###### `get_feature_columns() -> list`

Returns the list of feature columns used for model training.

**Returns:**
- list: Feature column names

**Example:**
```python
features = preprocessor.get_feature_columns()
# Returns: ['ram_gb', 'storage_gb', 'screen_inches', ...]
```

---

## Model Training

### Module: `src.train_model`

#### Class: `LaptopPriceModel`

Main class for training and evaluating ML models.

##### Constructor

```python
LaptopPriceModel()
```

Initializes the model trainer with multiple regression algorithms.

##### Methods

###### `prepare_data(df: pd.DataFrame, target_col: str = 'price') -> tuple`

Prepares data for model training.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `target_col` (str, optional): Target column name. Default: 'price'

**Returns:**
- tuple: (X: features, y: target, feature_names: list)

**Example:**
```python
model = LaptopPriceModel()
X, y, features = model.prepare_data(df)
```

---

###### `train_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple`

Trains multiple models and selects the best one.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable
- `test_size` (float, optional): Test set proportion. Default: 0.2

**Returns:**
- tuple: (results: dict, data_splits: tuple)

**Example:**
```python
results, (X_train, X_test, y_train, y_test) = model.train_models(X, y, test_size=0.2)
```

---

###### `save_model(filepath: str = 'models/laptop_price_model.pkl')`

Saves the trained model and preprocessor.

**Parameters:**
- `filepath` (str, optional): Path to save model. Default: 'models/laptop_price_model.pkl'

**Example:**
```python
model.save_model('models/my_model.pkl')
```

---

###### `load_model(filepath: str = 'models/laptop_price_model.pkl')`

Loads a saved model and preprocessor.

**Parameters:**
- `filepath` (str, optional): Path to model file. Default: 'models/laptop_price_model.pkl'

**Example:**
```python
model.load_model('models/my_model.pkl')
```

---

###### `predict(X: pd.DataFrame) -> np.ndarray`

Makes predictions using the trained model.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix

**Returns:**
- np.ndarray: Predicted prices

**Example:**
```python
predictions = model.predict(X_test)
```

---

###### `get_feature_importance(feature_names: list) -> pd.DataFrame`

Gets feature importance from the model (if available).

**Parameters:**
- `feature_names` (list): List of feature names

**Returns:**
- pd.DataFrame: Feature importance dataframe or None

**Example:**
```python
importance = model.get_feature_importance(feature_names)
print(importance)
```

---

## Utilities

### Module: `src.utils`

#### Functions

##### `create_sample_dataset(n_samples: int = 100) -> pd.DataFrame`

Creates a sample laptop dataset.

**Parameters:**
- `n_samples` (int, optional): Number of samples to generate. Default: 100

**Returns:**
- pd.DataFrame: Sample dataset

**Example:**
```python
from src.utils import create_sample_dataset

df = create_sample_dataset(n_samples=200)
print(df.shape)  # (200, 9)
```

---

##### `load_data(filepath: str = None) -> pd.DataFrame`

Loads data from file or creates sample dataset.

**Parameters:**
- `filepath` (str, optional): Path to CSV file. If None, creates sample data

**Returns:**
- pd.DataFrame: Loaded or generated dataset

**Example:**
```python
from src.utils import load_data

df = load_data('data/laptop_data.csv')
# or
df = load_data()  # Creates sample data
```

---

##### `save_data(df: pd.DataFrame, filepath: str = 'data/laptop_data.csv')`

Saves dataframe to CSV.

**Parameters:**
- `df` (pd.DataFrame): Dataframe to save
- `filepath` (str, optional): Save path. Default: 'data/laptop_data.csv'

**Example:**
```python
from src.utils import save_data

save_data(df, 'data/my_laptop_data.csv')
```

---

## Configuration

### Module: `config`

#### Variables

##### Model Configuration

```python
MODEL_CONFIG = {
    'test_size': 0.2,           # Train/test split ratio
    'random_state': 42,         # Random seed
    'cv_folds': 5              # Cross-validation folds
}
```

##### Algorithm Configurations

```python
RF_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'max_depth': None,
    'min_samples_split': 2
}

GB_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'learning_rate': 0.1,
    'max_depth': 3
}

XGB_CONFIG = {
    'n_estimators': 100,
    'random_state': 42,
    'learning_rate': 0.1,
    'max_depth': 6
}
```

##### Data Configuration

```python
DATA_CONFIG = {
    'sample_size': 100,
    'data_path': 'data/laptop_data.csv',
    'model_path': 'models/laptop_price_model.pkl'
}
```

##### Feature Lists

```python
FEATURES = {
    'numeric': ['ram_gb', 'storage_gb', 'screen_inches', 'processor_gen'],
    'categorical': ['brand', 'processor_type', 'os', 'gpu']
}

BRANDS = ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', ...]

PROCESSORS = ['Intel Core i3 10th Gen', 'Intel Core i5 11th Gen', ...]

OPERATING_SYSTEMS = ['Windows 11', 'Windows 10', 'macOS', 'Ubuntu', 'Linux']

GPUS = ['Intel Integrated', 'NVIDIA GTX 1650', ...]
```

---

## Complete Usage Example

### End-to-End Workflow

```python
# 1. Import modules
from src.data_preprocessing import DataPreprocessor
from src.train_model import LaptopPriceModel
from src.utils import load_data, save_data
import pandas as pd

# 2. Load or create data
df = load_data('data/laptop_data.csv')

# 3. Initialize model
model = LaptopPriceModel()

# 4. Prepare data
X, y, feature_names = model.prepare_data(df)

# 5. Train models
results, splits = model.train_models(X, y, test_size=0.2)

# 6. Save best model
model.save_model('models/best_model.pkl')

# 7. Make predictions on new data
new_laptop = pd.DataFrame([{
    'laptop_name': 'Dell XPS 15',
    'brand': 'Dell',
    'processor': 'Intel Core i7 11th Gen',
    'ram': '16GB',
    'storage': '512GB',
    'screen_size': '15.6 inches',
    'os': 'Windows 11',
    'gpu': 'NVIDIA GTX 1650'
}])

X_new, _, _ = model.prepare_data(new_laptop, target_col=None)
prediction = model.predict(X_new)
print(f"Predicted price: ${prediction[0]:,.2f}")
```

---

## Error Handling

### Common Exceptions

```python
# ValueError: No model trained
try:
    predictions = model.predict(X)
except ValueError as e:
    print("Error: Train or load a model first")

# FileNotFoundError: Model file not found
try:
    model.load_model('nonexistent.pkl')
except FileNotFoundError:
    print("Error: Model file not found")

# KeyError: Missing required columns
try:
    X, y, features = model.prepare_data(incomplete_df)
except KeyError as e:
    print(f"Error: Missing column {e}")
```

---

## Type Hints

The codebase uses Python type hints for better code clarity:

```python
def predict(self, X: pd.DataFrame) -> np.ndarray:
    """Type-hinted function signature"""
    pass

def extract_ram(self, ram_str: str) -> int:
    """Clear input/output types"""
    pass
```

---

## Future API Extensions

### REST API Endpoints (Planned)

```python
# Potential FastAPI implementation

@app.post("/predict")
async def predict_price(laptop: LaptopSpecs):
    """Endpoint for price prediction"""
    pass

@app.get("/models")
async def list_models():
    """List available models"""
    pass

@app.post("/train")
async def train_model(config: TrainingConfig):
    """Trigger model training"""
    pass
```

---

## Testing

### Unit Test Examples

```python
import unittest
from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = DataPreprocessor()
    
    def test_extract_brand(self):
        self.assertEqual(self.preprocessor.extract_brand("Dell XPS"), "Dell")
        
    def test_extract_ram(self):
        self.assertEqual(self.preprocessor.extract_ram("16GB"), 16)
        
    def test_extract_storage(self):
        self.assertEqual(self.preprocessor.extract_storage("1TB"), 1024)
```

---

## Performance Considerations

### Optimization Tips

1. **Batch Predictions**: Process multiple laptops at once
2. **Model Caching**: Load model once, reuse for predictions
3. **Feature Caching**: Save preprocessed features
4. **Parallel Processing**: Use joblib for parallel model training

### Memory Management

```python
# Efficient data loading
df = pd.read_csv('data.csv', usecols=['necessary', 'columns'])

# Clear large objects
del large_dataframe
import gc; gc.collect()
```

---

## Version Compatibility

- **Python**: 3.8+
- **pandas**: 2.0.3
- **numpy**: 1.24.3
- **scikit-learn**: 1.3.0
- **streamlit**: 1.28.0

---

For more information, see the [main documentation](README.md).
