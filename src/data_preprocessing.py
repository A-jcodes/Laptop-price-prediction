import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

class DataPreprocessor:
    """
    A class to handle data preprocessing for laptop price prediction.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def extract_brand(self, laptop_name):
        """Extract brand from laptop name"""
        if pd.isna(laptop_name):
            return 'Unknown'
        brands = ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', 
                  'Microsoft', 'Razer', 'Samsung', 'Huawei', 'Xiaomi']
        for brand in brands:
            if brand.lower() in laptop_name.lower():
                return brand
        return 'Other'
    
    def extract_ram(self, ram_str):
        """Extract RAM size in GB"""
        if pd.isna(ram_str):
            return 0
        match = re.search(r'(\d+)', str(ram_str))
        if match:
            return int(match.group(1))
        return 0
    
    def extract_storage(self, storage_str):
        """Extract storage size in GB"""
        if pd.isna(storage_str):
            return 0
        storage_str = str(storage_str).upper()
        match = re.search(r'(\d+)', storage_str)
        if match:
            size = int(match.group(1))
            if 'TB' in storage_str:
                size = size * 1024
            return size
        return 0
    
    def extract_screen_size(self, screen_str):
        """Extract screen size in inches"""
        if pd.isna(screen_str):
            return 0
        match = re.search(r'(\d+\.?\d*)', str(screen_str))
        if match:
            return float(match.group(1))
        return 0
    
    def extract_processor_info(self, processor_str):
        """Extract processor generation and type"""
        if pd.isna(processor_str):
            return 'Unknown', 0
        
        processor_str = str(processor_str)
        
        # Extract generation
        gen_match = re.search(r'(\d+)th', processor_str, re.IGNORECASE)
        generation = int(gen_match.group(1)) if gen_match else 0
        
        # Extract type (i3, i5, i7, i9, Ryzen, etc.)
        if 'i9' in processor_str.lower():
            proc_type = 'i9'
        elif 'i7' in processor_str.lower():
            proc_type = 'i7'
        elif 'i5' in processor_str.lower():
            proc_type = 'i5'
        elif 'i3' in processor_str.lower():
            proc_type = 'i3'
        elif 'ryzen 9' in processor_str.lower():
            proc_type = 'Ryzen 9'
        elif 'ryzen 7' in processor_str.lower():
            proc_type = 'Ryzen 7'
        elif 'ryzen 5' in processor_str.lower():
            proc_type = 'Ryzen 5'
        elif 'ryzen 3' in processor_str.lower():
            proc_type = 'Ryzen 3'
        else:
            proc_type = 'Other'
            
        return proc_type, generation
    
    def preprocess_data(self, df):
        """
        Main preprocessing function
        """
        df = df.copy()
        
        # Extract features if columns exist
        if 'laptop_name' in df.columns:
            df['brand'] = df['laptop_name'].apply(self.extract_brand)
        
        if 'ram' in df.columns:
            df['ram_gb'] = df['ram'].apply(self.extract_ram)
        
        if 'storage' in df.columns:
            df['storage_gb'] = df['storage'].apply(self.extract_storage)
        
        if 'screen_size' in df.columns:
            df['screen_inches'] = df['screen_size'].apply(self.extract_screen_size)
        
        if 'processor' in df.columns:
            processor_info = df['processor'].apply(self.extract_processor_info)
            df['processor_type'] = processor_info.apply(lambda x: x[0])
            df['processor_gen'] = processor_info.apply(lambda x: x[1])
        
        # Encode categorical variables
        categorical_cols = ['brand', 'processor_type', 'os', 'gpu']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen labels
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] 
                        if str(x) in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        return df
    
    def get_feature_columns(self):
        """Return list of feature columns for model training"""
        return ['ram_gb', 'storage_gb', 'screen_inches', 'processor_gen',
                'brand_encoded', 'processor_type_encoded', 'os_encoded', 'gpu_encoded']
