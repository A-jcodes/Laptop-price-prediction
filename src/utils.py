import pandas as pd
import numpy as np

def create_sample_dataset(n_samples=100):
    """
    Create a sample laptop dataset for demonstration
    """
    np.random.seed(42)
    
    brands = ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI']
    processors = ['Intel Core i3 10th Gen', 'Intel Core i5 11th Gen', 'Intel Core i7 11th Gen',
                  'Intel Core i9 12th Gen', 'AMD Ryzen 5 5000', 'AMD Ryzen 7 5000',
                  'Apple M1', 'Apple M2']
    os_list = ['Windows 11', 'Windows 10', 'macOS', 'Ubuntu']
    gpu_list = ['Intel Integrated', 'NVIDIA GTX 1650', 'NVIDIA RTX 3050', 
                'NVIDIA RTX 3060', 'AMD Radeon', 'Apple GPU']
    
    data = []
    
    for i in range(n_samples):
        brand = np.random.choice(brands)
        processor = np.random.choice(processors)
        ram = np.random.choice([4, 8, 16, 32])
        storage = np.random.choice([256, 512, 1024, 2048])
        screen_size = np.random.choice([13.3, 14.0, 15.6, 17.3])
        os = np.random.choice(os_list)
        gpu = np.random.choice(gpu_list)
        
        # Generate price based on features
        base_price = 300
        
        # Brand factor
        brand_prices = {'Dell': 50, 'HP': 40, 'Lenovo': 45, 'Asus': 55, 
                       'Acer': 35, 'Apple': 500, 'MSI': 100}
        base_price += brand_prices.get(brand, 50)
        
        # Processor factor
        if 'i9' in processor or 'M2' in processor:
            base_price += 400
        elif 'i7' in processor or 'Ryzen 7' in processor or 'M1' in processor:
            base_price += 250
        elif 'i5' in processor or 'Ryzen 5' in processor:
            base_price += 150
        else:
            base_price += 50
        
        # RAM factor
        base_price += ram * 20
        
        # Storage factor
        base_price += storage * 0.15
        
        # Screen size factor
        base_price += screen_size * 10
        
        # GPU factor
        if 'RTX 3060' in gpu:
            base_price += 300
        elif 'RTX 3050' in gpu:
            base_price += 200
        elif 'GTX 1650' in gpu:
            base_price += 100
        
        # Add some random noise
        price = base_price * np.random.uniform(0.9, 1.1)
        
        laptop_name = f"{brand} Laptop {i+1}"
        
        data.append({
            'laptop_name': laptop_name,
            'brand': brand,
            'processor': processor,
            'ram': f"{ram}GB",
            'storage': f"{storage}GB" if storage < 1024 else f"{storage//1024}TB",
            'screen_size': f"{screen_size} inches",
            'os': os,
            'gpu': gpu,
            'price': round(price, 2)
        })
    
    df = pd.DataFrame(data)
    return df

def load_data(filepath=None):
    """
    Load laptop data from file or create sample dataset
    """
    if filepath and pd.io.common.file_exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Data loaded from {filepath}")
    else:
        df = create_sample_dataset()
        print("Sample dataset created")
    
    return df

def save_data(df, filepath='data/laptop_data.csv'):
    """
    Save dataframe to CSV
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
