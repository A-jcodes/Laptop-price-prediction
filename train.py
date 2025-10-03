"""
Script to train the laptop price prediction model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data, save_data
from src.train_model import LaptopPriceModel

def main():
    print("=" * 50)
    print("Laptop Price Prediction - Model Training")
    print("=" * 50)
    
    # Load or create data
    data_path = 'data/laptop_data.csv'
    if os.path.exists(data_path):
        df = load_data(data_path)
    else:
        print("\nNo existing dataset found. Creating sample dataset...")
        df = load_data()
        os.makedirs('data', exist_ok=True)
        save_data(df, data_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Initialize model
    model = LaptopPriceModel()
    
    # Prepare data
    print("\nPreparing data...")
    X, y, feature_names = model.prepare_data(df)
    print(f"Features used: {feature_names}")
    
    # Train models
    print("\nTraining models...")
    results, (X_train, X_test, y_train, y_test) = model.train_models(X, y)
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/laptop_price_model.pkl')
    
    # Feature importance
    feature_imp = model.get_feature_importance(feature_names)
    if feature_imp is not None:
        print("\nFeature Importance:")
        print(feature_imp.to_string(index=False))
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
