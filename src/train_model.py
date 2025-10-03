import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import xgboost as xgb
from src.data_preprocessing import DataPreprocessor

class LaptopPriceModel:
    """
    A class to train and evaluate laptop price prediction models.
    """
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0)
        }
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = DataPreprocessor()
        
    def prepare_data(self, df, target_col='price'):
        """
        Prepare data for training
        """
        # Preprocess data
        df_processed = self.preprocessor.preprocess_data(df)
        
        # Get feature columns
        feature_cols = self.preprocessor.get_feature_columns()
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df_processed.columns]
        
        X = df_processed[available_features]
        y = df_processed[target_col] if target_col in df_processed.columns else None
        
        return X, y, available_features
    
    def train_models(self, X, y, test_size=0.2):
        """
        Train multiple models and select the best one
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_rmse': cv_rmse
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV RMSE: {cv_rmse:.2f}")
        
        # Select best model based on R² score
        self.best_model_name = max(results, key=lambda x: results[x]['r2'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        
        return results, (X_train, X_test, y_train, y_test)
    
    def save_model(self, filepath='models/laptop_price_model.pkl'):
        """
        Save the best model and preprocessor
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'preprocessor': self.preprocessor
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/laptop_price_model.pkl'):
        """
        Load a saved model and preprocessor
        """
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.preprocessor = model_data['preprocessor']
        print(f"Model loaded: {self.best_model_name}")
    
    def predict(self, X):
        """
        Make predictions using the best model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained or loaded!")
        
        return self.best_model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance if available
        """
        if self.best_model is None:
            raise ValueError("No model has been trained or loaded!")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_imp
        else:
            return None
