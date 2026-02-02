"""
ML Predictor - Train and predict HVS rates
Uses scikit-learn RandomForest for simplicity
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from ..analytics.hvs_handler import get_hvs_raw_data

logger = logging.getLogger(__name__)


class HVSPredictor:
    """ML model for predicting HVS rates"""
    
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_columns = ['enterprise_id', 'subaccount', 'device_type', 'service_type']
        self.is_trained = False
        self.training_stats = {}
    
    def train(self) -> Dict:
        """Train model on HVS data from Databricks"""
        try:
            # Get data
            df = get_hvs_raw_data(limit=5000)
            
            if df.empty:
                return {
                    "success": False,
                    "message": "No HVS data found. Upload HVS data first."
                }
            
            if len(df) < 10:
                return {
                    "success": False,
                    "message": f"Not enough data to train (have {len(df)}, need at least 10 records)"
                }
            
            logger.info(f"Training model on {len(df)} records")
            
            # Prepare features
            df_train = df.copy()
            
            # Encode categorical features
            self.encoders = {}
            for col in self.feature_columns:
                if col in df_train.columns:
                    self.encoders[col] = LabelEncoder()
                    df_train[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        df_train[col].fillna('UNKNOWN').astype(str)
                    )
            
            # Build feature matrix
            feature_cols = [f'{c}_encoded' for c in self.feature_columns if c in df_train.columns]
            if 'bandwidth_mbps' in df_train.columns:
                feature_cols.append('bandwidth_mbps')
                df_train['bandwidth_mbps'] = df_train['bandwidth_mbps'].fillna(0)
            
            X = df_train[feature_cols].fillna(0)
            y = df_train['monthly_rate']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            self.is_trained = True
            self.training_stats = {
                "total_records": len(df),
                "train_records": len(X_train),
                "test_records": len(X_test),
                "train_r2": round(train_score, 4),
                "test_r2": round(test_score, 4),
                "features_used": feature_cols,
                "unique_enterprises": df['enterprise_id'].nunique() if 'enterprise_id' in df.columns else 0,
                "unique_subaccounts": df['subaccount'].nunique() if 'subaccount' in df.columns else 0
            }
            
            return {
                "success": True,
                "message": f"Model trained! R² score: Train={train_score:.3f}, Test={test_score:.3f}",
                "stats": self.training_stats
            }
        
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(
        self,
        enterprise_id: str,
        subaccount: str,
        device_type: str = "ROUTER",
        service_type: str = "INTERNET",
        bandwidth_mbps: int = 100
    ) -> Dict:
        """Predict rate for given parameters"""
        try:
            if not self.is_trained or self.model is None:
                # Try to train first
                train_result = self.train()
                if not train_result.get("success"):
                    return train_result
            
            # Encode inputs
            features = []
            for col, val in [
                ('enterprise_id', enterprise_id),
                ('subaccount', subaccount),
                ('device_type', device_type),
                ('service_type', service_type)
            ]:
                if col in self.encoders:
                    try:
                        encoded = self.encoders[col].transform([str(val)])[0]
                    except ValueError:
                        # Unknown value - use 0
                        encoded = 0
                    features.append(encoded)
            
            features.append(bandwidth_mbps)
            
            # Predict
            prediction = self.model.predict([features])[0]
            
            return {
                "success": True,
                "predicted_rate": round(float(prediction), 2),
                "inputs": {
                    "enterprise_id": enterprise_id,
                    "subaccount": subaccount,
                    "device_type": device_type,
                    "service_type": service_type,
                    "bandwidth_mbps": bandwidth_mbps
                }
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_predictions_for_all(self) -> pd.DataFrame:
        """Get predictions for all unique enterprise/subaccount combinations"""
        try:
            if not self.is_trained or self.model is None:
                train_result = self.train()
                if not train_result.get("success"):
                    return pd.DataFrame()
            
            # Get unique combinations
            df = get_hvs_raw_data(limit=1000)
            
            if df.empty:
                return pd.DataFrame()
            
            # Get unique combinations
            unique_combos = df.groupby(['enterprise_id', 'subaccount']).agg({
                'device_type': 'first',
                'service_type': 'first',
                'bandwidth_mbps': 'mean',
                'monthly_rate': 'mean'  # Actual rate for comparison
            }).reset_index()
            
            predictions = []
            for _, row in unique_combos.iterrows():
                result = self.predict(
                    enterprise_id=row['enterprise_id'],
                    subaccount=row['subaccount'],
                    device_type=row.get('device_type', 'ROUTER'),
                    service_type=row.get('service_type', 'INTERNET'),
                    bandwidth_mbps=int(row.get('bandwidth_mbps', 100))
                )
                
                if result.get("success"):
                    predictions.append({
                        'enterprise_id': row['enterprise_id'],
                        'subaccount': row['subaccount'],
                        'actual_rate': round(row['monthly_rate'], 2),
                        'predicted_rate': result['predicted_rate'],
                        'difference': round(result['predicted_rate'] - row['monthly_rate'], 2)
                    })
            
            return pd.DataFrame(predictions)
        
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return pd.DataFrame()
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            return {"success": False, "message": "Model not trained"}
        
        try:
            importances = self.model.feature_importances_
            feature_names = [f'{c}_encoded' for c in self.feature_columns] + ['bandwidth_mbps']
            
            importance_dict = dict(zip(feature_names, importances))
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "success": True,
                "importance": sorted_importance
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global predictor instance
predictor = HVSPredictor()
