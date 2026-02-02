"""
Churn Predictor - ML model for customer churn prediction (PRODUCTION VERSION)
==============================================================================
DATASET-AGNOSTIC: Works with ANY classification dataset
- Auto-detects target column and features
- Model persistence (saves/loads automatically)
- Handles any categorical/numeric features
- Binary classification support
"""

import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..analytics.data_handler import get_churn_data

logger = logging.getLogger(__name__)


class MLPredictor:
    """ML model for predicting customer churn - DATASET AGNOSTIC"""
    
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_columns = []
        self.categorical_features = []
        self.numeric_features = []
        self.target_column = None
        self.is_trained = False
        self.training_stats = {}
        self.feature_importance = {}
        self.model_path = "models/churn_model.pkl"
        
        # Try to load existing model on initialization
        self.load_model()
    
    def _auto_detect_features(self, df: pd.DataFrame):
        """
        Automatically detect target column and feature types.
        Works with ANY classification dataset.
        """
        # Find target column (churn/attrition/cancelled/etc.)
        target_patterns = ['churn', 'attrition', 'cancelled', 'left', 
                          'churned', 'departed', 'exited', 'churn_value', 'churn_label']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in target_patterns):
                self.target_column = col
                logger.info(f"✓ Auto-detected target column: {col}")
                break
        
        if not self.target_column:
            raise ValueError(
                "Could not find target column. Expected column name containing: "
                "churn, attrition, cancelled, left, churned, departed, or exited"
            )
        
        # Exclude ID and metadata columns
        exclude_patterns = ['id', 'customer', 'employee', 'user', 'client',
                           'date', 'timestamp', 'upload', 'source', 'file']
        
        self.categorical_features = []
        self.numeric_features = []
        
        for col in df.columns:
            # Skip target and excluded columns
            if col == self.target_column:
                continue
            
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in exclude_patterns):
                continue
            
            # CRITICAL: Check for potential data leakage
            # Skip any column that looks like it might be derived from the target
            leakage_patterns = ['churn', 'attrition', 'cancelled', 'churned', 'left', 'departed', 'exited']
            if any(pattern in col_lower for pattern in leakage_patterns):
                logger.warning(f"⚠️ SKIPPING column '{col}' - potential data leakage (related to target variable)")
                continue
            
            # Classify by data type
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        logger.info(f"✓ Categorical features ({len(self.categorical_features)}): {self.categorical_features[:5]}...")
        logger.info(f"✓ Numeric features ({len(self.numeric_features)}): {self.numeric_features[:5]}...")
        
        return self.target_column, self.categorical_features, self.numeric_features
    
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        df_prep = df.copy()
        
        # Auto-detect features if training for the first time
        if is_training and not self.categorical_features:
            self._auto_detect_features(df)
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in df_prep.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_prep[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        df_prep[col].fillna('Unknown').astype(str)
                    )
                else:
                    # Handle unseen categories
                    df_prep[f'{col}_encoded'] = df_prep[col].fillna('Unknown').astype(str).apply(
                        lambda x: self._safe_transform(self.encoders[col], x)
                    )
        
        return df_prep
    
    def _safe_transform(self, encoder: LabelEncoder, value: str) -> int:
        """Safely transform a value, returning 0 for unknown"""
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return 0
    
    def save_model(self, path: Optional[str] = None) -> bool:
        """Save trained model to disk"""
        try:
            if not self.is_trained:
                logger.warning("No trained model to save")
                return False
            
            save_path = path or self.model_path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'encoders': self.encoders,
                'feature_columns': self.feature_columns,
                'categorical_features': self.categorical_features,
                'numeric_features': self.numeric_features,
                'target_column': self.target_column,
                'training_stats': self.training_stats,
                'feature_importance': self.feature_importance
            }
            
            joblib.dump(model_data, save_path)
            logger.info(f"✓ Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """Load trained model from disk"""
        try:
            load_path = path or self.model_path
            
            if not Path(load_path).exists():
                logger.info(f"No saved model found at {load_path}")
                return False
            
            model_data = joblib.load(load_path)
            
            self.model = model_data['model']
            self.encoders = model_data['encoders']
            self.feature_columns = model_data['feature_columns']
            self.categorical_features = model_data.get('categorical_features', [])
            self.numeric_features = model_data.get('numeric_features', [])
            self.target_column = model_data.get('target_column', 'churn_value')
            self.training_stats = model_data['training_stats']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = True
            
            logger.info(f"✓ Model loaded from {load_path}")
            logger.info(f"  Algorithm: {self.training_stats.get('algorithm')}")
            logger.info(f"  Accuracy: {self.training_stats.get('accuracy')}")
            logger.info(f"  Features: {self.training_stats.get('features_used')}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def train(self, algorithm: str = 'random_forest') -> Dict:
        """
        Train churn prediction model - WORKS WITH ANY DATASET
        
        Args:
            algorithm: 'random_forest' or 'gradient_boosting'
        
        Returns:
            Training results with metrics
        """
        try:
            # Get data
            df = get_churn_data(limit=10000)
            
            if df.empty:
                return {
                    "success": False,
                    "message": "No churn data found. Upload data first."
                }
            
            if len(df) < 50:
                return {
                    "success": False,
                    "message": f"Not enough data to train (have {len(df)}, need at least 50 records)"
                }
            
            logger.info(f"Training churn model on {len(df)} records")
            
            # Prepare features (auto-detect on first training)
            df_prep = self._prepare_features(df, is_training=True)
            
            # Build feature columns
            self.feature_columns = []
            
            # Encoded categorical features
            for col in self.categorical_features:
                if f'{col}_encoded' in df_prep.columns:
                    self.feature_columns.append(f'{col}_encoded')
            
            # Numeric features
            for col in self.numeric_features:
                if col in df_prep.columns:
                    df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce').fillna(0)
                    self.feature_columns.append(col)
            
            if not self.feature_columns:
                return {
                    "success": False,
                    "message": "No valid features found in data"
                }
            
            # Prepare X and y
            X = df_prep[self.feature_columns].fillna(0)
            
            # Convert target to binary if needed
            target_values = df_prep[self.target_column].unique()
            if len(target_values) == 2:
                # Binary classification - convert to 0/1
                if df_prep[self.target_column].dtype == 'object':
                    # String values like 'Yes'/'No', 'True'/'False'
                    positive_values = ['yes', 'true', '1', 'y', 'churned', 'left', 'departed']
                    y = df_prep[self.target_column].astype(str).str.lower().isin(positive_values).astype(int)
                else:
                    # Numeric values
                    y = df_prep[self.target_column].astype(int)
            else:
                return {
                    "success": False,
                    "message": f"Target column must be binary. Found {len(target_values)} unique values: {target_values}"
                }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            if algorithm == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=4,              # Limit depth to prevent overfitting
                    learning_rate=0.05,       # Lower learning rate
                    subsample=0.8,            # Use 80% of data per tree
                    random_state=42
                )
            else:
                # RandomForest with BALANCED regularization
                self.model = RandomForestClassifier(
                    n_estimators=100,           # Good number of trees
                    max_depth=8,                # Balanced depth (not too shallow, not too deep)
                    min_samples_split=10,       # Reasonable split requirement
                    min_samples_leaf=5,         # Reasonable leaf requirement
                    max_features='sqrt',        # Use subset of features
                    max_samples=0.8,            # Bootstrap 80% of data
                    random_state=42,
                    n_jobs=-1
                )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = 0.5
            
            # WARNING: Check for overfitting (perfect scores are suspicious)
            if accuracy >= 0.99 and precision >= 0.99 and recall >= 0.99:
                logger.warning("⚠️ MODEL ALERT: Perfect scores detected! This likely indicates:")
                logger.warning("  1. Data leakage (target column leaked into features)")
                logger.warning("  2. Duplicate rows in training data")
                logger.warning("  3. Dataset too small or too simple")
                logger.warning("  Please verify your data quality and feature selection!")
                logger.warning(f"  Churn rate: {y.mean():.1%}, Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Feature importance
            self.feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            
            # Sort by importance
            self.feature_importance = dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            self.is_trained = True
            self.training_stats = {
                "algorithm": algorithm,
                "total_records": len(df),
                "train_records": len(X_train),
                "test_records": len(X_test),
                "churn_rate": f"{y.mean() * 100:.1f}%",
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "auc_roc": round(auc, 4),
                "features_used": len(self.feature_columns),
                "target_column": self.target_column
            }
            
            # Save model automatically after training
            self.save_model()
            
            return {
                "success": True,
                "message": f"Model trained! Accuracy: {accuracy:.1%}, AUC: {auc:.3f}",
                "stats": self.training_stats,
                "feature_importance": dict(list(self.feature_importance.items())[:10])  # Top 10
            }
        
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, customer_data: Dict) -> Dict:
        """
        Predict churn probability for a single customer
        
        Args:
            customer_data: Dict with customer features
        
        Returns:
            Prediction result with probability
        """
        try:
            # Try to load model if not trained
            if not self.is_trained or self.model is None:
                if not self.load_model():
                    # If no saved model, train new one
                    train_result = self.train()
                    if not train_result.get("success"):
                        return train_result
            
            # Create single-row DataFrame
            df = pd.DataFrame([customer_data])
            
            # Prepare features
            df_prep = self._prepare_features(df)
            
            # Build feature vector
            features = []
            for col in self.feature_columns:
                if col in df_prep.columns:
                    features.append(df_prep[col].iloc[0])
                elif col.replace('_encoded', '') in customer_data:
                    # Try to encode
                    base_col = col.replace('_encoded', '')
                    if base_col in self.encoders:
                        val = self._safe_transform(self.encoders[base_col], str(customer_data[base_col]))
                        features.append(val)
                    else:
                        features.append(0)
                else:
                    features.append(0)
            
            # Predict - Use DataFrame to preserve feature names
            X = pd.DataFrame([features], columns=self.feature_columns)
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Get churn probability (class 1)
            churn_prob = float(probability[1] if len(probability) > 1 else probability[0])
            
            # CRITICAL: Ensure probability is between 0 and 1
            churn_prob = np.clip(churn_prob, 0.0, 1.0)
            
            # Log for debugging
            logger.info(f"Prediction: {prediction}, Raw probability: {probability}, Clipped: {churn_prob}")
            
            # Determine risk level
            if churn_prob >= 0.7:
                risk_level = "HIGH"
            elif churn_prob >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                "success": True,
                "will_churn": bool(prediction),
                "churn_probability": round(float(churn_prob), 3),
                "risk_level": risk_level,
                "recommendation": self._get_recommendation(churn_prob, customer_data)
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_recommendation(self, churn_prob: float, customer_data: Dict) -> str:
        """Generate recommendation based on prediction"""
        if churn_prob >= 0.7:
            return "HIGH RISK: Immediate retention action needed. Consider loyalty discount or personal outreach."
        elif churn_prob >= 0.4:
            return "MEDIUM RISK: Monitor customer satisfaction and engagement. Consider proactive outreach."
        else:
            return "LOW RISK: Customer appears satisfied. Continue current service."
    
    def predict_batch(self, limit: int = 100) -> Dict:
        """Predict churn for all customers in database"""
        try:
            if not self.is_trained or self.model is None:
                if not self.load_model():
                    train_result = self.train()
                    if not train_result.get("success"):
                        return train_result
            
            # Get data
            df = get_churn_data(limit=limit)
            
            if df.empty:
                return {"success": False, "message": "No data found"}
            
            # Prepare features
            df_prep = self._prepare_features(df)
            
            # Get features
            X = df_prep[self.feature_columns].fillna(0)
            
            # Predict
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            # CRITICAL: Clip probabilities to valid range
            probabilities = np.clip(probabilities, 0.0, 1.0)
            
            # Build results
            results = []
            for i, (_, row) in enumerate(df.iterrows()):
                prob = float(probabilities[i])
                # Try to find ID column
                id_col = next((c for c in df.columns if 'id' in c.lower()), None)
                customer_id = row.get(id_col, f'Customer_{i}') if id_col else f'Customer_{i}'
                
                # Handle string target values (Yes/No, True/False, etc.)
                actual_value = row.get(self.target_column, 0)
                if isinstance(actual_value, str):
                    positive_values = ['yes', 'true', '1', 'y', 'churned', 'left', 'departed']
                    actual_churn = 1 if actual_value.lower() in positive_values else 0
                else:
                    actual_churn = int(actual_value) if actual_value else 0
                
                results.append({
                    'customer_id': customer_id,
                    'churn_probability': round(prob, 3),
                    'predicted_churn': bool(predictions[i]),
                    'actual_churn': actual_churn,
                    'risk_level': 'HIGH' if prob >= 0.7 else ('MEDIUM' if prob >= 0.4 else 'LOW')
                })
            
            # Sort by churn probability (highest first)
            results.sort(key=lambda x: x['churn_probability'], reverse=True)
            
            return {
                "success": True,
                "predictions": results,
                "summary": {
                    "total": len(results),
                    "high_risk": sum(1 for r in results if r['risk_level'] == 'HIGH'),
                    "medium_risk": sum(1 for r in results if r['risk_level'] == 'MEDIUM'),
                    "low_risk": sum(1 for r in results if r['risk_level'] == 'LOW')
                }
            }
        
        except Exception as e:
            logger.error(f"Batch prediction error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return {"success": False, "message": "Model not trained yet"}
        
        return {
            "success": True,
            "feature_importance": self.feature_importance,
            "training_stats": self.training_stats
        }


# Global predictor instance
churn_predictor = MLPredictor()

# Made with Bob
