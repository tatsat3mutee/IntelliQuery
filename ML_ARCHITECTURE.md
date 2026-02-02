# 🤖 Machine Learning Architecture - Complete Guide

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [ML Model Architecture](#ml-model-architecture)
3. [Component Deep Dive](#component-deep-dive)
4. [Data Flow](#data-flow)
5. [Algorithm Explanation](#algorithm-explanation)
6. [Code Structure](#code-structure)
7. [API Integration](#api-integration)
8. [Best Practices](#best-practices)

---

## 1. System Overview

### What This System Does

BillingRAG is an **Intelligent Customer Analytics Platform** that combines:

- **RAG (Retrieval Augmented Generation)**: Document Q&A using AI
- **ML Prediction**: Customer churn prediction using machine learning
- **Analytics**: Data visualization and insights

### ML Component Purpose

The ML component predicts **customer churn** (likelihood of customer leaving) using:

- Historical customer data
- Behavioral patterns
- Service usage information

---

## 2. ML Model Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BillingRAG Platform                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Data       │───▶│   ML Model   │───▶│  Predictions │  │
│  │   Upload     │    │   Training   │    │  & Insights  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Databricks (Data Storage)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | scikit-learn | Model training & prediction |
| **Algorithm** | Random Forest Classifier | Binary classification (churn/no-churn) |
| **Data Processing** | pandas, numpy | Data manipulation |
| **Storage** | Databricks | Data warehouse |
| **API** | FastAPI | REST endpoints |
| **Persistence** | joblib | Model serialization |

---

## 3. Component Deep Dive

### 3.1 MLPredictor Class (`src/intelliquery/ml/predictor.py`)

**Purpose**: Main ML engine for churn prediction

**Key Attributes**:

```python
class MLPredictor:
    model: RandomForestClassifier      # Trained ML model
    encoders: Dict[str, LabelEncoder]  # Categorical feature encoders
    feature_columns: List[str]         # Feature names used in training
    categorical_features: List[str]    # Categorical feature names
    numeric_features: List[str]        # Numeric feature names
    target_column: str                 # Target variable name
    is_trained: bool                   # Training status flag
    training_stats: Dict               # Model performance metrics
    feature_importance: Dict           # Feature importance scores
```

---

### 3.2 Auto-Feature Detection

**Method**: `_auto_detect_features(df: pd.DataFrame)`

**What It Does**:

1. **Finds Target Column** automatically
2. **Classifies Features** as categorical or numeric
3. **Prevents Data Leakage** by excluding suspicious columns

**Code Flow**:

```python
# Step 1: Find target column
target_patterns = ['churn', 'attrition', 'cancelled', 'left']
for col in df.columns:
    if any(pattern in col.lower() for pattern in target_patterns):
        self.target_column = col  # Found it!

# Step 2: Exclude ID/metadata columns
exclude_patterns = ['id', 'customer', 'date', 'timestamp']

# Step 3: Classify remaining columns
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        self.numeric_features.append(col)  # Numbers
    else:
        self.categorical_features.append(col)  # Text/categories
```

**Why This Matters**:

- ✅ Works with ANY dataset (not hardcoded)
- ✅ Prevents using customer IDs as features (data leakage)
- ✅ Automatically adapts to new data structures

---

### 3.3 Feature Preparation

**Method**: `_prepare_features(df: pd.DataFrame, is_training: bool)`

**What It Does**:
Converts raw data into ML-ready format

**Process**:

```
Raw Data                    Prepared Data
─────────                   ─────────────
Gender: "Male"       ───▶   Gender_encoded: 1
Contract: "Monthly"  ───▶   Contract_encoded: 0
Tenure: 12          ───▶   Tenure: 12 (unchanged)
```

**Encoding Explained**:

**Label Encoding** converts categories to numbers:

```python
# Training Phase
encoder = LabelEncoder()
encoder.fit(['Male', 'Female'])
# Male → 0, Female → 1

# Prediction Phase
encoder.transform(['Male'])  # Returns: 0
```

**Why Encoding?**:

- ML models only understand numbers
- Preserves categorical relationships
- Consistent across training/prediction

---

### 3.4 Model Training

**Method**: `train(algorithm: str = 'random_forest')`

**Training Pipeline**:

```
1. Load Data (from Databricks)
   ↓
2. Auto-Detect Features
   ↓
3. Encode Categorical Features
   ↓
4. Split Data (80% train, 20% test)
   ↓
5. Train Random Forest Model
   ↓
6. Evaluate Performance
   ↓
7. Save Model to Disk
```

**Random Forest Explained**:

```
Random Forest = Multiple Decision Trees

Tree 1:     Tree 2:     Tree 3:     ...     Tree 100:
  ┌─┐         ┌─┐         ┌─┐                 ┌─┐
  │?│         │?│         │?│                 │?│
  └┬┘         └┬┘         └┬┘                 └┬┘
 ┌─┴─┐       ┌─┴─┐       ┌─┴─┐               ┌─┴─┐
 │   │       │   │       │   │               │   │
Churn No    No Churn   Churn No            No Churn

Final Prediction = Majority Vote
If 60 trees say "Churn" and 40 say "No Churn" → Predict Churn (60%)
```

**Hyperparameters Explained**:

```python
RandomForestClassifier(
    n_estimators=100,        # Number of trees (more = better, slower)
    max_depth=8,             # Tree depth (deeper = more complex)
    min_samples_split=10,    # Min samples to split node
    min_samples_leaf=5,      # Min samples in leaf node
    max_features='sqrt',     # Features per split (sqrt(n) features)
    max_samples=0.8,         # Bootstrap 80% of data per tree
    random_state=42,         # Reproducibility seed
    n_jobs=-1                # Use all CPU cores
)
```

**Why These Values?**:

- `max_depth=8`: Prevents overfitting (trees not too deep)
- `min_samples_split=10`: Requires enough data to split
- `min_samples_leaf=5`: Prevents tiny leaves (overfitting)
- `max_features='sqrt'`: Adds randomness (better generalization)
- `max_samples=0.8`: Each tree sees different data (diversity)

---

### 3.5 Model Evaluation

**Metrics Used**:

| Metric | Formula | What It Means | Good Value |
|--------|---------|---------------|------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness | 75-85% |
| **Precision** | TP / (TP + FP) | Of predicted churns, how many actually churned | 70-80% |
| **Recall** | TP / (TP + FN) | Of actual churns, how many we caught | 60-75% |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Balance of precision & recall | 65-78% |
| **AUC-ROC** | Area under ROC curve | Model's ability to distinguish classes | 0.80-0.90 |

**Confusion Matrix**:

```
                Predicted
              No Churn  Churn
Actual  No    ┌────────┬────────┐
Churn   Churn │   TN   │   FP   │  False Positive (False Alarm)
              ├────────┼────────┤
        Churn │   FN   │   TP   │  True Positive (Correct!)
              └────────┴────────┘
                False Negative (Missed)
```

**Example**:

```
Accuracy: 80%    → 80 out of 100 predictions correct
Precision: 75%   → Of 100 predicted churns, 75 actually churned
Recall: 70%      → Of 100 actual churns, we caught 70
AUC: 0.85        → 85% chance model ranks churner higher than non-churner
```

---

### 3.6 Prediction

**Method**: `predict(customer_data: Dict)`

**Prediction Flow**:

```
Input: Customer Data (Dict)
   ↓
1. Load Model (if not loaded)
   ↓
2. Encode Categorical Features
   ↓
3. Build Feature Vector
   ↓
4. Model Prediction
   ↓
5. Calculate Probability
   ↓
6. Determine Risk Level
   ↓
Output: Prediction Result
```

**Example**:

```python
# Input
customer_data = {
    "Gender": "Male",
    "Contract": "Month-to-month",
    "Tenure Months": 3,
    "Monthly Charges": 85.0
}

# Processing
Gender_encoded: 1
Contract_encoded: 0
Tenure Months: 3
Monthly Charges: 85.0

# Model Output
Raw Probability: [0.25, 0.75]  # [No Churn, Churn]
Churn Probability: 75%

# Risk Level
if probability >= 70%: "HIGH RISK"
elif probability >= 40%: "MEDIUM RISK"
else: "LOW RISK"

# Final Output
{
    "success": True,
    "will_churn": True,
    "churn_probability": 0.75,
    "risk_level": "HIGH",
    "recommendation": "Immediate retention action needed..."
}
```

---

### 3.7 Feature Importance

**What It Shows**:
Which features most influence churn predictions

**How It Works**:
Random Forest calculates importance by measuring how much each feature reduces impurity (Gini index) across all trees.

**Example Output**:

```python
{
    "Contract_encoded": 0.25,      # 25% importance
    "Tenure Months": 0.20,         # 20% importance
    "Monthly Charges": 0.15,       # 15% importance
    "Internet Service_encoded": 0.12,
    "Tech Support_encoded": 0.10,
    ...
}
```

**Interpretation**:

- Contract type is the **most important** predictor (25%)
- Tenure is second most important (20%)
- Together, top 3 features explain 60% of predictions

---

## 4. Data Flow

### Complete Data Journey

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA UPLOAD                                               │
│    User uploads Excel/CSV → FastAPI → Databricks            │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MODEL TRAINING                                            │
│    ┌──────────────────────────────────────────────────┐    │
│    │ a. Load data from Databricks                      │    │
│    │ b. Auto-detect target & features                  │    │
│    │ c. Encode categorical features                    │    │
│    │ d. Split train/test (80/20)                       │    │
│    │ e. Train Random Forest (100 trees)                │    │
│    │ f. Evaluate metrics                               │    │
│    │ g. Save model to models/churn_model.pkl           │    │
│    └──────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. PREDICTION                                                │
│    ┌──────────────────────────────────────────────────┐    │
│    │ a. User inputs customer data via UI               │    │
│    │ b. API receives form data                         │    │
│    │ c. Load trained model                             │    │
│    │ d. Encode input features                          │    │
│    │ e. Model predicts probability                     │    │
│    │ f. Calculate risk level                           │    │
│    │ g. Return prediction + recommendation             │    │
│    └──────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. VISUALIZATION                                             │
│    Generate charts: Risk distribution, Feature importance    │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Algorithm Explanation

### Why Random Forest?

**Advantages**:

1. ✅ **Handles Mixed Data**: Categorical + Numeric features
2. ✅ **Robust**: Less prone to overfitting than single decision tree
3. ✅ **Feature Importance**: Built-in feature ranking
4. ✅ **No Scaling Needed**: Works with different feature scales
5. ✅ **Non-Linear**: Captures complex patterns
6. ✅ **Interpretable**: Can visualize decision paths

**Disadvantages**:

1. ❌ **Slower**: Training 100 trees takes time
2. ❌ **Memory**: Stores all trees in memory
3. ❌ **Black Box**: Hard to explain individual predictions

### How Random Forest Works

**Step-by-Step**:

```
1. Bootstrap Sampling (with replacement)
   Original Data: [1,2,3,4,5,6,7,8,9,10]
   Sample 1: [1,3,3,5,7,8,9,10]  ← Some duplicates, some missing
   Sample 2: [2,2,4,5,6,7,8,9]
   ...
   Sample 100: [1,2,4,4,6,7,9,10]

2. Build Decision Tree on Each Sample
   Tree 1 uses Sample 1
   Tree 2 uses Sample 2
   ...
   Tree 100 uses Sample 100

3. Random Feature Selection at Each Split
   At each node, only consider sqrt(n_features) random features
   Example: If 20 features, only consider 4-5 at each split

4. Aggregate Predictions (Voting)
   For classification: Majority vote
   For regression: Average
```

**Example Decision Tree**:

```
                    Tenure < 12 months?
                    /                \
                  Yes                 No
                  /                    \
        Contract = Monthly?      Monthly Charges > $70?
           /        \                /              \
         Yes        No              Yes             No
         /           \              /                \
    CHURN (90%)   NO CHURN    CHURN (60%)      NO CHURN
                   (20%)                         (10%)
```

---

## 6. Code Structure

### File Organization

```
src/intelliquery/ml/
├── __init__.py              # Package initialization
├── predictor.py             # Main ML predictor (532 lines)
│   ├── MLPredictor class
│   │   ├── __init__()
│   │   ├── _auto_detect_features()
│   │   ├── _prepare_features()
│   │   ├── _safe_transform()
│   │   ├── save_model()
│   │   ├── load_model()
│   │   ├── train()
│   │   ├── predict()
│   │   ├── predict_batch()
│   │   ├── get_feature_importance()
│   │   └── _get_recommendation()
│   └── churn_predictor (global instance)
└── ml_predictor.py          # HVS predictor (separate use case)
```

### Key Classes & Methods

#### MLPredictor Class

| Method | Lines | Purpose | Input | Output |
|--------|-------|---------|-------|--------|
| `__init__` | 31-44 | Initialize predictor | None | None |
| `_auto_detect_features` | 46-100 | Find target & features | DataFrame | target, features |
| `_prepare_features` | 102-124 | Encode features | DataFrame | Encoded DataFrame |
| `train` | 191-358 | Train model | algorithm | Training results |
| `predict` | 360-436 | Single prediction | customer_data | Prediction result |
| `predict_batch` | 447-515 | Batch predictions | limit | Batch results |
| `save_model` | 133-159 | Save to disk | path | Success bool |
| `load_model` | 161-189 | Load from disk | path | Success bool |

---

## 7. API Integration

### Endpoints

#### 1. Train Model

```http
POST /train-model
Content-Type: application/x-www-form-urlencoded

algorithm=random_forest

Response:
{
    "success": true,
    "message": "Model trained! Accuracy: 80.1%, AUC: 0.857",
    "stats": {
        "algorithm": "random_forest",
        "total_records": 7043,
        "train_records": 5634,
        "test_records": 1409,
        "accuracy": 0.801,
        "precision": 0.698,
        "recall": 0.444,
        "f1_score": 0.543,
        "auc_roc": 0.857
    }
}
```

#### 2. Predict Churn (Dynamic)

```http
POST /predict-churn
Content-Type: application/x-www-form-urlencoded

Gender=Male
Contract=Month-to-month
Tenure Months=3
Monthly Charges=85.0
Internet Service=Fiber optic

Response:
{
    "success": true,
    "will_churn": true,
    "churn_probability": 0.752,
    "risk_level": "HIGH",
    "recommendation": "HIGH RISK: Immediate retention action needed..."
}
```

#### 3. Batch Predictions

```http
GET /predict-batch?limit=100

Response:
{
    "success": true,
    "predictions": [
        {
            "customer_id": "7590-VHVEG",
            "churn_probability": 0.89,
            "predicted_churn": true,
            "actual_churn": 1,
            "risk_level": "HIGH"
        },
        ...
    ],
    "summary": {
        "total": 100,
        "high_risk": 23,
        "medium_risk": 31,
        "low_risk": 46
    }
}
```

---

## 8. Best Practices

### Model Training

✅ **DO**:

- Use balanced regularization (not too weak, not too strong)
- Check for data leakage (target in features)
- Validate on separate test set
- Monitor overfitting (train vs test accuracy)
- Save model after training

❌ **DON'T**:

- Use customer IDs as features
- Include target-derived columns
- Ignore class imbalance
- Skip feature importance analysis
- Hardcode feature names

### Prediction

✅ **DO**:

- Handle missing features gracefully
- Clip probabilities to [0, 1]
- Provide risk levels and recommendations
- Log predictions for monitoring
- Use DataFrame for feature names

❌ **DON'T**:

- Assume all features present
- Return raw model output
- Ignore unseen categories
- Skip input validation
- Use numpy arrays (loses feature names)

### Code Quality

✅ **DO**:

- Use type hints
- Add docstrings
- Log important events
- Handle exceptions
- Make code dataset-agnostic

❌ **DON'T**:

- Hardcode column names
- Ignore errors silently
- Skip input validation
- Use magic numbers
- Couple code to specific dataset

---

## Summary

### What We Built

A **production-ready, dataset-agnostic ML system** that:

1. ✅ Automatically detects features from any dataset
2. ✅ Trains Random Forest classifier with balanced regularization
3. ✅ Provides accurate churn predictions (75-85% accuracy)
4. ✅ Handles missing data and unseen categories
5. ✅ Offers flexible API for any dataset structure
6. ✅ Includes model persistence and feature importance

### Key Innovations

- **Auto-feature detection**: No hardcoded column names
- **Dynamic API**: Accepts any form data
- **Balanced regularization**: Not too simple, not too complex
- **Robust encoding**: Handles unseen categories safely
- **Production-ready**: Error handling, logging, validation

### Performance Targets

- Accuracy: 75-85%
- Precision: 70-80%
- Recall: 60-75%
- AUC: 0.80-0.90
- Training time: < 30 seconds
- Prediction time: < 100ms

---

**Made with ❤️ by Bob - Your AI Software Engineer**
