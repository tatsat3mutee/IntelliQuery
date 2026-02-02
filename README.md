# 🤖 IntelliQuery - Intelligent Customer Analytics Platform

**Production-ready RAG, ML, and Analytics system** - Works with ANY dataset!

## ✨ Key Features

- 🔍 **RAG-based Q&A**: Document search with LLM-powered answers
- 🗣️ **Natural Language to SQL**: Convert questions to SQL queries
- 🎯 **ML Predictions**: Dataset-agnostic classification (churn, attrition, etc.)
- 📊 **Dynamic Data Handling**: Works with ANY CSV/Excel dataset
- 📈 **Interactive Charts**: Automatic visualization generation
- 🔄 **Auto-Feature Detection**: No hardcoded column names
- 🚀 **Production-Ready**: Error handling, logging, model persistence

## 🎯 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create `.env` file:
```env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi_your_token_here
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your_warehouse_id
DATABRICKS_CATALOG=main
DATABRICKS_SCHEMA=intelliquery_data
DATABRICKS_LLM_ENDPOINT=your_llm_endpoint
DATABRICKS_EMBEDDING_ENDPOINT=your_embedding_endpoint
```

### 3. Start Server
```bash
python run.py
```

### 4. Access UI
Open http://localhost:8000

## 📁 Project Structure

```
IntelliQuery/
├── src/intelliquery/
│   ├── core/           # Configuration, database, exceptions
│   ├── rag/            # Document processing, vector search
│   ├── analytics/      # Data handling, text-to-SQL, routing
│   ├── ml/             # ML predictions (Random Forest)
│   ├── visualization/  # Chart generation
│   ├── api/            # FastAPI application
│   └── utils/          # Utility functions
├── models/             # Trained ML models (auto-saved)
├── templates/          # Web UI templates
├── notebooks/          # Jupyter notebooks
└── docs/               # Documentation
```

## 📚 Documentation

### Core Documentation
- **[ML_ARCHITECTURE.md](ML_ARCHITECTURE.md)** - Complete ML system guide (717 lines)
  - System overview & architecture
  - Algorithm explanation (Random Forest)
  - Component deep dive
  - Code structure & best practices
  
- **[MODEL_FIX_SUMMARY.md](MODEL_FIX_SUMMARY.md)** - Recent fixes & improvements
  - Overfitting prevention
  - Dynamic API endpoint
  - Testing instructions

### Additional Guides
- **[ARCHITECTURE_FLOWS.md](ARCHITECTURE_FLOWS.md)** - System architecture flows
- **[DATASET_ADAPTABILITY_GUIDE.md](DATASET_ADAPTABILITY_GUIDE.md)** - Working with different datasets
- **[SAMPLE_QUERIES.md](SAMPLE_QUERIES.md)** - Example queries

## 🤖 ML Model Details

### Algorithm: Random Forest Classifier
- **Trees**: 100
- **Max Depth**: 8 (balanced)
- **Features**: Auto-detected from data
- **Accuracy**: 75-85% (realistic)
- **Training Time**: < 30 seconds
- **Prediction Time**: < 100ms

### Key Features
✅ **Dataset Agnostic**: Works with ANY classification dataset
✅ **Auto-Feature Detection**: Finds target & features automatically
✅ **Balanced Regularization**: Prevents overfitting & underfitting
✅ **Dynamic API**: No hardcoded field names
✅ **Model Persistence**: Auto-saves after training
✅ **Feature Importance**: Built-in ranking

## 🚀 Usage Examples

### 1. Train Model
```bash
# Via UI: Click "Train Model" button
# Via API:
curl -X POST http://localhost:8000/train-model \
  -d "algorithm=random_forest"
```

### 2. Predict Churn
```bash
# Via UI: Fill form and click "Predict Churn"
# Via API:
curl -X POST http://localhost:8000/predict-churn \
  -d "Gender=Male" \
  -d "Contract=Month-to-month" \
  -d "Tenure Months=3" \
  -d "Monthly Charges=85.0"
```

### 3. Batch Predictions
```bash
curl http://localhost:8000/predict-batch?limit=100
```

## 🔧 Development

### Retrain Model (After Code Changes)
```bash
python retrain_model.py
```

### Run Tests
```bash
pytest tests/
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

## 📊 Model Performance

### Expected Metrics
- **Accuracy**: 75-85%
- **Precision**: 70-80%
- **Recall**: 60-75%
- **F1 Score**: 65-78%
- **AUC-ROC**: 0.80-0.90

### Risk Distribution
- **High Risk**: ~20% of customers
- **Medium Risk**: ~30% of customers
- **Low Risk**: ~50% of customers

## 🎓 How It Works

### Training Flow
```
Upload Data → Auto-Detect Features → Encode Categories →
Split Train/Test → Train Random Forest → Evaluate → Save Model
```

### Prediction Flow
```
Input Data → Load Model → Encode Features →
Predict Probability → Calculate Risk → Return Result
```

## 🐛 Troubleshooting

### Issue: Model gives same prediction for everything
**Solution**: Delete old model and retrain
```bash
python retrain_model.py
```

### Issue: 100% accuracy (overfitting)
**Solution**: Check for data leakage, retrain with new parameters

### Issue: Feature not found errors
**Solution**: Model auto-handles missing features (defaults to 0)

## 📝 Version History

**Version**: 2.1.0
**Status**: Production-Ready
**Last Updated**: 2026-02-02

### Recent Changes (v2.1.0)
- ✅ Fixed overfitting (balanced regularization)
- ✅ Made API dynamic (no hardcoded fields)
- ✅ Added comprehensive ML documentation
- ✅ Cleaned up unnecessary files
- ✅ Improved prediction variance

## 📄 License

Proprietary - All rights reserved

## 👨‍💻 Built With

- **Python 3.8+**
- **FastAPI** - Web framework
- **scikit-learn** - ML library
- **pandas/numpy** - Data processing
- **Databricks** - Data warehouse
- **joblib** - Model persistence

---