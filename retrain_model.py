"""
Script to retrain the churn model with new regularization parameters
This will delete the old overfitted model and train a new one
"""

import os
from pathlib import Path

# Delete old model
model_path = Path("models/churn_model.pkl")
if model_path.exists():
    os.remove(model_path)
    print("[OK] Deleted old model file")
else:
    print("[INFO] No existing model found")

print("\n" + "="*60)
print("MODEL RETRAIN INSTRUCTIONS")
print("="*60)
print("\n1. The old overfitted model has been deleted")
print("2. Start your Flask app: python run.py")
print("3. Go to http://127.0.0.1:8000")
print("4. Click 'Train Model' button")
print("\n[EXPECTED RESULTS]")
print("   - Accuracy: 75-85% (NOT 100%)")
print("   - Varied predictions (not all the same)")
print("   - No feature name warnings")
print("   - Batch predictions work correctly")
print("\n[WARNING] If you still see 100% accuracy:")
print("   - Check for data leakage (duplicate columns)")
print("   - Verify dataset has enough variety")
print("   - Check logs for warnings")
print("="*60)

# Made with Bob
