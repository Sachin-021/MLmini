from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from scipy.io import arff
import os

# ✅ Step 1: Load ARFF file
file_path = r"C:\Users\Sachi\Downloads\dataset_"

if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
    exit()

# ✅ Step 2: Load data
try:
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    df['fraud'] = df['fraud'].astype(int)
    print("✅ Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# ✅ Step 3: Prepare features and labels
X = df.drop("fraud", axis=1)
y = df["fraud"]

# Convert object columns to numeric
for col in X.select_dtypes(include='object'):
    X[col] = pd.to_numeric(X[col], errors='coerce')

# ✅ Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Step 5: Define Random Forest model
rf_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}

# ✅ Step 6: Grid Search
print("\n🔍 Running Grid Search with Random Forest...")
grid_search = GridSearchCV(
    rf_model, param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# ✅ Step 7: Evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n✅ Best Parameters Found:", grid_search.best_params_)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"]))
print("📈 ROC AUC Score:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))
print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
