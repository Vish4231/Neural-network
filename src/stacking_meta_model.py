import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load stacking outputs
stacking_df = pd.read_csv('model/stacking_outputs.csv')

# Features: base model outputs
X = stacking_df[['xgb', 'lgbm', 'cat', 'nn']]
y = stacking_df['target']

# Train/test split (use same random state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train meta-model
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(X_train, y_train)

# Predict
y_pred = meta_model.predict(X_test)

# Evaluate
print("\nStacking Meta-Model Test accuracy:", accuracy_score(y_test, y_pred))
print("Stacking Meta-Model Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Stacking Meta-Model Classification report:\n", classification_report(y_test, y_pred, target_names=['Not Top 5', 'Top 5'])) 