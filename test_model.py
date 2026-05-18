import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make sure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load saved model from models folder
model = joblib.load("models/random_forest_thermal_model.pkl")

# Load saved test data from data folder
test_data = pd.read_csv("data/test_data.csv")

# Separate features and actual labels
X_test = test_data.drop(columns=["image_name", "actual_risk"])
y_test = test_data["actual_risk"]

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
labels = ["LOW", "MEDIUM", "HIGH"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("\nConfusion Matrix:")
print(labels)
print(cm)

# FPR, FNR, Specificity
print("\nSpecificity, FPR and FNR per class:")

for i, cls in enumerate(labels):
    TP = cm[i, i]
    FN = np.sum(cm[i, :]) - TP
    FP = np.sum(cm[:, i]) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) != 0 else 0

    print(f"{cls}: Specificity = {specificity:.3f}, FPR = {FPR:.3f}, FNR = {FNR:.3f}")

# Save prediction result
results = test_data.copy()
results["predicted_risk"] = y_pred
results["result"] = np.where(results["actual_risk"] == results["predicted_risk"], "Correct", "Wrong")

# Save as CSV
results.to_csv("outputs/final_test_predictions.csv", index=False)

# Save as Excel
results.to_excel("outputs/final_test_predictions.xlsx", index=False)

print("\nFinal prediction results updated:")
print("outputs/final_test_predictions.csv")
print("outputs/final_test_predictions.xlsx")

print(results[["image_name", "actual_risk", "predicted_risk", "result"]])