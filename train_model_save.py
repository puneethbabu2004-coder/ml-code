import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load feature dataset
df = pd.read_csv("thermal_features.csv")

# Input features and output label
X = df.drop(columns=["label", "image_name"])
y = df["label"]

# Split data into train and test
X_train, X_test, y_train, y_test, image_train, image_test = train_test_split(
    X,
    y,
    df["image_name"],
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "random_forest_thermal_model.pkl")

# Save test data for separate testing
test_data = X_test.copy()
test_data["image_name"] = image_test.values
test_data["actual_risk"] = y_test.values

test_data.to_csv("test_data.csv", index=False)

print("Model trained successfully.")
print("Saved model: random_forest_thermal_model.pkl")
print("Saved test data: test_data.csv")