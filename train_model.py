# train_model_balanced.py
# Disaster Type Prediction - Balanced Dataset Model Training

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the balanced dataset
df = pd.read_csv("Disaster_dataset.csv")

# Encode the categorical column 'region_type'
label_enc = LabelEncoder()
df["region_type"] = label_enc.fit_transform(df["region_type"])

# Split data into features and target
X = df.drop("Disaster_Type", axis=1)
y = df["Disaster_Type"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "disaster_model.pkl")

# Save evaluation report
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("disaster_classification_report.csv")

# Print result summary
print("✅ Model trained and saved as disaster_model.pkl")
print("✅ Evaluation report saved as disaster_classification_report.csv")
