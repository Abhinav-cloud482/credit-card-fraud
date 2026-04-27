import pandas as pd
import numpy as np
import time
import unittest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# === Load and preprocess data ===

df = pd.read_csv("creditcard.csv")
df.dropna(inplace=True)

# Normalize numerical columns
for col in ["Debt", "Income", "Age"]:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# Convert categorical columns
df = pd.get_dummies(df, columns=["Married", "BankCustomer", "PriorDefault",
                                 "Employed", "DriversLicense", "Citizen",
                                 "Industry", "Ethnicity"])

# Features and target
X = df.drop("Approved", axis=1)
y = df["Approved"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === Model Training ===

model = RandomForestClassifier(n_estimators=100, random_state=42)

train_start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - train_start

# === Model Evaluation ===

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)

print(f"Model Evaluation:\n Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
print(f"Training Time: {train_time:.2f} seconds")

# === Predict Function ===

def predict_approval(application):
    if len(application) != X.shape[1]:
        raise ValueError(f"Expected input with {X.shape[1]} features, got {len(application)}")
    application_df = pd.DataFrame([application], columns=X.columns)
    return model.predict(application_df)[0]

# Example prediction
new_application = np.zeros(X.shape[1])
print(f"Example Prediction - Approved? {predict_approval(new_application)}")

# === Unit and Integration Tests ===

class TestCreditApprovalModel(unittest.TestCase):

    def test_valid_prediction(self):
        """Test valid input prediction."""
        sample = np.zeros(X.shape[1])
        result = predict_approval(sample)
        self.assertIn(result, [0, 1])

    def test_invalid_prediction_shape(self):
        """Test error raised with invalid input shape."""
        invalid_sample = np.zeros(X.shape[1] - 1)
        with self.assertRaises(ValueError):
            predict_approval(invalid_sample)

    def test_no_missing_values(self):
        """Ensure dataset has no missing values after cleaning."""
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_metrics_threshold(self):
        """Ensure model meets minimum performance criteria."""
        self.assertGreaterEqual(accuracy, 0.7)
        self.assertGreaterEqual(precision, 0.6)
        self.assertGreaterEqual(recall, 0.6)

    def test_training_time(self):
        """Check model training completes in reasonable time."""
        self.assertLess(train_time, 10)

if __name__ == "__main__":
    print("\n=== Running Tests ===")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
