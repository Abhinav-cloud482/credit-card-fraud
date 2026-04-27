import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
df = pd.read_csv("creditcard.csv")

# Handle missing values if any
df.dropna(inplace=True)

# Normalize numerical columns for better model performance
df["Debt"] = (df["Debt"] - df["Debt"].mean()) / df["Debt"].std()
df["Income"] = (df["Income"] - df["Income"].mean()) / df["Income"].std()
df["Age"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()

# Convert categorical columns into numerical representation
df = pd.get_dummies(df, columns=["Married", "BankCustomer", "PriorDefault", "Employed", "DriversLicense", "Citizen", "Industry", "Ethnicity"])

# Define target variable
X = df.drop("Approved", axis=1)
y = df["Approved"]

# Ensure balanced dataset split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)

print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

# Function to predict approval of a new applicant
def predict_approval(application):
    application_df = pd.DataFrame([application], columns=X.columns)
    return model.predict(application_df)[0]

# Example new application ensuring correct feature length
new_application = np.zeros(X.shape[1])  # Placeholder ensuring proper size
new_application[0] = 30  # Modify accordingly

print(f"Approved? {predict_approval(new_application)}")
