import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("creditcard.csv")

# Drop missing values
df.dropna(inplace=True)

# Normalize numerical columns
scaler = StandardScaler()
for col in ["Debt", "Income", "Age"]:
    df[col] = scaler.fit_transform(df[[col]])

# Encode categorical variables
df = pd.get_dummies(df, columns=["Married", "BankCustomer", "PriorDefault", "Employed", "DriversLicense", "Citizen", "Industry", "Ethnicity"])

# Split dataset
X = df.drop("Approved", axis=1)
y = df["Approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=1):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=1):.2f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=1))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance visualization
importances = model.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feat_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Predict for user input
def get_user_input():
    print("\n=== New Application Input ===")
    input_data = {}
    for feature in X.columns:
        while True:
            try:
                value = float(input(f"Enter value for {feature}: "))
                input_data[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    return pd.DataFrame([input_data])

def predict_approval(application_df):
    return model.predict(application_df)[0]

# Uncomment to allow terminal input
# user_app = get_user_input()
# prediction = predict_approval(user_app)
# print(f"\nPrediction: {'Approved' if prediction == 1 else 'Rejected'}")
