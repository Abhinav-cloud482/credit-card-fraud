import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init

init(autoreset=True)
warnings.filterwarnings("ignore")

# === Load Dataset ===
df = pd.read_csv("creditcard.csv")

# === Handle Missing Values ===
df.dropna(inplace=True)

# === Separate Features ===
target = 'Approved'
features = df.columns.drop(target)

numerical_features = df[features].select_dtypes(include=['int64', 'float64']).columns
categorical_features = df[features].select_dtypes(include=['object', 'bool']).columns

# === Preprocessing ===
# Scale numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# === Train-Test Split ===
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# === Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluation ===
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)

print(Fore.CYAN + Style.BRIGHT + "\nModel Performance Summary:")
print(Fore.GREEN + f"✔️ Accuracy : {accuracy:.2f}")
print(Fore.GREEN + f"✔️ Precision: {precision:.2f}")
print(Fore.GREEN + f"✔️ Recall   : {recall:.2f}")
print("\n" + Fore.YELLOW + classification_report(y_test, y_pred))

# === Feature Importance Plot ===
def plot_feature_importance(model, X):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:]  # Top 10
    features = X.columns[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[indices], y=features)
    plt.title('Top 10 Important Features')
    plt.tight_layout()
    plt.show()

plot_feature_importance(model, X)

# === Predict New Applicant ===
def predict_approval(application_dict):
    """Predict approval from a dict of applicant data"""
    try:
        application_df = pd.DataFrame([application_dict])
        # Ensure all expected columns exist
        for col in X.columns:
            if col not in application_df.columns:
                application_df[col] = 0  # default
        
        # Reorder to match training set
        application_df = application_df[X.columns]
        prediction = model.predict(application_df)[0]
        print(Fore.MAGENTA + f"🔍 New Application Prediction: {'APPROVED' if prediction == 1 else 'DENIED'}")
        return prediction
    except Exception as e:
        print(Fore.RED + f"❌ Error in prediction: {e}")
        return None

# === Sample Dynamic Input ===
sample_application = {
    'Age': (30 - df['Age'].mean()) / df['Age'].std(),
    'Debt': 0.5,
    'Income': 0.3,
    'Married_yes': 1,
    'BankCustomer_yes': 1,
    'PriorDefault_yes': 0,
    'Employed_yes': 1,
    'DriversLicense_yes': 1,
    'Citizen_USA': 1,
    'Industry_Services': 1,
    'Ethnicity_Caucasian': 1
}

predict_approval(sample_application)
