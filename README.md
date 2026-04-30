# credit-card-fraud

## Credit Card Approval Prediction using Machine Learning

This project builds a machine learning model to predict whether a credit card application will be approved or not based on applicant details. It uses a Random Forest Classifier along with data preprocessing, evaluation metrics, and automated testing.


## Project Overview

The goal of this project is to :-

- Predict credit card approval decisions
- Apply data preprocessing techniques
- Train and evaluate a machine learning model
- Validate performance using unit and integration tests

## Project Structure

```
├── credit card fraud.py              # Main model training and prediction script
├── credit_approval_with_tests.py    # Model with testing and performance checks
├── creditcard.csv                   # Dataset file
└── README.md                        # Project documentation
```

## Features

- Data cleaning (handling missing values)
- Feature normalization (Debt, Income, Age)
- Categorical encoding using one-hot encoding
- Model training using Random Forest
- Performance evaluation :-
    - Accuracy
    - Precision
    - Recall
- Prediction function for new applications
- Unit and integration testing using unittest
- Training time measurement

## Machine Learning Model

- Algorithm Used : Random Forest Classifier
- Library : scikit-learn
- Train-Test Split : 70% training, 30% testing (stratified)

## Evaluation Metrics

The model is evaluated using :-

- Accuracy – Overall correctness
- Precision – Correct positive predictions
- Recall – Ability to detect approved applications

Minimum performance thresholds (in tests) :-

- Accuracy ≥ 0.7
- Precision ≥ 0.6
- Recall ≥ 0.6


## Dataset Description

The dataset (creditcard.csv) contains applicant details such as:

- Gender
- Age
- Debt
- Marital Status
- Bank Customer Status
- Employment Status
- Credit Score
- Income
- Industry
- Ethnicity
- Approval Status (Target Variable)


## How to Run the Project

1. Install Dependencies

```
pip install pandas numpy scikit-learn
```

2. Run Basic Model

```
python "credit card fraud.py"
```

3. Run Model with Tests

```
python credit_approval_with_tests.py
```

This will :-

- Train the model
- Print evaluation metrics
- Run unit tests


## Example Prediction

```
new_application = np.zeros(X.shape[1])
result = predict_approval(new_application)
print(result)  # Output: 0 (Not Approved) or 1 (Approved)
```


## Testing

The project includes automated tests to ensure :-

- Valid predictions return correct output (0 or 1)
- Errors are raised for invalid input shapes
- No missing values remain after preprocessing
- Model meets minimum performance thresholds
- Training completes within acceptable time
