"""
Utility functions for loading and preparing the UCI Adult Income dataset.
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


def load_adult_income_data(test_size=0.2, random_state=42):
    """
    Load and prepare the UCI Adult Income dataset.
    
    Args:
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split and preprocessed data
    """
    adult = fetch_ucirepo(id=2)
    # print(adult.metadata)
    X = adult.data.features
    y = adult.data.targets

    # Clean target values: strip whitespace and periods, unify categories
    y = y['income'].str.strip().str.replace('.', '', regex=False)

    # Encode categorical features
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    # Encode target variable (income)
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test


# Helper function to get dataset info
def get_dataset_info():
    X_train, X_test, y_train, y_test = load_adult_income_data()
    return {
        'name': 'UCI Adult Income',
        'n_features': X_train.shape[1],
        'n_samples': X_train.shape[0] + X_test.shape[0]
    }