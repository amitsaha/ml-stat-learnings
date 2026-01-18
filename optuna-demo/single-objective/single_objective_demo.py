"""
Single-objective optimization demo using Optuna with XGBoost on UCI Adult Income dataset.
This demo optimizes for accuracy using cross-validation.
"""
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
from utils.data_utils import load_adult_income_data


def objective(trial, X_train, y_train):
    """
    Objective function for Optuna to optimize XGBoost hyperparameters.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        
    Returns:
        float: Mean cross-validation accuracy score
    """
    # Define hyperparameter search space
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': 1,  # To avoid nested parallelism
    }
    
    # Create XGBoost classifier
    model = XGBClassifier(**param)
    
    # Perform 3 fold cross-validation and return mean accuracy
    # n_jobs = -1 to use all available cores
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1, error_score='raise')
    
    # optuna tries to maximize the objective value
    return float(scores.mean())


# Two key concepts for optuna:
# 1. Study - an optimization session
# 2. Trial - a single execution of the objective function

# here we run a single-objective optimization study
def run_single_objective_optimization(n_trials=50):
    """
    Run single-objective optimization study.
    
    Args:
        n_trials: Number of optimization trials
        
    Returns:
        study: Completed Optuna study object
    """
    # Load data
    print("Loading UCI Adult Income dataset...")
    X_train, X_test, y_train, y_test = load_adult_income_data()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create study
    study = optuna.create_study(direction='maximize', study_name='xgboost_single_objective')
    
    # Optimize
    print(f"\nStarting optimization with {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    
    # Print results
    print("\n" + "="*50)
    print("Optimization completed!")
    print("="*50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_params = study.best_params.copy()
    best_params.update({
        'random_state': 42,
        'eval_metric': 'logloss',
    })
    
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    test_accuracy = final_model.score(X_test, y_test)
    print(f"\nTest set accuracy: {test_accuracy:.4f}")
    
    return study


if __name__ == '__main__':
    study = run_single_objective_optimization(n_trials=5)
