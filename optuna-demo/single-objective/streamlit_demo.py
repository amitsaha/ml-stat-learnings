"""
Streamlit app for single-objective optimization using Optuna with XGBoost.
"""
import streamlit as st
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import pandas as pd
import plotly.graph_objects as go

from single_objective_demo import objective

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
from utils.data_utils import load_adult_income_data, get_dataset_info

# Set page config
st.set_page_config(
    page_title="Single-Objective Optuna Demo",
    page_icon="🎯",
    layout="wide"
)


# Title and description
st.title("🎯 Single-Objective Optimization with Optuna")
st.markdown("""
This demo shows how to use **Optuna** for single-objective hyperparameter optimization 
with **XGBoost** on the **UCI Adult Income** dataset.

The goal is to maximize the classification accuracy by finding the best hyperparameters.
""")

# Sidebar for parameters
st.sidebar.header("⚙️ Configuration")


# Dataset info
with st.sidebar.expander("📊 Dataset Information", expanded=False):
    try:
        dataset_info = get_dataset_info()
        st.write(f"**Name:** {dataset_info['name']}")
        st.write(f"**Features:** {dataset_info['n_features']}")
        st.write(f"**Samples:** {dataset_info['n_samples']}")
    except Exception as e:
        st.write("Unable to fetch dataset info")

# Optimization parameters
st.sidebar.subheader("Optimization Settings")
n_trials = st.sidebar.slider("Number of trials", min_value=10, max_value=100, value=30, step=10)

# Hyperparameter ranges
st.sidebar.subheader("Hyperparameter Ranges")

col1, col2 = st.sidebar.columns(2)
with col1:
    max_depth_min = st.number_input("max_depth min", value=3, min_value=1, max_value=10)
with col2:
    max_depth_max = st.number_input("max_depth max", value=10, min_value=3, max_value=20)

col1, col2 = st.sidebar.columns(2)
with col1:
    lr_min = st.number_input("learning_rate min", value=0.01, min_value=0.001, max_value=0.5, format="%.3f")
with col2:
    lr_max = st.number_input("learning_rate max", value=0.3, min_value=0.01, max_value=1.0, format="%.2f")

col1, col2 = st.sidebar.columns(2)
with col1:
    n_est_min = st.number_input("n_estimators min", value=50, min_value=10, max_value=200)
with col2:
    n_est_max = st.number_input("n_estimators max", value=300, min_value=100, max_value=500)

# Run optimization button
if st.sidebar.button("🚀 Run Optimization", type="primary"):
    # Create parameter configuration
    param_config = {
        'max_depth': (max_depth_min, max_depth_max),
        'learning_rate': (lr_min, lr_max),
        'n_estimators': (n_est_min, n_est_max)
    }
    
    # Load data
    with st.spinner("Loading dataset..."):
        X_train, X_test, y_train, y_test = load_adult_income_data()
    
    st.success(f"✓ Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Create and run study
    with st.spinner(f"Running optimization with {n_trials} trials... This may take a few minutes."):
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, X_train, y_train), 
            n_trials=n_trials,
            show_progress_bar=False
        )
    
    # Store results in session state
    st.session_state['study'] = study
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    
    st.success("✓ Optimization completed!")

# Display results
if 'study' in st.session_state:
    study = st.session_state['study']
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 History", "🎯 Best Parameters", "Learning rate/# Trees"])
    
    with tab1:
        st.header("Optimization Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Accuracy", f"{study.best_value:.4f}")
        with col2:
            st.metric("Best Trial", study.best_trial.number)
        with col3:
            st.metric("Total Trials", len(study.trials))
    
    with tab2:
        st.header("Optimization History")
        
        # Plot optimization history
        trials_data = []
        for trial in study.trials:
            if trial.value is not None:
                trials_data.append({
                    'Trial': trial.number,
                    'Accuracy': trial.value,
                    'learning_rate': trial.params.get('learning_rate', None),
                    'max_depth': trial.params.get('max_depth', None),
                    'n_estimators': trial.params.get('n_estimators', None),
                })
        
        df_trials = pd.DataFrame(trials_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_trials['Trial'],
            y=df_trials['Accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add best value line
        fig.add_trace(go.Scatter(
            x=df_trials['Trial'],
            y=[study.best_value] * len(df_trials),
            mode='lines',
            name='Best Accuracy',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Optimization History",
            xaxis_title="Trial Number",
            yaxis_title="Accuracy",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Best Hyperparameters")
        
        # Display best parameters
        best_params_df = pd.DataFrame([
            {'Parameter': k, 'Value': v}
            for k, v in study.best_params.items()
        ])
        
        st.dataframe(best_params_df, use_container_width=True, hide_index=True)
        
        # Parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(study)
            
            st.subheader("Parameter Importance")
            
            importance_df = pd.DataFrame([
                {'Parameter': k, 'Importance': v}
                for k, v in importance.items()
            ]).sort_values('Importance', ascending=False)
            
            fig_importance = go.Figure()
            fig_importance.add_trace(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Parameter'],
                orientation='h',
                marker=dict(color='lightblue')
            ))
            
            fig_importance.update_layout(
                title="Hyperparameter Importance",
                xaxis_title="Importance",
                yaxis_title="Parameter",
                height=400
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        except Exception as e:
            st.info("Parameter importance calculation requires at least 2 trials.")
    
    with tab4:
        st.header("Learning rate vs Number of Trees Scatter Plot")
        
        # Scatter plot of learning_rate vs n_estimators
        scatter_data = []
        for trial in study.trials:
            if trial.value is not None:
                scatter_data.append({
                    'learning_rate': trial.params.get('learning_rate', None),
                    'n_estimators': trial.params.get('n_estimators', None),
                    'Accuracy': trial.value
                })
        df_scatter = pd.DataFrame(scatter_data)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=df_scatter['learning_rate'],
            y=df_scatter['n_estimators'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_scatter['Accuracy'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Accuracy')
            )
        ))
        fig_scatter.update_layout(
            title="Learning Rate vs Number of Trees",
            xaxis_title="Learning Rate",
            yaxis_title="Number of Trees (n_estimators)",
            height=500
        )
        # Highlight the best learning rate and number of trees and mark it on the scatter
        best_lr = None
        best_n_est = None
        try:
            best_lr = study.best_params.get('learning_rate')
            best_n_est = study.best_params.get('n_estimators')
        except Exception:
            pass

        # Add a highlighted marker for the best point on the scatter (if both values are available)
        if best_lr is not None and best_n_est is not None:
            fig_scatter.add_annotation(
                x=best_lr,
                y=best_n_est,
                text="Best",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-40,
                bgcolor='rgba(255,255,255,0.8)'
            )
        st.plotly_chart(fig_scatter, use_container_width=True)


else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Optimization' to start!")
