"""
NFL Playoff Prediction - Baseline Logistic Regression Model
COSC325 Machine Learning Project - Midterm Report

This script implements:
1. Train/test split (temporal split to prevent data leakage)
2. Feature standardization
3. Logistic regression training with L2 regularization
4. Model evaluation with comprehensive metrics
5. Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def load_processed_data(filepath):
    """Load the processed dataset"""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape}")
    return df

def temporal_train_test_split(df, test_start_year=2023):
    """
    Split data temporally to prevent data leakage
    Train on seasons before test_start_year, test on remaining seasons
    """
    train_df = df[df['season'] < test_start_year].copy()
    test_df = df[df['season'] >= test_start_year].copy()
    
    print(f"\nTrain/Test Split:")
    print(f"  Training: {train_df['season'].min()}-{train_df['season'].max()} ({len(train_df)} samples)")
    print(f"  Testing:  {test_df['season'].min()}-{test_df['season'].max()} ({len(test_df)} samples)")
    print(f"  Train playoff rate: {train_df['made_playoffs'].mean():.1%}")
    print(f"  Test playoff rate:  {test_df['made_playoffs'].mean():.1%}")
    
    return train_df, test_df

def prepare_features(train_df, test_df):
    """
    Prepare feature matrices and target vectors
    Exclude non-feature columns (team, season, target)
    """
    # Identify feature columns (exclude team, season, made_playoffs)
    feature_cols = [col for col in train_df.columns 
                   if col not in ['team', 'season', 'made_playoffs']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['made_playoffs'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['made_playoffs'].values
    
    print(f"\nFeature Matrix Shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"\nFeatures used: {len(feature_cols)}")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def standardize_features(X_train, X_test):
    """
    Apply z-score standardization
    Fit scaler on training data only to prevent data leakage
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nStandardization complete")
    print(f"  Training mean: {X_train_scaled.mean():.6f}")
    print(f"  Training std:  {X_train_scaled.std():.6f}")
    print(f"  Test mean:     {X_test_scaled.mean():.6f}")
    print(f"  Test std:      {X_test_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler

def train_baseline_model(X_train, y_train):
    """
    Train logistic regression model with L2 regularization
    """
    print("\n" + "="*60)
    print("TRAINING BASELINE LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    model = LogisticRegression(
        C=1.0,                    # Inverse of regularization strength
        penalty='l2',             # L2 regularization (Ridge)
        solver='liblinear',       # Good for small datasets
        max_iter=1000,           # Maximum iterations
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    print(f"✓ Model trained successfully")
    print(f"  Training accuracy: {train_acc:.4f}")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Print results
    print("\nPerformance Metrics:")
    print("-" * 40)
    print(f"Training Accuracy:   {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy:       {metrics['test_accuracy']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"F1-Score:            {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print("-" * 40)
    print(f"                Predicted")
    print(f"              No     Yes")
    print(f"Actual  No  | {cm[0,0]:3d} | {cm[0,1]:3d} |")
    print(f"        Yes | {cm[1,0]:3d} | {cm[1,1]:3d} |")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print("-" * 40)
    print(classification_report(y_test, y_test_pred, 
                               target_names=['Non-Playoff', 'Playoff'],
                               digits=3))
    
    return metrics, cm, y_test_pred, y_test_proba

def analyze_feature_importance(model, feature_names):
    """
    Analyze and display feature importance from logistic regression coefficients
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Create dataframe for better visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print("-" * 60)
    for idx, row in feature_importance.head(10).iterrows():
        impact = "Positive" if row['Coefficient'] > 0 else "Negative"
        print(f"{row['Feature']:30s}: {row['Coefficient']:7.4f} ({impact})")
    
    return feature_importance

def create_evaluation_plots(y_test, y_test_pred, y_test_proba, cm):
    """
    Create visualization plots for model evaluation
    """
    print("\nCreating evaluation visualizations...")
    
    # Figure 1: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-Playoff', 'Playoff'],
               yticklabels=['Non-Playoff', 'Playoff'],
               cbar_kws={'label': 'Count'})
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Baseline Model', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrix.png")
    plt.close()
    
    # Figure 2: ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})', color='#3498db')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Baseline Logistic Regression', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roc_curve.png")
    plt.close()

def save_results(metrics, feature_importance):
    """
    Save numerical results to JSON file
    """
    results = {
        'model': 'Logistic Regression (L2)',
        'metrics': metrics,
        'top_features': feature_importance.head(10).to_dict('records')
    }
    
    with open('results/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to: baseline_results.json")

def main():
    """
    Main execution function
    """
    print("="*60)
    print("NFL PLAYOFF PREDICTION - BASELINE MODEL")
    print("="*60)
    
    # Load data
    df = load_processed_data('processed_nfl_data.csv')
    
    # Temporal train/test split
    train_df, test_df = temporal_train_test_split(df, test_start_year=2023)
    
    # Prepare features
    X_train, X_test, y_train, y_test, feature_cols = prepare_features(train_df, test_df)
    
    # Standardize features
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    # Train model
    model = train_baseline_model(X_train_scaled, y_train)
    
    # Evaluate model
    metrics, cm, y_test_pred, y_test_proba = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Feature importance
    feature_importance = analyze_feature_importance(model, feature_cols)
    
    # Create plots
    create_evaluation_plots(y_test, y_test_pred, y_test_proba, cm)
    
    # Save results
    save_results(metrics, feature_importance)
    
    print("\n" + "="*60)
    print("BASELINE MODEL COMPLETE")
    print("="*60)
    print("\nNext Steps for Final Report:")
    print("  1. Implement k-fold cross-validation")
    print("  2. Hyperparameter tuning with GridSearchCV")
    print("  3. Address class imbalance")
    print("  4. Compare with advanced models (Random Forest, XGBoost)")
    print("  5. Feature engineering improvements")

if __name__ == "__main__":
    main()