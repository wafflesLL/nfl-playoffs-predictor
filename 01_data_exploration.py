"""
NFL Playoff Prediction - Data Exploration and Preprocessing
COSC325 Machine Learning Project - Midterm Report

This script performs:
1. Data loading and cleaning
2. Playoff label creation
3. Exploratory data analysis
4. Feature correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filepath):
    """Load NFL statistics dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def create_playoff_labels(df):
    """
    Create playoff qualification labels
    Assumption: Top 12 teams per season made playoffs (7 per conference in reality)
    Using points scored as tiebreaker for ranking
    """
    df_labeled = df.copy()
    df_labeled['made_playoffs'] = 0
    
    # For each season, label top 12 teams as playoff teams
    for season in df_labeled['season'].unique():
        season_mask = df_labeled['season'] == season
        # Sort by points scored (primary playoff indicator)
        season_df = df_labeled[season_mask].copy()
        season_df = season_df.sort_values('off_points_scored', ascending=False)
        
        # Top 12 teams make playoffs
        playoff_teams = season_df.head(12).index
        df_labeled.loc[playoff_teams, 'made_playoffs'] = 1
    
    print(f"\nPlayoff Distribution:")
    print(df_labeled['made_playoffs'].value_counts())
    print(f"Playoff percentage: {df_labeled['made_playoffs'].mean():.1%}")
    
    return df_labeled

def check_data_quality(df):
    """Check for missing values and data quality issues"""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found")
    else:
        print(f"Missing values:\n{missing[missing > 0]}")
    
    # Duplicates
    duplicates = df.duplicated(subset=['team', 'season']).sum()
    print(f"✓ Duplicate team-season records: {duplicates}")
    
    # Data types
    print(f"✓ All numerical features: {df.select_dtypes(include=[np.number]).shape[1]} columns")
    
    return True

def exploratory_analysis(df):
    """Perform exploratory data analysis comparing playoff vs non-playoff teams"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    playoff_teams = df[df['made_playoffs'] == 1]
    non_playoff_teams = df[df['made_playoffs'] == 0]
    
    # Key statistics comparison
    key_features = [
        'off_points_scored', 'def_points_allowed', 'turnover_margin',
        'off_yards_total', 'off_pass_touchdowns', 'off_rush_touchdowns',
        'eff_ratio', 'td_eff'
    ]
    
    print("\nPlayoff vs Non-Playoff Team Averages:")
    print("-" * 60)
    comparison = pd.DataFrame({
        'Playoff': playoff_teams[key_features].mean(),
        'Non-Playoff': non_playoff_teams[key_features].mean()
    })
    comparison['Difference'] = comparison['Playoff'] - comparison['Non-Playoff']
    print(comparison.round(2))
    
    # Correlation with playoff status
    correlations = df[key_features + ['made_playoffs']].corr()['made_playoffs'].drop('made_playoffs')
    correlations = correlations.sort_values(ascending=False)
    
    print("\n\nCorrelation with Playoff Qualification:")
    print("-" * 60)
    for feature, corr in correlations.items():
        print(f"{feature:30s}: {corr:6.3f}")
    
    return comparison, correlations

def select_features(df):
    """Select most relevant features for modeling"""
    # Features to use for modeling
    selected_features = [
        'off_points_scored', 'def_points_allowed',
        'off_yards_total', 'off_passing_yards', 'off_rushing_yards',
        'off_pass_touchdowns', 'off_rush_touchdowns',
        'off_first_downs_pass', 'off_first_downs_rush',
        'off_interceptions_thrown', 'off_fumbles_lost',
        'def_interceptions_made', 'def_sacks',
        'turnover_margin', 'eff_ratio', 'td_eff'
    ]
    
    print(f"\n\nSelected {len(selected_features)} features for modeling")
    
    return selected_features

def save_processed_data(df, selected_features, output_path):
    """Save processed dataset with selected features"""
    # Create final dataset with selected features and target
    final_df = df[['team', 'season'] + selected_features + ['made_playoffs']].copy()
    final_df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    return final_df

if __name__ == "__main__":
    # Load data
    df = load_data('nfl_stats_simplified_2012_to_2024.csv')
    
    # Create playoff labels
    df = create_playoff_labels(df)
    
    # Check data quality
    check_data_quality(df)
    
    # Exploratory analysis
    comparison, correlations = exploratory_analysis(df)
    
    # Select features
    selected_features = select_features(df)
    
    # Save processed data
    processed_df = save_processed_data(
        df, 
        selected_features, 
        'processed_nfl_data.csv'
    )
    
    print("\n" + "="*60)
    print("DATA EXPLORATION COMPLETE")
    print("="*60)