"""
NFL Playoff Prediction - Visualizations
Creates figures for midterm report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Load processed data
df = pd.read_csv('processed_nfl_data.csv')

# Create output directory for figures
os.makedirs('figures', exist_ok=True)

# ============================================================
# Figure 1: Correlation Heatmap
# ============================================================
print("Creating Figure 1: Correlation Heatmap...")

# Select key features for correlation matrix
key_features = [
    'off_points_scored', 'def_points_allowed', 'turnover_margin',
    'off_pass_touchdowns', 'off_rush_touchdowns', 'eff_ratio', 
    'td_eff', 'off_yards_total', 'made_playoffs'
]

corr_matrix = df[key_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Key NFL Statistics', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_heatmap.png")
plt.close()

# ============================================================
# Figure 2: Playoff vs Non-Playoff Distributions
# ============================================================
print("Creating Figure 2: Feature Distributions...")

features_to_plot = [
    ('off_points_scored', 'Points Scored'),
    ('def_points_allowed', 'Points Allowed'),
    ('turnover_margin', 'Turnover Margin'),
    ('td_eff', 'Touchdown Efficiency')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, (feature, label) in enumerate(features_to_plot):
    playoff = df[df['made_playoffs'] == 1][feature]
    non_playoff = df[df['made_playoffs'] == 0][feature]
    
    axes[idx].hist(non_playoff, bins=20, alpha=0.6, label='Non-Playoff', color='#e74c3c')
    axes[idx].hist(playoff, bins=20, alpha=0.6, label='Playoff', color='#3498db')
    axes[idx].axvline(non_playoff.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Non-Playoff Mean: {non_playoff.mean():.1f}')
    axes[idx].axvline(playoff.mean(), color='#3498db', linestyle='--', linewidth=2, label=f'Playoff Mean: {playoff.mean():.1f}')
    axes[idx].set_xlabel(label, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Distribution Comparison: Playoff vs Non-Playoff Teams', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('figures/feature_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_distributions.png")
plt.close()

# ============================================================
# Figure 3: Box Plots for Key Statistics
# ============================================================
print("Creating Figure 3: Box Plot Comparisons...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

stats_to_plot = [
    ('off_points_scored', 'Offensive Points Scored'),
    ('turnover_margin', 'Turnover Margin'),
    ('eff_ratio', 'Efficiency Ratio')
]

for idx, (stat, title) in enumerate(stats_to_plot):
    data_to_plot = [
        df[df['made_playoffs'] == 0][stat],
        df[df['made_playoffs'] == 1][stat]
    ]
    
    bp = axes[idx].boxplot(data_to_plot, tick_labels=['Non-Playoff', 'Playoff'],
                           patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = ['#e74c3c', '#3498db']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Value', fontsize=10)
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.suptitle('Statistical Comparison: Playoff Qualification', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/boxplot_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: boxplot_comparison.png")
plt.close()

# ============================================================
# Figure 4: Scatter Plot - Points Scored vs Allowed
# ============================================================
print("Creating Figure 4: Scatter Plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot non-playoff teams
non_playoff = df[df['made_playoffs'] == 0]
ax.scatter(non_playoff['off_points_scored'], non_playoff['def_points_allowed'],
          alpha=0.6, s=80, c='#e74c3c', label='Non-Playoff', edgecolors='black', linewidth=0.5)

# Plot playoff teams
playoff = df[df['made_playoffs'] == 1]
ax.scatter(playoff['off_points_scored'], playoff['def_points_allowed'],
          alpha=0.8, s=100, c='#3498db', label='Playoff', edgecolors='black', linewidth=0.5, marker='^')

ax.set_xlabel('Offensive Points Scored', fontsize=13, fontweight='bold')
ax.set_ylabel('Defensive Points Allowed', fontsize=13, fontweight='bold')
ax.set_title('NFL Team Performance: Offense vs Defense', fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/scatter_points.png', dpi=300, bbox_inches='tight')
print("✓ Saved: scatter_points.png")
plt.close()

# ============================================================
# Figure 5: Feature Importance Preview (from correlations)
# ============================================================
print("Creating Figure 5: Feature Correlations...")

# Get correlations with playoff status
feature_list = [
    'off_points_scored', 'td_eff', 'eff_ratio', 'off_pass_touchdowns',
    'off_yards_total', 'turnover_margin', 'off_rush_touchdowns',
    'def_points_allowed'
]

correlations = df[feature_list + ['made_playoffs']].corr()['made_playoffs'].drop('made_playoffs')
correlations = correlations.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if x < 0 else '#3498db' for x in correlations.values]
correlations.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.7)
ax.set_xlabel('Correlation with Playoff Qualification', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Feature Correlation Analysis', fontsize=16, fontweight='bold', pad=15)
ax.axvline(x=0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figures/feature_correlations.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_correlations.png")
plt.close()

print("\n" + "="*60)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY")
print("Location: figures/")
print("="*60)