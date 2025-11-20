import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
from sklearn.model_selection import (
    KFold, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, StratifiedKFold, learning_curve, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, make_scorer
)
from sklearn.calibration import CalibratedClassifierCV
from statsmodels.stats.contingency_tables import mcnemar
import warnings
import json
import os
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs('results/enhanced_4models', exist_ok=True)
os.makedirs('figures/enhanced_4models', exist_ok=True)
os.makedirs('models', exist_ok=True)

class EnhancedNFLPlayoffPredictor:
    def __init__(self, weeks, experiment_name=None):
        self.weeks = weeks
        self.experiment_name = experiment_name or f"enhanced_4models_{weeks}weeks"
        self.models = {}
        self.best_model = None
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self):
        print(f"Loading {self.weeks}-week data...")
        
        weekly_df = pd.read_csv(f'./weeks/weekly_stats_{self.weeks}.csv')
        
        agg_dict = {col: 'mean' for col in weekly_df.columns 
                   if col not in ['team', 'season', 'week']}
        self.df_agg = weekly_df.groupby(['team', 'season']).agg(agg_dict).reset_index()
        
        playoffs = pd.read_csv('./weeks/playoffs_2021_to_2024.csv')
        playoffs_long = playoffs.melt(id_vars=['team'], 
                                     var_name='season', 
                                     value_name='made_playoffs')
        playoffs_long['season'] = playoffs_long['season'].astype(int)
        
        self.full_df = pd.merge(self.df_agg, playoffs_long, 
                               on=['team', 'season'], how='inner')
        
        return self.full_df
    
    def engineer_features(self, df):
        df_eng = df.copy()
        
        df_eng['points_efficiency'] = df_eng['off_points_scored'] / (
            df_eng['def_points_allowed'] + 1)
        df_eng['yards_efficiency'] = df_eng['off_yards_total'] / (
            df_eng['off_yards_total'].mean() + 1)
        
        df_eng['offensive_balance'] = abs(
            df_eng['off_pass_percentage'] - 0.5)
        df_eng['scoring_consistency'] = df_eng.groupby('team')['off_points_scored'].transform(
            lambda x: x.std() if len(x) > 1 else 0)
        
        df_eng['prev_season_playoffs'] = df_eng.sort_values('season').groupby('team')[
            'made_playoffs'].shift(1).fillna(0.5)
        
        for col in ['off_points_scored', 'def_points_allowed', 'off_yards_total', 'turnover_margin']:
            if col in df_eng.columns:
                df_eng[f'{col}_zscore'] = df_eng.groupby('season')[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8))
                df_eng[f'{col}_rank'] = df_eng.groupby('season')[col].rank(pct=True)
        
        if 'off_pass_touchdowns' in df_eng.columns and 'off_rush_touchdowns' in df_eng.columns:
            df_eng['total_offensive_tds'] = (df_eng['off_pass_touchdowns'] + 
                                            df_eng['off_rush_touchdowns'])
        
        if 'off_passing_yards' in df_eng.columns and 'off_rushing_yards' in df_eng.columns:
            df_eng['pass_rush_ratio'] = df_eng['off_passing_yards'] / (
                df_eng['off_rushing_yards'] + 1)
        
        if 'def_sacks' in df_eng.columns and 'def_interceptions_made' in df_eng.columns:
            df_eng['defensive_plays'] = (df_eng['def_sacks'] + 
                                        df_eng['def_interceptions_made'])
        
        if 'td_eff' in df_eng.columns:
            df_eng['red_zone_efficiency'] = df_eng['td_eff'] * df_eng['off_points_scored']
        
        return df_eng
    
    def select_features(self, X_train, y_train, method='mutual_info', k=30):
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
        elif method == 'rfe':
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            selector = RFE(estimator, n_features_to_select=min(k, X_train.shape[1]))
        else:
            return X_train, None
        
        X_selected = selector.fit_transform(X_train, y_train)
        return X_selected, selector
    
    def prepare_data(self, test_year=2024):
        self.full_df_eng = self.engineer_features(self.full_df)
        
        self.train_df = self.full_df_eng[self.full_df_eng['season'] < test_year].copy()
        self.test_df = self.full_df_eng[self.full_df_eng['season'] == test_year].copy()
        
        print(f"Training samples: {len(self.train_df)} | Test samples: {len(self.test_df)}")
        
        self.feature_cols = [col for col in self.train_df.columns 
                            if col not in ['team', 'season', 'made_playoffs']]
        
        self.X_train = self.train_df[self.feature_cols].values
        self.y_train = self.train_df['made_playoffs'].values
        self.X_test = self.test_df[self.feature_cols].values
        self.y_test = self.test_df['made_playoffs'].values
        
        self.X_train = np.nan_to_num(self.X_train, nan=0)
        self.X_test = np.nan_to_num(self.X_test, nan=0)
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def get_model_configs(self):
        return {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 1.0]
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=5):
        print("Hyperparameter Tuning...")
        
        configs = self.get_model_configs()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results = {}
        
        for name, config in configs.items():
            print(f"  Tuning {name}...")
            
            search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            
            results[name] = {
                'best_model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
            
            print(f"    Best ROC-AUC: {search.best_score_:.4f}")
            
            self.models[name] = search.best_estimator_
        
        return results
    
    def create_ensemble(self, X_train, y_train):
        estimators = [(name, model) for name, model in self.models.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        return ensemble
    
    def calibrate_probabilities(self, model, X_train, y_train):
        calibrated = CalibratedClassifierCV(model, cv=3, method='sigmoid')
        calibrated.fit(X_train, y_train)
        return calibrated
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_train_proba = model.decision_function(X_train)
            y_test_proba = model.decision_function(X_test)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
            'train_roc_auc': roc_auc_score(y_train, y_train_proba),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'test_avg_precision': average_precision_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        cm = confusion_matrix(y_test, y_test_pred)
        
        self.results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba
        }
        
        return metrics, cm, y_test_pred, y_test_proba
    
    def plot_learning_curves(self, model, X, y, model_name):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 
                'o-', color='b', label='Training score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 
                'o-', color='r', label='Validation score')
        plt.fill_between(train_sizes, 
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1, color='b')
        plt.fill_between(train_sizes,
                        np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                        alpha=0.1, color='r')
        plt.xlabel('Training Set Size')
        plt.ylabel('ROC-AUC Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'figures/enhanced_4models/learning_curve_{model_name}_{self.weeks}weeks.png')
        plt.close()
    
    def plot_roc_curves_comparison(self):
        plt.figure(figsize=(12, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#34495e']
        
        for (model_name, results), color in zip(self.results.items(), colors):
            if 'y_test_proba' in results:
                fpr, tpr, _ = roc_curve(self.y_test, results['y_test_proba'])
                auc = results['metrics']['test_roc_auc']
                plt.plot(fpr, tpr, linewidth=2.5, 
                        label=f'{model_name} (AUC = {auc:.3f})',
                        color=color)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves Comparison - {self.weeks} Weeks', fontsize=14)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'figures/enhanced_4models/roc_comparison_{self.weeks}weeks.png', dpi=300)
        plt.close()
    
    def plot_precision_recall_curves(self):
        plt.figure(figsize=(12, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#34495e']
        
        for (model_name, results), color in zip(self.results.items(), colors):
            if 'y_test_proba' in results:
                precision, recall, _ = precision_recall_curve(
                    self.y_test, results['y_test_proba']
                )
                avg_precision = results['metrics']['test_avg_precision']
                plt.plot(recall, precision, linewidth=2.5,
                        label=f'{model_name} (AP = {avg_precision:.3f})',
                        color=color)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curves - {self.weeks} Weeks', fontsize=14)
        plt.legend(loc='lower left', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'figures/enhanced_4models/precision_recall_{self.weeks}weeks.png', dpi=300)
        plt.close()
    
    def plot_model_comparison(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(self.results.keys())
        
        train_acc = [self.results[m]['metrics']['train_accuracy'] for m in model_names]
        test_acc = [self.results[m]['metrics']['test_accuracy'] for m in model_names]
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, train_acc, width, label='Train', color='#3498db')
        axes[0, 0].bar(x_pos + width/2, test_acc, width, label='Test', color='#e74c3c')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        roc_scores = [self.results[m]['metrics']['test_roc_auc'] for m in model_names]
        axes[0, 1].bar(model_names, roc_scores, color='#2ecc71')
        axes[0, 1].set_ylabel('ROC-AUC Score')
        axes[0, 1].set_title('ROC-AUC Comparison')
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, v in enumerate(roc_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        f1_scores = [self.results[m]['metrics']['test_f1'] for m in model_names]
        axes[1, 0].bar(model_names, f1_scores, color='#f39c12')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('F1-Score Comparison')
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        precision = [self.results[m]['metrics']['test_precision'] for m in model_names]
        recall = [self.results[m]['metrics']['test_recall'] for m in model_names]
        
        axes[1, 1].scatter(recall, precision, s=200, alpha=0.6)
        for i, txt in enumerate(model_names):
            axes[1, 1].annotate(txt, (recall[i], precision[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Recall Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance Comparison - {self.weeks} Weeks', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'figures/enhanced_4models/model_comparison_{self.weeks}weeks.png', dpi=300)
        plt.close()
    
    def analyze_feature_importance(self):
        importance_data = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                continue
            
            if model_name == 'ensemble':
                continue
            
            importance_df = pd.DataFrame({
                'feature': self.feature_cols[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            importance_data[model_name] = importance_df
            
            plt.figure(figsize=(10, 8))
            top_15 = importance_df.head(15)
            colors = ['#e74c3c' if i < 5 else '#3498db' if i < 10 else '#95a5a6' 
                     for i in range(len(top_15))]
            plt.barh(range(len(top_15)), top_15['importance'].values, color=colors)
            plt.yticks(range(len(top_15)), top_15['feature'].values)
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {model_name} ({self.weeks} weeks)')
            plt.tight_layout()
            plt.savefig(f'figures/enhanced_4models/features_{model_name}_{self.weeks}weeks.png')
            plt.close()
        
        self.feature_importance = importance_data
        return importance_data
    
    def statistical_significance_test(self):
        print("Statistical Significance Testing (McNemar's Test)...")
        
        significance_results = {}
        model_names = list(self.results.keys())
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                pred1 = self.results[model1]['y_test_pred']
                pred2 = self.results[model2]['y_test_pred']
                
                correct_both = np.sum((pred1 == self.y_test) & (pred2 == self.y_test))
                wrong_both = np.sum((pred1 != self.y_test) & (pred2 != self.y_test))
                model1_right_model2_wrong = np.sum((pred1 == self.y_test) & (pred2 != self.y_test))
                model1_wrong_model2_right = np.sum((pred1 != self.y_test) & (pred2 == self.y_test))
                
                contingency_table = [[correct_both, model1_wrong_model2_right],
                                    [model1_right_model2_wrong, wrong_both]]
                
                if model1_right_model2_wrong + model1_wrong_model2_right > 0:
                    result = mcnemar(contingency_table, exact=False)
                    
                    significance_results[f'{model1}_vs_{model2}'] = {
                        'statistic': result.statistic,
                        'p_value': result.pvalue,
                        'significant': result.pvalue < 0.05
                    }
                    
                    print(f"  {model1} vs {model2}: p={result.pvalue:.4f}")
        
        return significance_results
    
    def save_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_summary = {
            'experiment': self.experiment_name,
            'weeks': self.weeks,
            'timestamp': timestamp,
            'model_results': {},
            'best_model': None,
            'feature_importance': {}
        }
        
        best_score = 0
        best_model_name = None
        
        for model_name, results in self.results.items():
            results_summary['model_results'][model_name] = results['metrics']
            if results['metrics']['test_roc_auc'] > best_score:
                best_score = results['metrics']['test_roc_auc']
                best_model_name = model_name
        
        results_summary['best_model'] = {
            'name': best_model_name,
            'test_roc_auc': best_score,
            'test_accuracy': self.results[best_model_name]['metrics']['test_accuracy']
        }
        
        for model_name, importance_df in self.feature_importance.items():
            results_summary['feature_importance'][model_name] = \
                importance_df.head(10).to_dict('records')
        
        with open(f'results/enhanced_4models/results_{self.weeks}weeks_{timestamp}.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        if best_model_name:
            joblib.dump(self.models[best_model_name], 
                       f'models/best_model_{self.weeks}weeks_{timestamp}.pkl')
            print(f"Best model ({best_model_name}) saved!")
        
        return results_summary
    
    def run_complete_pipeline(self):
        print(f"\nENHANCED NFL PLAYOFF PREDICTOR - {self.weeks} WEEKS\n")
        
        self.load_data()
        self.prepare_data()
        
        X_selected, selector = self.select_features(
            self.X_train_scaled, self.y_train, 
            method='mutual_info', k=30
        )
        
        tuning_results = self.hyperparameter_tuning(
            self.X_train_scaled, self.y_train
        )
        
        ensemble = self.create_ensemble(self.X_train_scaled, self.y_train)
        
        for model_name in ['logistic_regression', 'xgboost', 'ensemble']:
            if model_name in self.models:
                calibrated = self.calibrate_probabilities(
                    self.models[model_name], 
                    self.X_train_scaled, 
                    self.y_train
                )
                self.models[f'{model_name}_calibrated'] = calibrated
        
        print("\nEvaluating Models...")
        for model_name, model in self.models.items():
            metrics, cm, y_pred, y_proba = self.evaluate_model(
                model, 
                self.X_train_scaled, 
                self.X_test_scaled,
                self.y_train, 
                self.y_test, 
                model_name
            )
            print(f"  {model_name}: Accuracy={metrics['test_accuracy']:.3f}, ROC-AUC={metrics['test_roc_auc']:.3f}")
        
        self.plot_roc_curves_comparison()
        self.plot_precision_recall_curves()
        self.plot_model_comparison()
        
        for model_name in ['logistic_regression', 'xgboost', 'ensemble']:
            if model_name in self.models:
                self.plot_learning_curves(
                    self.models[model_name], 
                    self.X_train_scaled, 
                    self.y_train, 
                    model_name
                )
        
        self.analyze_feature_importance()
        self.statistical_significance_test()
        
        results_summary = self.save_results()
        
        print("\n✅ Pipeline completed successfully!\n")
        
        return results_summary


def main():
    all_results = {}
    
    for weeks in [4, 6, 10]:
        print(f"\n{'='*60}")
        print(f"Processing {weeks} Weeks Model")
        print('='*60)
        
        predictor = EnhancedNFLPlayoffPredictor(weeks)
        results = predictor.run_complete_pipeline()
        all_results[f'{weeks}_weeks'] = results
    
    print("\nFINAL COMPARISON")
    print("="*60)
    
    comparison_data = []
    for weeks in [4, 6, 10]:
        week_results = all_results[f'{weeks}_weeks']
        best = week_results['best_model']
        comparison_data.append({
            'Weeks': weeks,
            'Best Model': best['name'],
            'Test Accuracy': f"{best['test_accuracy']:.1%}",
            'Test ROC-AUC': f"{best['test_roc_auc']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))
    
    with open('results/enhanced_4models/final_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    


if __name__ == "__main__":
    main()