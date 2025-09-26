"""
🚀 Startup Success Prediction Challenge
=======================================
This script analyzes startup data and builds ML models to predict startup success.

Author: GitHub Copilot
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and perform initial exploration of the data"""
    print("🔍 Loading and exploring data...")
    
    # Load datasets
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_sub = pd.read_csv('sample_submission.csv')
    
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_sub.shape}")
    
    # Basic info about the dataset
    print("\n📊 Dataset Overview:")
    print(train_df.info())
    
    # Target variable distribution
    print(f"\n🎯 Target Distribution:")
    target_counts = train_df['labels'].value_counts()
    print(target_counts)
    print(f"Success rate: {target_counts[1] / len(train_df) * 100:.2f}%")
    
    # Missing values
    print(f"\n❓ Missing Values in Training Set:")
    missing_train = train_df.isnull().sum()
    missing_train = missing_train[missing_train > 0]
    if len(missing_train) > 0:
        print(missing_train)
    else:
        print("No missing values found!")
    
    print(f"\n❓ Missing Values in Test Set:")
    missing_test = test_df.isnull().sum()
    missing_test = missing_test[missing_test > 0]
    if len(missing_test) > 0:
        print(missing_test)
    else:
        print("No missing values found!")
    
    return train_df, test_df, sample_sub

def exploratory_data_analysis(train_df):
    """Perform comprehensive EDA"""
    print("\n📈 Performing Exploratory Data Analysis...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Target distribution
    plt.subplot(3, 4, 1)
    train_df['labels'].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
    plt.title('Target Distribution\n(0: Failure, 1: Success)', fontweight='bold')
    plt.xlabel('Startup Status')
    plt.ylabel('Count')
    
    # 2. Funding total distribution by success
    plt.subplot(3, 4, 2)
    plt.boxplot([train_df[train_df['labels'] == 0]['funding_total_usd'], 
                 train_df[train_df['labels'] == 1]['funding_total_usd']], 
                labels=['Failure', 'Success'])
    plt.title('Funding Distribution by Success', fontweight='bold')
    plt.ylabel('Total Funding (USD)')
    plt.yscale('log')
    
    # 3. Number of relationships vs success
    plt.subplot(3, 4, 3)
    plt.boxplot([train_df[train_df['labels'] == 0]['relationships'], 
                 train_df[train_df['labels'] == 1]['relationships']], 
                labels=['Failure', 'Success'])
    plt.title('Relationships by Success', fontweight='bold')
    plt.ylabel('Number of Relationships')
    
    # 4. Funding rounds vs success
    plt.subplot(3, 4, 4)
    plt.boxplot([train_df[train_df['labels'] == 0]['funding_rounds'], 
                 train_df[train_df['labels'] == 1]['funding_rounds']], 
                labels=['Failure', 'Success'])
    plt.title('Funding Rounds by Success', fontweight='bold')
    plt.ylabel('Number of Funding Rounds')
    
    # 5. Milestones vs success
    plt.subplot(3, 4, 5)
    plt.boxplot([train_df[train_df['labels'] == 0]['milestones'], 
                 train_df[train_df['labels'] == 1]['milestones']], 
                labels=['Failure', 'Success'])
    plt.title('Milestones by Success', fontweight='bold')
    plt.ylabel('Number of Milestones')
    
    # 6. Age at first funding vs success
    plt.subplot(3, 4, 6)
    train_df_clean = train_df.dropna(subset=['age_first_funding_year'])
    plt.boxplot([train_df_clean[train_df_clean['labels'] == 0]['age_first_funding_year'], 
                 train_df_clean[train_df_clean['labels'] == 1]['age_first_funding_year']], 
                labels=['Failure', 'Success'])
    plt.title('Age at First Funding by Success', fontweight='bold')
    plt.ylabel('Years')
    
    # 7. Category distribution
    plt.subplot(3, 4, 7)
    category_success = train_df.groupby('category_code')['labels'].agg(['count', 'sum'])
    category_success['success_rate'] = category_success['sum'] / category_success['count']
    top_categories = category_success.sort_values('count', ascending=False).head(8)
    top_categories['success_rate'].plot(kind='bar', color='skyblue')
    plt.title('Success Rate by Top Categories', fontweight='bold')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    
    # 8. State distribution
    plt.subplot(3, 4, 8)
    state_cols = ['is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate']
    state_success = {}
    for state in state_cols:
        if train_df[state].sum() > 10:  # Only states with significant presence
            success_rate = train_df[train_df[state] == 1]['labels'].mean()
            state_success[state.replace('is_', '')] = success_rate
    
    pd.Series(state_success).plot(kind='bar', color='lightgreen')
    plt.title('Success Rate by State', fontweight='bold')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=0)
    
    # 9. Funding type analysis
    plt.subplot(3, 4, 9)
    funding_types = ['has_VC', 'has_angel', 'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD']
    funding_success = {}
    for funding_type in funding_types:
        if train_df[funding_type].sum() > 10:
            success_rate = train_df[train_df[funding_type] == 1]['labels'].mean()
            funding_success[funding_type.replace('has_', '')] = success_rate
    
    pd.Series(funding_success).plot(kind='bar', color='orange')
    plt.title('Success Rate by Funding Type', fontweight='bold')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    
    # 10. Average participants vs success
    plt.subplot(3, 4, 10)
    train_df_clean = train_df.dropna(subset=['avg_participants'])
    plt.boxplot([train_df_clean[train_df_clean['labels'] == 0]['avg_participants'], 
                 train_df_clean[train_df_clean['labels'] == 1]['avg_participants']], 
                labels=['Failure', 'Success'])
    plt.title('Avg Participants by Success', fontweight='bold')
    plt.ylabel('Average Participants')
    
    # 11. Correlation heatmap for numerical features
    plt.subplot(3, 4, 11)
    numerical_cols = ['age_first_funding_year', 'age_last_funding_year', 'relationships', 
                      'funding_rounds', 'funding_total_usd', 'milestones', 'avg_participants', 'labels']
    corr_matrix = train_df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix', fontweight='bold')
    
    # 12. Success rate by industry type
    plt.subplot(3, 4, 12)
    industry_cols = ['is_software', 'is_web', 'is_mobile', 'is_enterprise', 
                     'is_advertising', 'is_gamesvideo', 'is_ecommerce', 'is_biotech']
    industry_success = {}
    for industry in industry_cols:
        if train_df[industry].sum() > 10:
            success_rate = train_df[train_df[industry] == 1]['labels'].mean()
            industry_success[industry.replace('is_', '')] = success_rate
    
    pd.Series(industry_success).plot(kind='bar', color='purple', alpha=0.7)
    plt.title('Success Rate by Industry', fontweight='bold')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('startup_eda.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key insights
    print("\n🎯 Key Insights:")
    overall_success_rate = train_df['labels'].mean()
    print(f"• Overall success rate: {overall_success_rate:.2%}")
    
    # Success rate by funding amount quartiles (temporary for analysis only)
    train_temp = train_df.copy()
    train_temp['funding_quartile'] = pd.qcut(train_temp['funding_total_usd'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    funding_success = train_temp.groupby('funding_quartile')['labels'].mean()
    print(f"• Success rate by funding quartile:")
    for q, rate in funding_success.items():
        print(f"  - {q}: {rate:.2%}")
    
    # Most successful categories
    print(f"• Top 3 most successful categories:")
    top_success_categories = category_success.sort_values('success_rate', ascending=False).head(3)
    for idx, (category, data) in enumerate(top_success_categories.iterrows(), 1):
        print(f"  {idx}. {category}: {data['success_rate']:.2%} ({data['count']} startups)")

def feature_engineering(train_df, test_df):
    """Create new features and prepare data for modeling"""
    print("\n🔧 Feature Engineering...")
    
    def engineer_features(df):
        df = df.copy()
        
        # Handle missing values in age columns with median
        age_cols = ['age_first_funding_year', 'age_last_funding_year', 
                   'age_first_milestone_year', 'age_last_milestone_year']
        for col in age_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill avg_participants with median
        df['avg_participants'].fillna(df['avg_participants'].median(), inplace=True)
        
        # Create new features
        df['funding_per_round'] = df['funding_total_usd'] / np.maximum(df['funding_rounds'], 1)
        df['milestones_per_year'] = df['milestones'] / np.maximum(df['age_last_funding_year'], 1)
        df['relationships_per_milestone'] = df['relationships'] / np.maximum(df['milestones'], 1)
        df['has_milestones'] = (df['milestones'] > 0).astype(int)
        df['has_multiple_rounds'] = (df['funding_rounds'] > 1).astype(int)
        df['high_funding'] = (df['funding_total_usd'] > df['funding_total_usd'].median()).astype(int)
        df['funding_efficiency'] = df['funding_total_usd'] / np.maximum(df['relationships'], 1)
        
        # Time-based features
        df['funding_duration'] = df['age_last_funding_year'] - df['age_first_funding_year']
        df['milestone_duration'] = df['age_last_milestone_year'] - df['age_first_milestone_year']
        df['early_milestone'] = (df['age_first_milestone_year'] < 2).astype(int)
        
        # Log transform skewed features
        df['log_funding'] = np.log1p(df['funding_total_usd'])
        df['log_relationships'] = np.log1p(df['relationships'])
        
        # Interaction features
        df['vc_angel_combo'] = df['has_VC'] * df['has_angel']
        df['ca_tech_combo'] = df['is_CA'] * (df['is_software'] + df['is_web'] + df['is_mobile'])
        
        # Fill infinite values and remaining NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Handle numerical columns separately
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(0)
        
        return df
    
    # Apply feature engineering
    train_engineered = engineer_features(train_df)
    test_engineered = engineer_features(test_df)
    
    print(f"Original features: {train_df.shape[1] - 1}")  # -1 for labels column
    print(f"Engineered features: {train_engineered.shape[1] - 1}")  # -1 for labels column
    
    return train_engineered, test_engineered

def prepare_data_for_modeling(train_df, test_df):
    """Prepare features and target for machine learning models"""
    print("\n⚙️ Preparing data for modeling...")
    
    # Separate features and target
    feature_cols = [col for col in train_df.columns if col not in ['id', 'labels', 'category_code']]
    
    X = train_df[feature_cols].copy()
    y = train_df['labels'].copy()
    X_test = test_df[feature_cols].copy()
    
    # Handle categorical variables (category_code) separately if needed
    # For now, we'll exclude it as it's already encoded via one-hot columns
    
    print(f"Features selected: {len(feature_cols)}")
    print(f"Training samples: {X.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return X, y, X_test, feature_cols

def train_and_evaluate_models(X, y, feature_cols):
    """Train multiple models and evaluate their performance"""
    print("\n🤖 Training and evaluating models...")
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for Logistic Regression
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_val, y_pred_proba)
        accuracy = (y_pred == y_val).mean()
        
        results[name] = {
            'AUC': auc_score,
            'Accuracy': accuracy,
            'Model': model,
            'Scaler': scaler if name == 'Logistic Regression' else None
        }
        
        trained_models[name] = model
        
        print(f"{name} - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['AUC'])
    best_model = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"Best AUC Score: {best_model['AUC']:.4f}")
    
    # Feature importance for tree-based models
    if hasattr(best_model['Model'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model['Model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (idx, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        feature_importance.head(15).plot(x='feature', y='importance', kind='barh', 
                                       color='skyblue', legend=False)
        plt.title(f'Top 15 Feature Importance\n({best_model_name})', fontweight='bold')
        plt.xlabel('Importance')
        
        # ROC Curve
        plt.subplot(1, 2, 2)
        if best_model['Scaler'] is not None:
            y_pred_proba = best_model['Model'].predict_proba(X_val_scaled)[:, 1]
        else:
            y_pred_proba = best_model['Model'].predict_proba(X_val)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {best_model["AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {best_model_name}', fontweight='bold')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return best_model, results, scaler

def hyperparameter_tuning(X, y, best_model_info):
    """Perform hyperparameter tuning on the best model"""
    print("\n🎛️ Performing hyperparameter tuning...")
    
    model = best_model_info['Model']
    model_name = type(model).__name__
    
    # Define parameter grids for different models
    param_grids = {
        'RandomForestClassifier': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'XGBClassifier': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        'GradientBoostingClassifier': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
    
    if model_name in param_grids:
        param_grid = param_grids[model_name]
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        print(f"No hyperparameter tuning defined for {model_name}")
        return model

def generate_predictions(best_model, X_test, test_df, scaler=None):
    """Generate predictions for the test set"""
    print("\n🎯 Generating predictions...")
    
    # Make predictions
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        predictions_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        predictions = best_model.predict(X_test_scaled)
    else:
        predictions_proba = best_model.predict_proba(X_test)[:, 1]
        predictions = best_model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_df['id'],
        'labels': predictions
    })
    
    # Save predictions
    submission.to_csv('startup_predictions.csv', index=False)
    
    print(f"Predictions saved to 'startup_predictions.csv'")
    print(f"Predicted success rate: {predictions.mean():.2%}")
    print(f"Prediction distribution:")
    print(f"  • Failures (0): {(predictions == 0).sum()}")
    print(f"  • Successes (1): {(predictions == 1).sum()}")
    
    return submission

def main():
    """Main execution function"""
    print("🚀 Startup Success Prediction Challenge")
    print("=" * 50)
    
    # Load and explore data
    train_df, test_df, sample_sub = load_and_explore_data()
    
    # Perform EDA
    exploratory_data_analysis(train_df)
    
    # Feature engineering
    train_engineered, test_engineered = feature_engineering(train_df, test_df)
    
    # Prepare data for modeling
    X, y, X_test, feature_cols = prepare_data_for_modeling(train_engineered, test_engineered)
    
    # Train and evaluate models
    best_model_info, all_results, scaler = train_and_evaluate_models(X, y, feature_cols)
    
    # Hyperparameter tuning
    tuned_model = hyperparameter_tuning(X, y, best_model_info)
    
    # Generate final predictions
    submission = generate_predictions(tuned_model, X_test, test_df, best_model_info['Scaler'])
    
    print("\n✅ Analysis and modeling completed successfully!")
    print("📁 Files generated:")
    print("  • startup_eda.png - Exploratory Data Analysis plots")
    print("  • model_evaluation.png - Model evaluation plots")
    print("  • startup_predictions.csv - Final predictions")
    
    return submission, tuned_model, all_results

if __name__ == "__main__":
    submission, model, results = main()