# Remote Work Satisfaction Prediction

A machine learning project that predicts employee satisfaction with remote work using AutoGluon and XGBoost, with advanced techniques including SMOTE for class balancing and SHAP for model interpretability.This project analyzes the impact of remote work on mental health and builds a binary classification model to predict employee satisfaction with remote work arrangements. The pipeline includes:

- **Data Processing**: PySpark-based data cleaning and preprocessing
- **Class Balancing**: SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous)
- **Model Training**: AutoGluon AutoML with ensemble methods
- **Model Validation**: XGBoost with hyperparameter tuning
- **Interpretability**: SHAP (SHapley Additive exPlanations) analysis

## 🎯 Target Variable

**`Satisfaction_with_Remote_Work`** - Binary classification predicting whether employees are satisfied with remote work arrangements.

## 📊 Dataset

- **Source**: `Impact_of_Remote_Work_on_Mental_Health.csv`
- **Features**: Employee demographics, work conditions, mental health indicators, and remote work factors
- **Excluded Columns**: `Employee_ID` (non-predictive identifier)

## 🛠️ Technology Stack

### Core Libraries
- **PySpark**: Large-scale data processing and transformation
- **AutoGluon**: Automated machine learning framework
- **XGBoost**: Gradient boosting for validation
- **imbalanced-learn**: SMOTENC for handling class imbalance
- **SHAP**: Model interpretability and feature importance

### Visualization & Analysis
- **Matplotlib & Seaborn**: Statistical visualizations
- **Pandas & NumPy**: Data manipulation

## 🚀 Installation

```bash
# Upgrade pip and install core dependencies
pip install -U pip setuptools wheel

# Install MXNet (required for AutoGluon)
pip install -U "mxnet<2.0.0"

# Install AutoGluon
pip install autogluon --no-cache-dir

# Install additional dependencies
pip install imbalanced-learn shap xgboost pyspark matplotlib seaborn
```

## 🔄 Pipeline Workflow

### 1. **Data Loading & Initialization**
- Load CSV using PySpark with schema inference
- Define target variable and configuration parameters
- Set random seed for reproducibility (SEED=42)

### 2. **Data Cleaning**
- **Column Sanitization**: Rename columns to remove special characters and ensure valid naming
- **Schema Filtering**: Keep only scalar types (numeric, categorical, date/time)
- **Target Filtering**: Remove rows with null target values

### 3. **Feature Engineering & Imputation**
- **Numeric Features**: Median imputation using `percentile_approx`
- **Categorical Features**: Mode imputation with fallback to "Unknown"
- **Target Preservation**: Ensure target column is not imputed

### 4. **Train-Test Split**
- **Split Ratio**: 90% training, 10% testing
- **Method**: Random stratified split using PySpark
- **Validation**: Label distribution visualization across splits

### 5. **Class Imbalancing Handling**
- **Technique**: SMOTENC (handles mixed categorical/numeric features)
- **Strategy**: Auto-balancing to equalize class distributions
- **Process**:
  - Encode categorical features
  - Apply synthetic oversampling
  - Decode categorical features back to original representation

### 6. **AutoGluon Training**
- **Configuration**:
  - Problem Type: Binary classification
  - Preset: `best_quality` (maximum model quality)
  - Time Limit: 1800 seconds (30 minutes)
  - Bagging: 5-fold cross-validation
  - Stacking: 1 level of ensemble stacking
  - Evaluation Metric: Accuracy

- **Output**:
  - Ensemble of multiple model types (Random Forest, LightGBM, CatBoost, Neural Networks, etc.)
  - Automatic hyperparameter optimization
  - Model leaderboard with performance metrics

### 7. **Model Evaluation**
- **Metrics**:
  - Test set accuracy
  - Confusion matrix visualization
  - Per-model leaderboard scores
  
- **Visualizations**:
  - Model accuracy comparison (horizontal bar chart)
  - Confusion matrix heatmap
  - Top 20 feature importances

### 8. **XGBoost Validation**
- **Purpose**: Independent validation using a different framework
- **Configuration**:
  - 600 estimators
  - Max depth: 6
  - Learning rate: 0.05
  - Subsample: 0.85
  - Column sample: 0.85
  - L2 regularization: 1.0

- **Data Encoding**:
  - One-hot encoding for categorical features
  - Aligned encoding between train/test sets
  - Label encoding for target variable

### 9. **SHAP Analysis**
- **Explainer**: TreeExplainer for XGBoost model
- **Outputs**:
  - Top-3 features per class (mean absolute SHAP values)
  - Class-specific feature importance bar charts
  - Summary plot (beeswarm) showing feature impact distribution

## 📈 Key Visualizations

1. **Label Distribution**: Train vs Test split comparison
2. **Model Leaderboard**: AutoGluon model accuracy comparison
3. **Confusion Matrix**: True vs predicted labels heatmap
4. **Feature Importance**: Top 20 features by AutoGluon importance
5. **SHAP Feature Importance**: Top-3 features per class with mean |SHAP| values
6. **SHAP Summary Plot**: Feature impact distribution across samples

## ⚙️ Configuration Parameters

```python
CSV_PATH = "/content/Impact_of_Remote_Work_on_Mental_Health.csv"
TARGET = "Satisfaction_with_Remote_Work"
DROP_COLS = ["Employee_ID"]
TRAIN_FRAC = 0.9
SEED = 42
PRES = "best_quality"
EVAL_METRIC = "accuracy"
```

## 🎓 Key Features

### Advanced Techniques
1. **PySpark Integration**: Scalable data processing for large datasets
2. **SMOTENC**: Handles mixed-type features during oversampling
3. **AutoML**: Automated model selection and hyperparameter tuning
4. **Ensemble Learning**: 5-fold bagging with 1-level stacking
5. **Model Interpretability**: SHAP values for explainable AI

### Robustness
- Automatic column sanitization for special characters
- Comprehensive null handling (median/mode imputation)
- Aligned one-hot encoding between train/test sets
- Category alignment to prevent encoding mismatches

## 📊 Expected Outputs

- **Model Performance**: Classification accuracy, precision, recall, F1-score
- **Feature Insights**: Top features driving satisfaction predictions
- **Class-Specific Analysis**: Which features matter most for each satisfaction level
- **Visualizations**: 7+ charts for comprehensive model understanding

## 🔍 Model Interpretability

The project emphasizes explainability through:
- **AutoGluon Feature Importance**: Permutation-based importance scores
- **SHAP Values**: Additive feature attribution for individual predictions
- **Class-Specific Analysis**: Separate feature rankings for each outcome class

## 📝 Usage

1. **Update the CSV path** in cell 3 to point to your dataset location
2. **Run all cells sequentially** - the notebook is designed for linear execution
3. **Review visualizations** after each major step to understand data and model behavior
4. **Examine SHAP output** for feature interpretability and model insights

## ⚠️ Notes

- **Memory Requirements**: AutoGluon with `best_quality` preset can be memory-intensive
- **Time Limits**: Training is capped at 30 minutes; adjust `time_limit` parameter as needed
- **Google Colab Path**: The CSV path `/content/...` suggests Google Colab usage; update for local environments


**Framework Versions**: AutoGluon (latest), PySpark (latest), XGBoost (latest), SHAP (latest)
