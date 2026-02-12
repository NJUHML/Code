# SMmerge

## 1. Introduction

This project produces a daily 1 km resolution surface soil moisture (0–10 cm) dataset across China using an interpretable machine learning-based fusion framework.  
The dataset is publicly accessible at: https://doi.org/10.11888/Terre.tpdc.302923. 

## 2. Required Python Packages

The following Python packages are required to run the code:

| Package       | Version | Documentation / Installation                     |
|---------------|---------|--------------------------------------------------|
| XGBoost       | 2.1.1   | [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/) |
| LightGBM      | 4.5.0   | [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/) |
| CatBoost      | 1.2.7   | [https://catboost.ai/docs/en/](https://catboost.ai/docs/en/) |
| Scikit‑Learn  | 1.2.1   | [https://scikit‑learn.org/stable/](https://scikit‑learn.org/stable/) |
| Optuna        | 3.3.0   | [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/) |
| SHAP          | 0.46.0  | [https://shap.readthedocs.io/](https://shap.readthedocs.io/) |

## 3. Usage Instructions

- **Create database script**: `creat_database.py`  
- **Model training**: `CN1_optuna_train.py`  
- **Hyper-parameter tuning**: `optuna_CN1_model.py`  
- **Feature selection**: `select_featurev3_opt_CN1.py`  

### Visualization

- **Comprehensive model evaluation**: `ModelPreformanceEvaluation.ipynb`  
- **Drought case analysis**: `DroughtCaseAnalysis.ipynb`  

## 4. Citation

If you use this dataset or code in your research, please cite the corresponding data publication using the DOI:  
(https://doi.org/10.11888/Terre.tpdc.302923)
