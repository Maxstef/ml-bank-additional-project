# Bank Additional Project
## 1. Problem Description
The final goal is to build a model to predict whether a client will place a term deposit with the bank.
## 2. Dataset
Dataset source: [Kaggle](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv).

Original data come from the site [UCI Machine Learning Repository]
## 3. Project Goals
- Explore and understand the dataset
- Perform feature engineering
- Train multiple binary classification models
- Evaluate models using standard metrics (ROC-AUC, precision, recall)
- Analyse models features importance, predictions and mistakes
## 4. Project structure
```
ml-bank-additional-project/
│
├── README.md                # Project overview and instructions
├── requirements.txt         # Python dependencies
├── .gitignore               # list of files and dirs ignored by git
├── data/
│   ├── raw/                 # Original dataset (never modify)
│   └── processed/           # Cleaned/feature-engineered data
│
├── notebooks/               # all notebooks related to exploration, modeling, etc
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
|   └──...
│
├── src/                     # reusable Python code
│   ├── data.py
│   ├── utils.py
│   └── ....
│
├── models/                  # saved trained models
|   ├── model.parquet
│   └── model.joblib
...
```

## 5. Results

All experiments, intermediate findings, and detailed explanations are documented in the project notebooks:

* **EDA, initial hypotheses, and metric selection** – [`notebooks/01_eda.ipynb`](https://github.com/Maxstef/ml-bank-additional-project/blob/main/notebooks/01_eda.ipynb)
* **Data preprocessing and feature engineering** – [`notebooks/02_preprocessing.ipynb`](https://github.com/Maxstef/ml-bank-additional-project/blob/main/notebooks/02_preprocessing.ipynb)
* **Model training and comparison** – [`notebooks/03_modeling.ipynb`](https://github.com/Maxstef/ml-bank-additional-project/blob/main/notebooks/03_modeling.ipynb)
* **Feature importance, SHAP, and error analysis** – [`notebooks/04_model_analysis.ipynb`](https://github.com/Maxstef/ml-bank-additional-project/blob/main/notebooks/04_model_analysis.ipynb)
* **Class imbalance handling experiments** – [`notebooks/05_class_imbalance.ipynb`](https://github.com/Maxstef/ml-bank-additional-project/blob/main/notebooks/05_class_imbalance.ipynb)

Dataset source: [Kaggle](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv) (originally from UCI Machine Learning Repository).

### Best Model Performance (Validation)

| Model Name          | Validation F1 | Validation AUROC | Train F1 | Train AUROC |
| ------------------- | ------------- | ---------------- | -------- | ----------- |
| Random Forest       | **0.535**     | 0.806            | 0.536    | 0.930       |
| XGBoost             | **0.534**     | **0.807**        | 0.518    | 0.841       |
| LightGBM            | 0.528         | 0.804            | 0.506    | 0.827       |
| AdaBoost            | 0.523         | 0.801            | 0.498    | 0.789       |
| Decision Tree       | 0.522         | 0.799            | 0.505    | 0.794       |
| Logistic Regression | 0.507         | 0.801            | 0.484    | 0.792       |
| KNN                 | 0.406         | 0.744            | 0.491    | 0.919       |

### Key Outcomes

* Tree-based ensemble models clearly outperform linear and distance-based approaches.
* Random Forest achieved the highest F1 score but showed signs of overfitting.
* XGBoost provided the best balance between F1, AUROC, and generalization.
* Logistic Regression remained stable but was less capable of capturing complex patterns.

Final trained models and preprocessing pipelines are stored in the `/models` directory.

## 6. Run Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/Maxstef/ml-bank-additional-project.git
   cd ml-bank-additional-project
   ```

2. **Create a virtual environment** (recommended python version - 3.11.11):
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows `venv\Scripts\activate`
   ```
3. **Install packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
## 7. Conclusions

The initial EDA hypotheses were validated and refined through modeling, feature importance, SHAP analysis, class-imbalance experiments, and error analysis.

### Key drivers of predictions

The most influential factors across models are:

* **Macroeconomic indicators** (`nr.employed`, `emp.var.rate`, `cons.conf.idx`, `cons.price.idx`)
* **Previous campaign interactions** (`pdays`, `poutcome`, `campaign`)
* **Temporal effects** (month)
* **Contact type**

Client demographics (age, job, marital, education) are relevant but **secondary**.

---

### Model insights

* Tree-based ensembles (Random Forest, XGBoost, LightGBM) clearly outperform linear and distance-based models.
* XGBoost provides the best balance between performance and generalization.
* SHAP confirms both feature importance and directional impact on predictions.

---

### Class imbalance findings

* Resampling methods did not improve results.
* Best approach: **class weighting / `scale_pos_weight ≈ 4`** and **threshold tuning**.
* Performance differences are mostly **threshold-driven** (AUROC remains stable).

---

### Error patterns & business view

* Errors arise from how models combine multiple weak signals rather than single features.
* Improving **recall for the `yes` class** is important for real campaign scenarios.

**Practical guidance:**

* Higher precision → raise threshold, use Logistic Regression / Random Forest
* Higher recall → lower threshold, use XGBoost

## 8. Future Improvements

Potential directions for extending and improving this project include:

* Treat the dataset as a **time-aware problem**

  * Data is naturally ordered by time (month available)
  * Introduce a proper **time-based train/test split**
  * Potentially reconstruct yearly or seasonal trends

* Improve model performance tuning

  * Systematic **threshold optimization** for business objectives
  * **Probability calibration** (Platt scaling / isotonic regression)

* Extend modeling approaches

  * Experiment with **stacking/ensembles** (e.g., XGBoost + Logistic Regression)
  * **More extensive Optuna-based hyperparameter optimization** (larger search space and more trials, balancing F1, recall, and precision)
  * Explore cost-sensitive or business-driven learning objectives

* Improve evaluation strategy

  * Introduce **cost-sensitive evaluation** (false positive vs false negative trade-off)
  * Focus on business-oriented metrics beyond F1 (e.g., recall-driven or profit-based metrics)

* Feature engineering enhancements

  * Deeper temporal features (seasonality, rolling statistics)
  * Additional interaction features between economic indicators and campaign variables
