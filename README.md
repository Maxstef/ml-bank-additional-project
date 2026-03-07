# Bank Additional Project
## 1. Problem Description
The final goal is to build a model to predict whether a client will place a term deposit with the bank.
## 2. Dataset
https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv

Original data come from the site [UCI Machine Learning Repository]
## 3. Project Goals
- Explore and understand the dataset
- Perform feature engineering
- Train multiple binary classification models
- Evaluate models using standard metrics (ROC-AUC, precision, recall)
## 4. Project structure
```
ml-bank-additional-project/
│
├── README.md                # Project overview and instructions
├── requirements.txt         # Python dependencies
├── .gitignore
├── data/
│   ├── raw/                 # Original dataset (never modify)
│   └── processed/           # Cleaned/feature-engineered data
│
├── notebooks/               # all notebooks related to experimentation, exploration, modeling, etc
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
## 5. Run Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/Maxstef/ml-bank-additional-project.git
   cd ml-bank-additional-project
   ```

2. **Create a virtual environment** (optional but recommended):
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
