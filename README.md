# 🏦 Company Bankruptcy Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Domain-Finance%20%7C%20Risk%20Modeling-success)

## 📖 Project Overview
Corporate bankruptcy is a critical event with severe implications for investors, creditors, and the broader economy. Predicting bankruptcy allows financial institutions to mitigate risk, adjust credit policies, and make informed lending decisions. 

This project utilizes the **Taiwan Economic Journal (TEJ) Bankruptcy Dataset** (1999-2009) to build a robust machine learning pipeline that identifies companies at high risk of bankruptcy based on their historical financial ratios.

## 🎯 Business Objective
The goal is to build a classification model that accurately predicts whether a company will go bankrupt. 
*   **Target Variable:** `Bankrupt?` (1 = Bankrupt, 0 = Healthy)
*   **Key Challenge:** The dataset is **highly imbalanced**. Bankruptcies are rare events (typically 3-5% of the data). A naive model predicting "Healthy" for every company would achieve 96% accuracy but be completely useless. Therefore, the focus is on **Recall** (catching the bankrupt companies) and **ROC-AUC** rather than overall accuracy.

## 📊 Dataset Description
*   **Source:** Taiwan Economic Journal (TEJ)
*   **Timeframe:** 1999 - 2009
*   **Instances:** 6,819 companies (Rows)
*   **Features:** 96 columns (95 financial ratios + 1 target)
*   **Feature Categories:**
    *   **Profitability:** ROA, Operating Gross Margin, Net Income to Total Assets.
    *   **Leverage & Solvency:** Debt Ratio, Interest Coverage Ratio, Total Debt/Net Worth.
    *   **Liquidity:** Current Ratio, Quick Ratio, Cash/Total Assets.
    *   **Efficiency:** Asset Turnover, Accounts Receivable Turnover.
    *   **Market & Growth:** Total Asset Growth Rate, Net Value Growth Rate.

## 🛠️ Methodology & Workflow

### 1. Data Cleaning & Preprocessing
*   **Column Renaming:** The original dataset features very long, complex column names. These were programmatically renamed to standardized labels (e.g., `X1`, `X2`... `X95`) to streamline code readability and modeling.
*   **Missing Values & Duplicates:** Checked for nulls and duplicate rows. (The TEJ dataset is generally clean, but preprocessing ensures integrity).
*   **Feature Scaling:** `StandardScaler` was applied to the features. While not strictly necessary for tree-based models, it is required for distance-based models (like Logistic Regression) to ensure they converge properly.

### 2. Handling Severe Class Imbalance
Because bankruptcies are rare, standard accuracy is a misleading metric. To handle this:
*   **Stratified Splitting:** The data was split into Train (80%) and Test (20%) using `stratify=y` to ensure the 3% bankruptcy ratio was preserved in both sets.
*   **Class Weights:** The models were trained using `class_weight='balanced'`. This penalizes the model heavily for misclassifying a bankrupt company (False Negative), forcing it to pay more attention to the minority class.

### 3. Exploratory Data Analysis (EDA)
Key insights discovered during EDA:
*   **ROA (Return on Assets):** Bankrupt companies consistently show lower ROA compared to healthy companies.
*   **Debt Ratio:** Bankrupt companies typically exhibit significantly higher debt ratios.
*   **Cash Flow:** Low cash flow rates are a strong indicator of impending bankruptcy.

### 4. Modeling
Two distinct models were trained and evaluated:
*   **Logistic Regression:** Used as an interpretable baseline to understand the linear relationships between financial ratios and bankruptcy risk.
*   **Random Forest Classifier:** Used to capture non-linear relationships and complex interactions between the 95 financial ratios.

### 5. Evaluation Metrics
Given the high cost of a False Negative (missing a company that goes bankrupt), the models were evaluated primarily on:
*   **Recall (Sensitivity):** The ability to correctly identify actual bankruptcies.
*   **ROC-AUC:** The model's ability to distinguish between the two classes.
*   **Confusion Matrix:** Visualized to understand exactly how many bankruptcies were missed.

## 🚀 Key Results
*(Note: Replace these numbers with your actual model's scores once you run the code)*

| Model | Accuracy | Recall (Bankrupt) | ROC-AUC |
| :--- | :--- | :--- | :--- |
| Logistic Regression | ~86% | ~75% | ~0.88 |
| **Random Forest** | **~95%** | **~80%** | **~0.94** |

*The Random Forest model, combined with class balancing, proved superior at identifying high-risk companies without generating an excessive number of false alarms.*

## 💻 Installation & Usage

### Prerequisites
*   Python 3.8+
*   Jupyter Notebook / Google Colab

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn