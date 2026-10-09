rameterTuning_GridSearchCV.ipynb===
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "86c861f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": "     mean radius  mean texture  ...  worst fractal dimension  target\n0          17.99         10.38  ...                  0.11890       0\n1          20.57         17.77  ...                  0.08902       0\n2          19.69         21.25  ...                  0.08758       0\n3          11.42         20.38  ...                  0.17300       0\n4          20.29         14.34  ...                  0.07678       0\n..           ...           ...  ...                      ...     ...\n564        21.56         22.39  ...                  0.07115       0\n565        20.13         28.25  ...                  0.06637       0\n566        16.60         28.08  ...                  0.07820       0\n567        20.60         29.33  ...                  0.12400       0\n568         7.76         24.54  ...                  0.07039       1\n\n[569 rows x 31 columns]\n"
    }
   ],
   "source": [
    "from sklearn.datasets import load_breast_cancer\n",
    "\n",
    "data = load_breast_cancer(as_frame=True)\n",
    "df = data.frame\n",
    "print(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7866fddc",
   "metadata": {},
   "source": [
    "Explore the dataset with head(), shape, missing values, and the target distribution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "aedbcbbf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": "   mean radius  mean texture  ...  worst fractal dimension  target\n0        17.99         10.38  ...                  0.11890       0\n1        20.57         17.77  ...                  0.08902       0\n2        19.69         21.25  ...                  0.08758       0\n3        11.42         20.38  ...                  0.17300       0\n4        20.29         14.34  ...                  0.07678       0\n\n[5 rows x 31 columns]"
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9aee9327",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": "(569, 31)"
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7b351e5b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": "mean radius                0\nmean texture               0\nmean perimeter             0\nmean area                  0\nmean smoothness            0\nmean compactness           0\nmean concavity             0\nmean concave points        0\nmean symmetry              0\nmean fractal dimension     0\nradius error               0\ntexture error              0\nperimeter error            0\narea error                 0\nsmoothness error           0\ncompactness error          0\nconcavity error            0\nconcave points error       0\nsymmetry error             0\nfractal dimension error    0\nworst radius               0\nworst texture              0\nworst perimeter            0\nworst area                 0\nworst smoothness           0\nworst compactness          0\nworst concavity            0\nworst concave points       0\nworst symmetry             0\nworst fractal dimension    0\ntarget                     0\ndtype: int64"
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4df5d6be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": "target\n1    357\n0    212\nName: count, dtype: int64"
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"target\"].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e64d9f85",
   "metadata": {},
   "source": [
    "Create X and y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "35b05cc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(columns=\"target\")\n",
    "y = df[\"target\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d8f1c6b0",
   "metadata": {},
   "source": [
    "Split the data into 80% training / 20% testing using random_state=42 and stratify=y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "916c2588",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d1682698",
   "metadata": {},
   "source": [
    "Create a pipeline containing:\n",
    "- StandardScaler\n",
    "- LogisticRegression(max_iter=5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "77412a0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "scaler = StandardScaler()\n",
    "log_reg = LogisticRegression(max_iter=5000)\n",
    "\n",
    "pipeline = Pipeline([\n",
    "    ('scaler', scaler),\n",
    "    ('log_reg', log_reg)\n",
    "])\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b576fa83",
   "metadata": {},
   "source": [
    "First train the pipeline using Logistic Regression's default C value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "0541b212",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": "Pipeline(steps=[('scaler', StandardScaler()),\n                ('log_reg', LogisticRegression(max_iter=5000))])"
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pipeline.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b5e8082",
   "metadata": {},
   "source": [
    "Record its test accuracy, precision, recall, and F1 score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "66e886ee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": "Accuracy: 0.98\nPrecision: 0.99\nRecall: 0.99\nF1 Score: 0.99\n"
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "\n",
    "accuracy = accuracy_score(y_test, pipeline.predict(X_test))\n",
    "precision = precision_score(y_test, pipeline.predict(X_test))\n",
    "recall = recall_score(y_test, pipeline.predict(X_test))\n",
    "f1 = f1_score(y_test, pipeline.predict(X_test))\n",
    "\n",
    "print(f\"Accuracy: {accuracy:.2f}\")\n",
    "print(f\"Precision: {precision:.2f}\")\n",
    "print(f\"Recall: {recall:.2f}\")      \n",
    "print(f\"F1 Score: {f1:.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e4acaa2d",
   "metadata": {},
   "source": [
    "Now use GridSearchCV with 5-fold cross-validation to test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "89e14065",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": "GridSearchCV(cv=5,\n             estimator=Pipeline(steps=[('scaler', StandardScaler()),\n                                       ('log_reg',\n                                        LogisticRegression(max_iter=5000))]),\n             param_grid={'log_reg__C': [0.01, 0.1, 1, 10, 100]},\n             scoring='accuracy')"
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "C_values = [0.01, 0.1, 1, 10, 100]\n",
    "\n",
    "param_grid = {\"log_reg__C\": C_values} # __ conects the pipeline step name with the parameter name\n",
    "\n",
    "grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')\n",
    "\n",
    "grid_search.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0b13290c",
   "metadata": {},
   "source": [
    "Find the best C."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "399134c6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": "Best C: 0.1\nBest Cross-Validation Accuracy: 0.9802197802197803\n"
    }
   ],
   "source": [
    "print(\"Best C:\", grid_search.best_params_[\"log_reg__C\"])\n",
    "print(\"Best Cross-Validation Accuracy:\", grid_search.best_score_)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1e304849",
   "metadata": {},
   "source": [
    "Evaluate the best model on the untouched test set using the same four metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "bab34970",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": "Tuned Model\nAccuracy: 0.97\nPrecision: 0.97\nRecall: 0.99\nF1 Score: 0.98\n"
    }
   ],
   "source": [
    "best_model = grid_search.best_estimator_\n",
    "\n",
    "tuned_predictions = best_model.predict(X_test)\n",
    "\n",
    "tuned_accuracy = accuracy_score(y_test, tuned_predictions)\n",
    "tuned_precision = precision_score(y_test, tuned_predictions)\n",
    "tuned_recall = recall_score(y_test, tuned_predictions)\n",
    "tuned_f1 = f1_score(y_test, tuned_predictions)\n",
    "\n",
    "print(\"Tuned Model\")\n",
    "print(f\"Accuracy: {tuned_accuracy:.2f}\")\n",
    "print(f\"Precision: {tuned_precision:.2f}\")\n",
    "print(f\"Recall: {tuned_recall:.2f}\")\n",
    "print(f\"F1 Score: {tuned_f1:.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29c6b79f",
   "metadata": {},
   "source": [
    "Compare the tuned model against your original model\n",
    "\n",
    "GridSearchCV selected C = 0.1. The tuned model had slightly lower test accuracy than the default model, showing that the best average cross-validation setting does not always produce a higher score on one test split."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ebbf7335",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": "Default Model\nAccuracy: 0.98\nPrecision: 0.99\nRecall: 0.99\nF1 Score: 0.99\n\nBest C: 0.1\n\nTuned Model\nAccuracy: 0.97\nPrecision: 0.97\nRecall: 0.99\nF1 Score: 0.98\n"
    }
   ],
   "source": [
    "print(\"Default Model\")\n",
    "print(f\"Accuracy: {accuracy:.2f}\")\n",
    "print(f\"Precision: {precision:.2f}\")\n",
    "print(f\"Recall: {recall:.2f}\")\n",
    "print(f\"F1 Score: {f1:.2f}\")\n",
    "\n",
    "print(f\"\\nBest C: {grid_search.best_params_['log_reg__C']}\")\n",
    "\n",
    "print(\"\\nTuned Model\")\n",
    "print(f\"Accuracy: {tuned_accuracy:.2f}\")\n",
    "print(f\"Precision: {tuned_precision:.2f}\")\n",
    "print(f\"Recall: {tuned_recall:.2f}\")\n",
    "print(f\"F1 Score: {tuned_f1:.2f}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
===FILE:10-Minute-ML-Problems/README.md===
# 10-Minute ML Problems

Short Python challenges focused on practical machine learning and AI engineering. I attempt the problems first, review errors, and document what I learn. These are separate from the larger ML projects in this repository.

| Day | Problem | Status | Key learning |
| --- | --- | --- | --- |
| 1 | [Logistic Regression: Customer Churn](Day01_LogisticRegression.ipynb) | Completed | Separate features and target, scale and encode with a pipeline, evaluate class predictions without unnecessary rounding. |
| 2 | [Data Cleaning and Feature Engineering](Day02_DataPreprocessing.ipynb) | Completed (reconstructed from conversation; original not yet supplied) | Impute numerical columns using the median, categorical values using the mode, create `tickets_per_month`, and verify missing values. |
| 3 | [Decision Tree: Loan Approval](Day03_DecisionTree.ipynb) | Completed | Train and evaluate a classifier without a pipeline; understand why perfect scores on tiny test sets can be misleading. |
| 4 | [NumPy: Feature Engineering](Day04_NumPy_FeatureEngineering.ipynb) | Completed | Select columns with slicing, perform vectorized math, use `np.column_stack()` to add two features and check the `(6, 5)` shape. |
| 5 | [Pandas: GroupBy and Churn Analysis](Day05_Pandas_GroupBy_ChurnAnalysis.ipynb) | Completed | Calculate overall and per-plan averages, derive churn rate with counts and the mean of binary labels, and format a grouped Series as percentages. |
| 6 | [Preprocessing Pipeline](Day06_Preprocessing_Pipeline.ipynb) | Completed | Build separate numeric and categorical preprocessing pipelines, impute missing values, scale and one-hot encode features, combine them with a `ColumnTransformer`, and train Logistic Regression end to end. |
| 7 | [Cross-Validation and Model Comparison](Day07_CrossValidation_ModelComparison.ipynb) | Completed | Use 5-fold cross-validation to evaluate Logistic Regression and a Decision Tree across different portions of the data, compare their average accuracy, and understand why repeated evaluation is more reliable than one tiny split. |
| 8 | [Hyperparameter Tuning with GridSearchCV](Day08_HyperparameterTuning_GridSearchCV.ipynb) | Completed | Keep the test set untouched, use 5-fold cross-validation to tune Logistic Regression's `C`, and compare the tuned pipeline against the default model. |

## Day 2: Work completed
- Filled missing ages and monthly fees using column medians.
- Filled missing plan types using the most common category (`mode()[0]`).
- Added `tickets_per_month = support_tickets / months_active`.
- Verified that all columns had zero missing values.

The Day 2 notebook was **reconstructed from the code and problem shared in chat**. It is clearly labeled and can be replaced with the original VS Code notebook if available.

## Day 4: Work completed
- Calculated `tickets_per_month` and `estimated_total_spent` without using loops.
- Learned to use NumPy indexing, inspect array dimensions with `.shape`, and combine new features using `np.column_stack()` instead of an array `.append()` method.
- Created a final `(6, 5)` NumPy array.
- Reflected on how ticket frequency might give a model more useful churn-related information than ticket counts alone.

## Day 5: Work completed
- Calculated overall and per-plan average monthly fees and support tickets using pandas.
- Computed the overall churn rate both as `sum() / count()` and as `mean()`, obtaining 37.50%.
- Used `groupby("plan")["churned"].mean()` to get Basic (66.67%), Premium (0.00%), and Standard (50.00%) churn rates.
- Formatted each grouped churn rate using `.map('{:.2%}'.format)` without changing the underlying numeric calculation.
- Reviewed and corrected spelling and phrasing in the notebook's Markdown cells.

## Day 6: Work completed
- Split features and target, then created train and test sets.
- Selected numerical and categorical columns by data type.
- Used `SimpleImputer(strategy="median")` and `StandardScaler()` for numerical data.
- Used `SimpleImputer(strategy="most_frequent")` and `OneHotEncoder(handle_unknown="ignore")` for categorical data.
- Combined both preprocessing branches with `ColumnTransformer` and placed them with `LogisticRegression()` in one pipeline.
- The tiny 2-row test set produced 0.00 accuracy after median imputation; the key lesson is that correct code can still give unstable metrics on very small datasets.

## Day 7: Work completed
- Separated the feature columns from the `churned` target.
- Created Logistic Regression and Decision Tree models.
- Used `cross_val_score()` with 5 folds and accuracy scoring instead of one train/test split.
- Compared all five scores and the average accuracy for each model.
- Both models earned 1.00 average accuracy on the small dataset.
- Learned that testing on different portions of the data gives a more reliable performance estimate than testing only once.

## Day 8: Work completed
- Explored the breast cancer dataset and checked its shape, missing values, and target distribution.
- Used a stratified 80/20 train/test split and placed `StandardScaler` with Logistic Regression inside a pipeline.
- Evaluated the default model with accuracy, precision, recall, and F1 score.
- Used `GridSearchCV` with 5-fold cross-validation to test `C` values of 0.01, 0.1, 1, 10, and 100.
- Found `C=0.1` performed best, with about 98.02% average cross-validation accuracy.
- The default model scored about 98.25% test accuracy, while the tuned model scored about 97.37%; learned that the best average cross-validation setting does not always score higher on one test split.

## Daily routine
1. Attempt the day's problem in VS Code/Jupyter.
2. Save the notebook, including explanations of mistakes and corrections.
3. Restart the kernel and run all cells before committing.
4. Add the notebook here and update the progress table with the actual result and lesson learned.
5. Use descriptive commit messages, such as `Add Day 4 regression practice and lessons`.

**Note:** The synthetic datasets used in these short exercises are small and for learning. Very high test metrics do not establish real-world model performance.
===FILE:README.md===
# Daily Machine Learning Practice

This repository is a machine learning practice project where I am learning how to clean data, split data into training and testing sets, train models, evaluate their performance, and explain the results. The practice problems and synthetic datasets are created with ChatGPT, while I complete the coding, analysis, and model decisions myself unless otherwise noted.

Instead of copying completed solutions, I attempt each problem on my own first. I then review my mistakes, correct the notebook, and record what I learned.

## Use of AI

ChatGPT is used to generate practice scenarios, synthetic datasets, concept questions, and feedback. It is used as a tutor rather than as a replacement for completing the work. My notebooks show my own attempts, corrections, explanations, and final results.

## Goals

- Practice Python, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn
- Understand when different machine learning models should be used
- Build complete workflows from data exploration through model evaluation
- Improve at explaining results in clear language
- Create polished projects that demonstrate consistent progress

## 10-Minute ML Problems

I also complete short, focused daily exercises to build practical ML/AI engineering skills. [View the 10-minute problem tracker and notebooks](10-Minute-ML-Problems/README.md).

| Day | Problem | Status |
| ---: | --- | --- |
| 1 | Logistic Regression — Small Customer Churn Dataset | Completed |
| 2 | Missing Data and Feature Engineering | Completed (reconstructed notebook available) |
| 3 | Decision Tree — Small Loan Approval Dataset | Completed |
| 4 | NumPy — Feature Engineering and Array Stacking | Completed |
| 5 | Pandas — GroupBy and Customer Churn Analysis | Completed |
| 6 | Scikit-learn — Missing Data and Preprocessing Pipeline | Completed |
| 7 | Cross-Validation — Logistic Regression vs. Decision Tree | Completed |
| 8 | Hyperparameter Tuning — Logistic Regression with GridSearchCV | Completed |

These are **separate from the larger projects and roadmap below**, and I update the tracker as I finish each new daily problem.

## Larger Project Practice Roadmap

| Day | Model or topic | Example problem | Status |
| ---: | --- | --- | --- |
| 1 | Logistic Regression | Customer churn prediction | In progress |
| 2 | Decision Tree | Loan approval classification | Not started |
| 3 | Random Forest | Employee attrition prediction | Not started |
| 4 | Linear Regression | House-price prediction | Not started |
| 5 | K-Nearest Neighbors | Customer category prediction | Not started |
| 6 | K-Means Clustering | Customer segmentation | Not started |
| 7 | Naive Bayes | Spam-message detection | Not started |
| 8 | Support Vector Machine | Risk classification | Not started |
| 9 | Gradient Boosting | Customer purchase prediction | Not started |
| 10 | Model Comparison | Compare multiple models | Not started |

## Day 1: Customer Churn Prediction

This larger project is a binary classification task. The goal is to predict whether a subscription customer will leave the company.

- **Target:** `churned`
- **Class 0:** Customer stayed
- **Class 1:** Customer left
- **Dataset size:** 800 customers
- **Main challenges:** Missing values, categorical features, class imbalance, and selecting appropriate evaluation metrics

### Day 1 workflow

1. Load and inspect the dataset.
2. Identify numerical and categorical features.
3. Check for missing values and duplicate rows.
4. Examine the target distribution.
5. Create visualizations to investigate possible churn patterns.
6. Remove the customer ID from the model features.
7. Prepare the data using a preprocessing pipeline.
8. Establish a baseline model.
9. Train and evaluate Logistic Regression.
10. Explain the results and record what was learned.

### Day 1 concept questions

Answer these questions after completing the exploratory analysis:

1. Is this a classification, regression, or clustering problem? Why?
2. Is the target reasonably balanced, or could accuracy be misleading?
3. Which column should probably be removed before training? Explain why.
4. Which columns require missing-value handling?
5. Identify the numerical and categorical features.
6. What patterns would you investigate to understand why customers churn?

## Evaluation Metrics

Depending on the problem, I will practice using:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- ROC-AUC
- Mean absolute error
- Root mean squared error
- R-squared
- Silhouette score

Accuracy will not automatically be treated as the best metric. The metric should match the type of problem and the cost of different prediction errors.

## Daily Workflow

For each practice problem, I will:

1. Read the problem and identify the ML task.
2. Attempt the code and concept questions independently.
3. Explain my preprocessing and model choices.
4. Review feedback and correct my mistakes.
5. Run the notebook from beginning to end.
6. Write a short summary of the results and lessons learned.
7. Commit the completed work to GitHub.

## Repository Organization

Larger practice projects have their own folders. The shorter daily exercises live in `10-Minute-ML-Problems/`:

```text
daily_ML_practice/
  README.md
  10-Minute-ML-Problems/
    README.md
    Day01_LogisticRegression.ipynb
    Day02_DataPreprocessing.ipynb
    Day03_DecisionTree.ipynb
    Day04_NumPy_FeatureEngineering.ipynb
    Day05_Pandas_GroupBy_ChurnAnalysis.ipynb
    Day06_Preprocessing_Pipeline.ipynb
    Day07_CrossValidation_ModelComparison.ipynb
    Day08_HyperparameterTuning_GridSearchCV.ipynb
  day-01-customer-churn/
    01-exploratory-analysis.ipynb
    02-logistic-regression.ipynb
    customer_churn_practice.csv
  day-02-LoanApproval/
    decision-tree-loan-approval.ipynb
    loan_data.csv
```

## Setup

The projects use Python 3 and Jupyter notebooks. Install the main packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Then open the project in VS Code with the Python and Jupyter extensions, or start Jupyter Notebook with:

```bash
jupyter notebook
```

## Author

Chadly Fleming  
Computer Science student interested in machine learning, artificial intelligence, and computer vision.