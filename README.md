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
