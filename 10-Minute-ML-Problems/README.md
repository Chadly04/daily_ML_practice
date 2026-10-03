# 10-Minute ML Problems

Short Python challenges focused on practical machine learning and AI engineering. I attempt the problems first, review errors, and document what I learn. These are separate from the larger ML projects in this repository.

| Day | Problem | Status | Key learning |
| --- | --- | --- | --- |
| 1 | [Logistic Regression: Customer Churn](Day01_LogisticRegression.ipynb) | Completed | Separate features and target, scale and encode with a pipeline, evaluate class predictions without unnecessary rounding. |
| 2 | [Data Cleaning and Feature Engineering](Day02_DataPreprocessing.ipynb) | Completed (reconstructed from conversation; original not yet supplied) | Impute numerical columns using the median, categorical values using the mode, create `tickets_per_month`, and verify missing values. |
| 3 | [Decision Tree: Loan Approval](Day03_DecisionTree.ipynb) | Completed | Train and evaluate a classifier without a pipeline; understand why perfect scores on tiny test sets can be misleading. |
| 4 | [NumPy: Feature Engineering](Day04_NumPy_FeatureEngineering.ipynb) | Completed | Select columns with slicing, perform vectorized math, use `np.column_stack()` to add two features and check the `(6, 5)` shape. |

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

## Daily routine
1. Attempt the day's problem in VS Code/Jupyter.
2. Save the notebook, including explanations of mistakes and corrections.
3. Restart the kernel and run all cells before committing.
4. Add the notebook here and update the progress table with the actual result and lesson learned.
5. Use descriptive commit messages, such as `Add Day 4 regression practice and lessons`.

**Note:** The synthetic datasets used in these short exercises are small and for learning. Very high test metrics do not establish real-world model performance.
