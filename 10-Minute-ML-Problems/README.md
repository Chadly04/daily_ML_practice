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

## Daily routine
1. Attempt the day's problem in VS Code/Jupyter.
2. Save the notebook, including explanations of mistakes and corrections.
3. Restart the kernel and run all cells before committing.
4. Add the notebook here and update the progress table with the actual result and lesson learned.
5. Use descriptive commit messages, such as `Add Day 4 regression practice and lessons`.

**Note:** The synthetic datasets used in these short exercises are small and for learning. Very high test metrics do not establish real-world model performance.
