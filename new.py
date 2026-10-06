import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

data = {
    "age": [22, 35, 41, 29, 52, 46, 31, 27, 60, 38, 44, 25],
    "monthly_fee": [45, 80, 95, 60, 110, 100, 72, 55, 120, 85, 90, 50],
    "months_active": [3, 24, 36, 8, 48, 30, 15, 6, 60, 20, 28, 4],
    "support_tickets": [4, 1, 0, 3, 0, 1, 2, 4, 0, 1, 2, 5],
    "plan_type": [
        "Basic", "Premium", "Premium", "Basic",
        "Premium", "Premium", "Standard", "Basic",
        "Premium", "Standard", "Standard", "Basic"
    ],
    "churned": [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data)
print(df)


# Separate X and y.
X = df. drop(columns=['churned'])
y = df['churned']

##Identify the numeric and categorical features.
num = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat = df.select_dtypes(include=['object']).columns.tolist()


#Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Use StandardScaler on the numeric columns.
#Use OneHotEncoder on plan_type.
#Use a ColumnTransformer.
processor = ColumnTransformer(
    transformers = [
        ("num", standard_scaler, num),
        ("cat", one_hot_encoder, cat)
    ]
#Put the preprocessing and LogisticRegression model into a Pipeline.
#Train the model.
#Make predictions.

# print:
#  - Accuracy
# - Precision
# - Recall
# - F1 score