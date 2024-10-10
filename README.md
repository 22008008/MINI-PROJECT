# Ex.No: 10 Learning – Use Supervised Learning  

NAME : SRI RANJANI PRIYA P
REGISTER NUMBER : 212222220049

# AIM: 
To write a program to train the classifier for Plant Growth Data.
# Algorithm:

Load Data: Load the dataset and split it into features (X) and target (y).
Train-Test Split: Split the data into training and testing sets.
Handle Imbalance: Apply SMOTE to balance the training set.
Preprocess: Use a column transformer to scale numerical features and one-hot encode categorical features.
Pipeline Setup: Build a pipeline with preprocessing and a RandomForestClassifier.
Hyperparameter Tuning: Use GridSearchCV to tune hyperparameters like n_estimators, max_depth, etc.
Model Training: Train the model on the balanced training data.
Evaluate: Predict on test data, and evaluate using accuracy, confusion matrix, and classification report.

# Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('/content/plant_growth_data.csv')

print(data.head())
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pip install pandas scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),          # Scale numeric features
        ('cat', OneHotEncoder(), categorical_cols)        # One-Hot Encode categorical features
    ])

from sklearn.pipeline import Pipeline

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),          # Scale numeric features
        ('cat', OneHotEncoder(), categorical_cols)        # One-Hot Encode categorical features
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

```

### Output:
https://colab.research.google.com/drive/1S8KGNjOaAs_Q--kMDVUIZPAr0NJx_6hq#scrollTo=eH1FtbmB66q2&line=1&uniqifier=1

### Result:
Thus the system was trained successfully and the prediction was carried out.
