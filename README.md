# Ex.No: 10 Plant Growth Data classification– Machine Learning  

NAME : SRI RANJANI PRIYA P
REGISTER NUMBER : 212222220049

# AIM: 
To write a program to train the classifier for Plant Growth Data.

# Algorithm:

1.Load Data: Load the dataset and split it into features (X) and target (y).
2.Train-Test Split: Split the data into training and testing sets.
3.Handle Imbalance: Apply SMOTE to balance the training set.
4.Preprocess: Use a column transformer to scale numerical features and one-hot encode categorical features.
5.Pipeline Setup: Build a pipeline with preprocessing and a RandomForestClassifier.
6.Hyperparameter Tuning: Use GridSearchCV to tune hyperparameters like n_estimators, max_depth, etc.
7.Model Training: Train the model on the balanced training data.
8.Evaluate: Predict on test data, and evaluate using accuracy, confusion matrix, and classification report.

# Program:
```

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


classifiers = {
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    })}


for name, (clf, params) in classifiers.items():
    grid_search = GridSearchCV(clf, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    y_pred = grid_search.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name}: Best Params: {grid_search.best_params_}, Accuracy: {accuracy:.4f}")

```
# Classification Figures:
```
https://github.com/user-attachments/assets/f3c332cc-4cc4-4a23-bb50-d3b2f3173a49

![image](https://github.com/user-attachments/assets/30191d6b-beed-4a4a-b11f-0801999600c3)

![image](https://github.com/user-attachments/assets/8167ba3f-3306-4640-83cd-ee5c0db36e6d)

![image](https://github.com/user-attachments/assets/f1b08fe2-0b80-434e-9211-281bf1af92af)

![image](https://github.com/user-attachments/assets/d63982d1-2da3-4152-bac5-51fd9e10affc)

![image](https://github.com/user-attachments/assets/b1b29ac1-1384-484b-a4e3-129e1293153c)

![image](https://github.com/user-attachments/assets/53510f14-7631-486e-a419-abfea4a14238)
```

# Output:
```
                           
![image](https://github.com/user-attachments/assets/0819a527-61ec-4290-9cd5-ec6140c2b812)

```
# Result:
Thus the system was trained successfully and the prediction was carried out.
