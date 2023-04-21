import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Read in data
df = pd.read_csv('winequality-red.csv', sep=";")

# Split data into features and labels
X = df.drop(['quality'], axis=1)
y = df['quality']

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM model with hyperparameter tuning
svm = SVC(random_state=0)
parameters = {'kernel': ['linear', 'rbf', 'poly'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(estimator = svm, param_grid = parameters, cv = 5, n_jobs = -1)
grid_search.fit(X_train, y_train)

# best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# predict the test set results using the best model
best_svm = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], random_state=0)
best_svm.fit(X_train, y_train)
y_pred = best_svm.predict(X_test)

# evaluate the performance of the model
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Use the cross_val_score function from sklearn.model_selection to evaluate the model using cross-validation
scores = cross_val_score(best_svm, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())
