import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sb
import pickle

# Read in data
df = pd.read_csv('winequality-white.csv', sep=";")

## VIEW CONTENT
grouped_df = df.groupby("quality").sum()
print(grouped_df)


# ploting heatmap
plt.figure(figsize=[19,10],facecolor='white')
sb.heatmap(df.corr(),annot=True)
#plt.show()

# Create new features - Feature Engineering
#df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']

# Create new features - Feature Engineering
#df['alcohol_acidity'] = df['alcohol'] / df['volatile acidity']
#df['chlorides_sulphates'] = df['chlorides'] / df['sulphates']
#df['total_acidity'] = df['fixed acidity'] * df['volatile acidity']

# create a correlation matrix
corr_matrix = df.corr().abs()

# get the upper triangular matrix of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# find the columns with correlation greater than a threshold
threshold = 0.7
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# drop the highly correlated columns from the dataset
df.drop(to_drop, axis=1, inplace=True)

df.isnull().sum()

df.update(df.fillna(df.mean()))

# Split data into features and labels
X = df.drop(['quality'], axis=1)
y = df['quality']

# set the random seed to a fixed value
np.random.seed(50)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Apply SMOTE to balance the classes
smote = SMOTE(k_neighbors = 4)
X_train, y_train = smote.fit_resample(X_train, y_train)


# Define hyperparameters to search over
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]}


# Perform grid search cross-validation to find best hyperparameters
White_model = RandomForestClassifier()
grid_search = GridSearchCV(White_model, param_grid=param_grid, cv = 10)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)


# Train model with best hyperparameters and make predictions
White_model = RandomForestClassifier(class_weight='balanced', **grid_search.best_params_)
White_model.fit(X_train, y_train)
y_pred = White_model.predict(X_test)

# The accuracy score for the white model
White_score = accuracy_score(y_test, y_pred)
print(f'The accuracy for the White model is: {White_score}')

# Evaluate performance with multiple metrics
print("Classification report:\n", classification_report(y_test, y_pred, zero_division=1))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


# Print feature importances
importances = White_model.feature_importances_
for i, feature in enumerate(X.columns):
    print(feature + ":", importances[i])


# Use the cross_val_score function from sklearn.model_selection to evaluate the model using cross-validation
scores = cross_val_score(White_model, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())


# Train model with best hyperparameters and make predictions
White_model = RandomForestClassifier(**grid_search.best_params_)
White_model.fit(X_train, y_train)
y_pred = White_model.predict(X_test)


# Calculate accuracy score and save the model using Pickle
White_score = accuracy_score(y_test, y_pred)
with open('White_model.pkl', 'wb') as file:
    pickle.dump(White_model, file)
with open('White_score.pkl', 'wb') as file:
    pickle.dump(White_score, file)


# Load the model from disk
with open('White_model.pkl', 'rb') as file:
    White_model = pickle.load(file)

# Load the score from disk
with open('White_score.pkl', 'rb') as file:
    White_score = pickle.load(file)

# Test the model with the test data
score = White_model.score(X_test, y_test)

# Compare the saved score with the score calculated from the model
if score == White_score:
    print("The model is correctly fitted and ready for use.")
else:
    print("The model is not correctly fitted or ready for use.")

